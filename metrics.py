import os
import json
import subprocess
import argparse

import numpy as np
from scipy.stats import kendalltau
import spacy
import smatch
from torchmetrics.text.rouge import ROUGEScore
import corenlp

import sys
sys.path.insert(0, 'SummEval/evaluation')
from summ_eval.rouge_metric import RougeMetric
sys.path.insert(0, 'transition-amr-parser')
from transition_amr_parser.parse import AMRParser

from data import read_data, linearize_dependency, parse_amr, exists, save, load
from resources import temp_output_f, temp_target_f

def score_sembleu(candidates, targets):
    print('Scoring sembleu...')
    scores = []
    skipped = 0
    for c, t in zip(candidates, targets):
        if c is None or t is None:
            scores.append(None)
            skipped += 1
            continue
        with open(temp_output_f, 'w') as outf, open(temp_target_f, 'w') as targf:
            outf.write(c)
            targf.write(t)
        output = subprocess.run(['./sembleu/eval.sh', temp_output_f, temp_target_f],
                                    capture_output=True, encoding='utf-8').stdout
        output = output.strip().split('\n')
        try:
            assert len(output) == 4
        except:
            scores.append(None)
            skipped += 1
            continue
        output = output[2]
        scores.append(float(output))
    assert len(scores) == len(candidates)
    print(f'...scored {len(scores)} pairs; skipped {skipped}.')
    return scores

def score_smatch(candidates, targets):
    print('Scoring smatch...')
    blank = []
    with open(temp_output_f, 'w') as outf, open(temp_target_f, 'w') as targf:
        i = -1
        for c, t in zip(candidates, targets):
            i += 1
            if c is None or t is None:
                outf.write('(p / placeholder)\n\n')
                targf.write('(p / placeholder)\n\n')
                blank.append(i)
                continue
            outf.write(c.strip() + '\n\n')
            targf.write(t.strip() + '\n\n')
    args = ['smatch.py', '--ms', '--significant', '4', '-f', temp_output_f, temp_target_f]
    output = subprocess.run(args, text=True, capture_output=True)
    scores = [x.strip() for x in output.stdout.strip().split('\n')]
    scores = [float(x.split('F-score: ')[-1]) for x in scores]
    scores = [s if i not in blank else None for i, s in enumerate(scores)]
    assert len(scores) == len(candidates)
    print(f'...scored {len(scores)} pairs; skipped {len(blank)}.')
    return scores

def score(outputs, targets):
    print('Scoring AMR metrics...')
    smatch = {'smatch': score_smatch(outputs, targets)}
    sembleu = {'sembleu': score_sembleu(outputs, targets)}

    return smatch, sembleu

def score_rouge(outputs, targets, implementation='textmetrics', pretokenize=False):
    '''
    Collects only ROUGE scores.
    '''
    if implementation == 'summeval':
        rouge = RougeMetric()
        if pretokenize:
            with corenlp.CoreNLPClient(annotators='tokenize') as client:
                targets = [' '.join([w.word for w in client.annotate(t).sentencelessToken]) for t in targets]
        #scores = rouge.evaluate_batch(outputs, [[t] for t in targets])['rouge']
        scores = {}
        for o, t in zip(outputs, targets):
            single = rouge.evaluate_example(o, t)['rouge']
            for m in single:
                if m not in scores:
                    scores[m] = []
                scores[m].append(single[m])
        scores = {m: [scores[m]] for m in scores}
        return scores
    
    rouge = ROUGEScore()
    scores = {}
    i = 1
    
    for o, t in zip(outputs, targets):
        single = rouge(o, t)
        for k in single:
            if k not in scores:
                scores[k] = []
            scores[k].append(single[k].item())
        if i % 100 == 0:
            print(f'{i} of {len(outputs)} decoded...')
        i += 1
    return scores

def print_banner(s):
    print('--------------------------------')
    print(s)
    print('--------------------------------')

def print_table(t):
    cols = [str(k) for k in t.keys()]
    rows = [str(k) for k in t[cols[0]].keys()]

    print(f'{"(metric/aspect)":20}' + '\t  ' + '  '.join(cols))
    for r in rows:
        print(f'{r:24}' + '\t' + '\t'.join([f'{t[c][r].statistic:0.3f}' for c in cols]))

    print()

def print_correlations(first, second):
    '''
    Compute and print correlations between two dictionaries of scores.
    '''
    best = {}
    fbest = {}

    correlations = {}
    
    for metric in first:
        best[metric] = None
        fbest[metric] = None
        correlations[metric] = {}
        for second_metric in second:
            if second_metric.endswith('_cb') or second_metric.endswith('_ce'):
                continue
            #corr = np.corrcoef(first[metric], second[second_metric])[0][-1]
            try:
                assert len(first[metric]) == len(second[second_metric])
            except:
                print(f'Failed on length check: metric {metric} has {len(first[metric])} elements and metric {second_metric} has {len(second[second_metric])} elements.')
                exit(0)
            l = len(first[metric])
            filtered_first = [first[metric][i] for i in range(l) if first[metric][i] is not None and second[second_metric][i] is not None]
            filtered_second = [second[second_metric][i] for i in range(l) if first[metric][i] is not None and second[second_metric][i] is not None]
            assert len(filtered_first) == len(filtered_second)
            corr = kendalltau(filtered_first, filtered_second)
            if best[metric] == None or corr > best[metric][1]:
                best[metric] = (second_metric, corr)
            #if second_metric.endswith('fmeasure') and (fbest[metric] == None or corr > fbest[metric][1]):
            if second_metric.endswith('f_score') and (fbest[metric] == None or corr > fbest[metric][1]):
                fbest[metric] = (second_metric, corr)
            correlations[metric][second_metric] = corr
            #print(f'Correlation between {metric} and {second_metric}: {corr}')
        #print()

    print_table(correlations)

    #print(best)
    #print()
    #print(fbest)

def aggregate(scores):
    '''
    Aggregates a dict of metric scores into system-level and individual-level scores.
    '''
    keys = [x for x in scores.keys()]
    keys.sort()
    if type(scores[keys[0]]) == dict:
        metrics = [k for k in scores[keys[0]].keys()]
        metrics.sort()
        model_level = {metric: [np.mean([x for x in scores[m][metric] if x is not None]) for m in keys] for metric in metrics}
        total = {}
        for metric in metrics:
            if metric not in total:
                total[metric] = []
            for model in keys:
                if len(scores[model][metric]) == 1:
                    if type(scores[model][metric]) == list:
                        total[metric] += scores[model][metric][0]
                    else:
                        assert type(scores[model][metric]) == dict
                        key = [k for k in scores[model][metric].keys()][0]
                        total[metric] += scores[model][metric][key]
                else:
                    total[metric] += scores[model][metric]
        return model_level, total
    
    model_level = {m: np.mean(scores[m]) for m in scores}
    total = []
    for m in scores:
        total += scores[m]

    return model_level, total

def score_all(dataset, output_base,
                  parser_checkpoint='DATA/AMR3.0/models/amr3.0-structured-bart-large-neur-al/seed42/checkpoint_wiki.smatch_top5-avg.pt',
                  multi_references=True, overwrite=False):
    '''
    Scoring pipeline.
    '''
    output_dir = os.path.join(output_base, dataset)
    os.makedirs(os.path.join(args.output, args.dataset), exist_ok=True)

    # read in initial data and human scores
    fields = ['outputs', 'targets', 'scores']
    if overwrite or not exists(output_dir, fields):
        outputs, targets, scores = read_data(args.dataset)
        if not multi_references:
            targets = {m: [t[0] for t in targets[m]] for m in targets}
        save([outputs, targets, scores], output_dir, fields)
    else:
        outputs, targets, scores = load(output_dir, fields)

    # get dependency linearizations
    fields = ['outputs_dep', 'targets_dep']
    if overwrite or not exists(output_dir, fields):
        print('Linearizing dependencies...')
        nlp = spacy.load('en_core_web_trf')
        outputs_dep = {m: linearize_dependency(outputs[m], nlp) for m in outputs}
        targets_dep = {m: linearize_dependency(targets[m], nlp) for m in targets}
        save([outputs_dep, targets_dep], output_dir, fields)
    else:
        outputs_dep, targets_dep = load(output_dir, fields)

    # get amr
    fields = ['outputs_amr', 'targets_amr']
    if overwrite or not exists(output_dir, fields):
        print('Parsing AMR...')
        parser = None
        if parser_checkpoint:
            parser = AMRParser.from_checkpoint(parser_checkpoint)
        outputs_amr = {m: parse_amr(outputs[m], parser) for m in outputs}
        targets_amr = {m: parse_amr(targets[m], parser) for m in targets}
        save([outputs_amr, targets_amr], output_dir, fields)
    else:
        outputs_amr, targets_amr = load(output_dir, fields)
        
    # calculate rouge scores
    fields = ['rouge_scores']
    if overwrite or not exists(output_dir, fields):
        print('Scoring ROUGE...')
        rouge_scores = {m: score_rouge(outputs[m], targets[m]) for m in outputs}
        save([rouge_scores], output_dir, fields)
    else:
        rouge_scores = load(output_dir, fields)[0]

    # calculate amr scores
    fields = ['smatch_dep', 'sembleu_dep']
    if overwrite or not exists(output_dir, fields):
        smatch_dep = {}
        sembleu_dep = {}
        for m in outputs_dep:
            smatch_dep[m], sembleu_dep[m] = score(outputs_dep[m], targets_dep[m])
        save([smatch_dep, sembleu_dep], output_dir, fields)
    else:
        smatch_dep, sembleu_dep = load(output_dir, fields)

    # calculate amr scores
    fields = ['smatch_amr', 'sembleu_amr']
    if overwrite or not exists(output_dir, fields):
        smatch_amr = {}
        sembleu_amr = {}
        for m in outputs_amr:
            smatch_amr[m], sembleu_amr[m] = score(outputs_amr[m], targets_amr[m])
        save([smatch_amr, sembleu_amr], output_dir, fields)
    else:
        smatch_amr, sembleu_amr = load(output_dir, fields)

    rouge_model, rouge_all = aggregate(rouge_scores)
    original_model, original_all = aggregate(scores)

    keys = [k for k in smatch_amr.keys()]

    smatch_dep_model, smatch_dep_all = aggregate(smatch_dep)
    smatch_amr_model, smatch_amr_all = aggregate(smatch_amr)
    sembleu_dep_model, sembleu_dep_all = aggregate(sembleu_dep)
    sembleu_amr_model, sembleu_amr_all = aggregate(sembleu_amr)

    # print
    print_banner('ROUGE system level')
    print_correlations(original_model, rouge_model)

    print_banner('Smatch dependency system level')
    print_correlations(original_model, smatch_dep_model)

    print_banner('Smatch AMR system level')
    print_correlations(original_model, smatch_amr_model)

    print_banner('Sembleu dependency system level')
    print_correlations(original_model, sembleu_dep_model)

    print_banner('Sembleu AMR system level')
    print_correlations(original_model, sembleu_amr_model)

    print_banner('ROUGE individual')
    print_correlations(original_all, rouge_all)

    print_banner('Smatch dependency individual')
    print_correlations(original_all, smatch_dep_all)

    print_banner('Smatch AMR individual')
    print_correlations(original_all, smatch_amr_all)

    print_banner('Sembleu dependency individual')
    print_correlations(original_all, sembleu_dep_all)

    print_banner('Sembleu AMR individual')
    print_correlations(original_all, sembleu_amr_all)

    print_banner('Correlations between ROUGE and AMR metrics')
    print_correlations(smatch_dep_all, rouge_all)
    print_correlations(sembleu_dep_all, rouge_all)
    print_correlations(smatch_amr_all, rouge_all)
    print_correlations(sembleu_amr_all, rouge_all)

    print_banner('Correlations between AMR metrics')
    print_correlations(smatch_dep_all, sembleu_dep_all)
    print_correlations(smatch_amr_all, sembleu_dep_all)
    print_correlations(smatch_dep_all, sembleu_amr_all)
    print_correlations(smatch_amr_all, sembleu_amr_all)
    print_correlations(smatch_dep_all, smatch_amr_all)
    print_correlations(sembleu_dep_all, sembleu_amr_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='Base working directory to store files in.')
    parser.add_argument('-d', '--dataset', type=str, default='summeval',
                        help='Dataset to evaluate metrics on.')
    parser.add_argument('--aggregate', action='store_true',
                        help='Whether to aggregate across references (or discard all but the first).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to redo all computations and overwrite existing intermediate files.')
    
    args = parser.parse_args()

    score_all(args.dataset, args.output, args.aggregate, args.overwrite)
