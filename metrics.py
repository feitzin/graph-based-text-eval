import os
import json
import subprocess
import argparse

import spacy
import smatch
from torchmetrics.text.rouge import ROUGEScore

from data import read_data, linearize_dependency, exists, save, load
from resources import temp_output_f, temp_target_f

def score_sembleu(candidates, targets):
    scores = []
    for c, t in zip(candidates, targets):
        with open(temp_output_f, 'w') as outf, open(temp_target_f, 'w') as targf:
            outf.write(c)
            targf.write(t)
        output = subprocess.run(['smatch.py', '--ms', '-f', temp_output_f, temp_target_f], capture_output=True).stdout
        scores.append(float(output.split('\n')[-2]))
    return scores

def score_smatch(candidates, targets):
    with open(temp_output_f, 'w') as outf, open(temp_target_f, 'w') as targf:
        for c, t in zip(candidates, targets):
            outf.write(c + '\n')
            targf.write(t + '\n')
    args = ['smatch.py', '--ms', '-f', temp_output_f, temp_target_f]
    output = subprocess.run(args, text=True, capture_output=True)
    scores = [x.strip() for x in output.stdout.split('\n')]
    scores = [float(x.split('F-score: ')[-1]) for x in scores]
    return scores

def score(outputs, targets):
    print('Scoring AMR metrics...')
    smatch = score_smatch(outputs, targets)
    sembleu = score_sembleu(outputs, targets)

    return smatch, sembleu

def score_rouge(outputs, targets):
    '''
    Collects only ROUGE scores.
    '''
    print('Scoring on ROUGE...')
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

def print_correlations(first, second):
    '''
    Compute and print correlations between two dictionaries of scores.
    '''
    for metric in first:
        for second_metric in second:
            corr = np.corrcoef(first[metric], second[second_metric])
            print(f'Correlation between {metric} and {second_metric}: {corr}')
        print()

def score_all(dataset, output_base, multi_references=False, overwrite=False):
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
            targets = [t[0] for t in targets]
        save([outputs, targets, scores], output_dir, fields)
    else:
        outputs, targets, scores = load(output_dir, fields)

    # get dependency linearizations
    fields = ['outputs_dep', 'targets_dep']
    if overwrite or not exists(output_dir, fields):
        print('Linearizing dependencies...')
        nlp = spacy.load('en_core_web_trf')
        outputs_dep = [linearize_dependency(s, nlp) for s in outputs]
        targets_dep = [linearize_dependency(s, nlp) for s in targets]
        save([outputs_dep, targets_dep], output_dir, fields)
    else:
        outputs_dep, targets_dep = load(output_dir, fields)

    # calculate rouge scores
    fields = ['rouge_scores']
    if overwrite or not exists(output_dir, fields):
        rouge_scores = score_rouge(outputs, targets)
        save([rouge_scores], output_dir, fields)
    else:
        rouge_scores = load(output_dir, fields)

    # calculate amr scores
    fields = ['smatch_scores', 'sembleu_scores']
    if overwrite or not exists(output_dir, fields):
        smatch_scores, sembleu_scores = score(outputs_dep, targets_dep)
        save([smatch_scores, sembleu_scores], output_dir, fields)
    else:
        smatch_scores, sembleu_scores = load(output_dir, fields)

    # print
    print_banner('ROUGE correlations')
    print_correlations(scores, rouge_scores)

    print_banner('Smatch correlations')
    print_correlations(scores, smatch_scores)

    print_banner('Sembleu correlations')
    print_correlations(scores, sembleu_scores)

    print_banner('Correlations between ROUGE and AMR metrics')
    print_correlations(rouge_scores, smatch_scores)
    print_correlations(rouge_scores, sembleu_scores)

    print_banner('Correlations between AMR metrics')
    print_correlations(smatch_scores, sembleu_scores)

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
