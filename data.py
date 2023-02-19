import os
import json
import spacy
import numpy as np

from resources import webnlg_dir, summeval_dir, summeval_path

def get_handle(n, handles):
    prefix = n.text[0]
    if prefix not in handles:
        handles[prefix] = n
        return prefix, handles
    i = 1
    handle = f'{prefix}{i}'
    while handle in handles:
        i += 1
        handle = f'{prefix}{i}'
    handles[handle] = n
    return handle, handles

def linearize_single(n, handles):
    handle, handles = get_handle(n, handles)
    linearized = f'( {handle} / {n.text} '
    for c in n.children:
        child, handles = linearize_single(c, handles)
        linearized += f'{c.dep_}: {child} '
    linearized += ')'
    
    return linearized, handles

def linearize_dependency(sents, nlp=None, clip=True):
    if nlp == None:
        nlp = spacy.load('en_core_web_trf')

    linearized = []

    for sent in sents:
        doc = nlp(sent)
        if clip:
            s = [s for s in doc.sents][0]
            single, handles = linearize_single(s.root, {})
            linearized.append(single)
        else:
            linearized.append(' '.join([linearize_single(s.root, {}) for s in doc.sents]))

    return linearized

def read_webnlg():
    pass

def read_summeval(f=summeval_path):
    full = [json.loads(l) for l in open(f, 'r').readlines()]
    targets = []
    outputs = []
    scores = {}
    metrics = ['coherence', 'consistency', 'fluency', 'relevance']

    for m in metrics:
        scores[m] = []
    
    for d in full:
        outputs.append(d['decoded'])
        targets.append(d['references'])
        for m in metrics:
            scores[m].append([np.mean(a[m]) for a in d['expert_annotations']])
    return outputs, targets, scores

def read_data(dataset):
    if dataset == 'webnlg':
        return read_webnlg()
    elif dataset == 'summeval':
        return read_summeval()
    else:
        print(f'This script does not support dataset {dataset}. Try [webnlg, summeval].')
        exit(1)

def exists(output_dir, fieldnames):
    '''
    Helper function. Check if multiple files already exist.
    '''
    for f in fieldnames:
        if not os.path.isfile(os.path.join(output_dir, f + '.json')):
            return False
    return True

def save(fields, output_dir, fieldnames, verbose=True):
    '''
    Helper function. Saves data fields with given names to an output directory.
    '''
    for field, f in zip(fields, fieldnames):
        with open(os.path.join(output_dir, f + '.json'), 'w') as out:
            json.dump(field, out)
    if verbose:
        print(f'Saved fields {fieldnames} to base directory {output_dir}.')

def load(output_dir, fieldnames, verbose=True):
    '''
    Helper function. Loads data fields with given names from an output directory.
    '''
    data = [json.load(open(os.path.join(output_dir, f + '.json'), 'r')) for f in fieldnames]
    if verbose:
        print(f'Loaded fields {fieldnames} from base directory {output_dir}.')
    return data

def separate_lines(data, data_type):
    pass

if __name__ == "__main__":
    pass
