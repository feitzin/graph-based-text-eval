import sys
import json

def check(f):
    x = json.load(open(f, 'r'))
    layer = x
    while type(layer) == dict:
        keys = [k for k in layer.keys()]
        print(keys)
        layer = layer[keys[0]]

    print(type(layer))
    if type(layer) == list:
        print(len(layer))

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print('--------------------------------')
        print(f'Filename {f}:')
        print('--------------------------------')
        check(f)
        print()
