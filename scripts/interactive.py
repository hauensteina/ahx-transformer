from pdb import set_trace as BP
import argparse
import os
import re
import torch
from torch import tensor
from transformer import TransformerModel
from helpers import load_model, read_lines

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Interactively type a prompt and see the model's response.

    Synopsis:
      {name} --model_file <model.pt> 

    Example:
      python {name} --model_file da_small.pt
      python {name} --model_file s3://my-bucket/my_model.pt

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('--model_file', type=str, required=True)
    args = parser.parse_args()
    args = args.__dict__
    model = run(**args)

def run(model_file):
    print('Loading model...')
    model = load_model(TransformerModel, DEVICE, model_file)
    chars = set(sorted(model.tokenizer.chars)) - set('{},0')
    print(f'Valid input characters are: {chars}')
    while True:
        prompt = input('Enter a prompt: ')
        prompt = '{' + prompt + ','
        stoptoken = model.tokenizer.encode('}')
        try:
            out = model.generate(prompt, stoptoken)
        except Exception as e:
            print(e.__repr__())
            continue
        print(out)

if __name__ == '__main__':
    main()
