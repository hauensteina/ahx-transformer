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
      {name}: Evaluate a transformer model on a given dataset.

    Synopsis:
      {name} --model_file <model.pt> --data_file <dataset.txt>

    Description:
        Runs the model on the dataset and prints loss and accuracy.

    Example:
      python {name} --model_file da_small.pt --data_file ../data/samples_da_small_val.txt
      python {name} --model_file s3://my-bucket/my_model.pt --data_file s3://my-bucket/samples_da_val.txt

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    args = parser.parse_args()
    args = args.__dict__
    model = run(**args)

def run(model_file, data_file):
    model = load_model(TransformerModel, DEVICE, model_file)
    lines = read_lines(data_file)
    lines = [ '{' + l + '}' for l in lines ]
    stoptoken = model.tokenizer.encode('}')
    n_errors = 0
    model.eval() # Switch to eval mode
    errlist = []
    maxerrs = 10
    with torch.no_grad():
        for idx,line in enumerate(lines):
            if idx % 100 == 0:
                print(f'>>>> Processing line {idx}/{len(lines)}')
            prompt = line.split(',')[0] + ','
            out = model.generate(prompt, stoptoken)
            if out != line:
                n_errors += 1
                if len(errlist) < maxerrs:
                    errlist.append( (prompt, line, out) )

    if len(errlist) > 0:
        print(f'\n>>>> First {maxerrs} Errors:')
        for prompt,expected,out in errlist:
            print(f'>>>>   prompt:   {prompt}')
            print(f'>>>>   expected: {expected}')
            print(f'>>>>   got:      {out}')
            print(f' ')

    print( (f'model:{model_file}, data:{data_file}, samples:{len(lines)}, ' + 
            f'errors:{n_errors}, error_rate:{100*n_errors/len(lines):.2f}' ))

if __name__ == '__main__':
    main()
