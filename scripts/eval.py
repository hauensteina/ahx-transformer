
import argparse
import os
import re
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from tokenizer import Tokenizer
from helpers import save_model, load_model, read_lines

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Evaluate a transformer model on a given dataset.

    Synopsis:
      {name} --model <model.pt> --dataset <dataset.txt>

    Description:
        Runs the model on the dataset and prints loss and accuracy.

    Example:
      python {name} --model cp_0010.pt --dataset samples_da_val.txt
      python {name} --model s3://my-bucket/my_model.pt --dataset s3://my-bucket/samples_da_val.txt

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    args = args.__dict__
    model = run(**args)

def run(model_file, data_file):
    model = load_model(model_file)
    lines = read_lines(data_file)
    lines = [ '{' + l + '}' for l in lines ]
    stoptoken = model.tokenizer.encode('}')
    n_errors = 0
    model.eval() # Switch to eval mode
    with torch.no_grad():
        for line in lines:
            prompt = line.split(',')[0]
            out = model.generate(prompt, stoptoken)
            if out != line.split(',')[1][:-1]:
                n_errors += 1

    print( f'model:{model_file}, data:{data_file}, samples:{len(lines)}, errors:{n_errors}, error_rate:{100*n_errors/len(lines):.2f}' )

if __name__ == '__main__':
    main()
