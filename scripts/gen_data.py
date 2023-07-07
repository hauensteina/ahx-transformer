
import argparse
import os
import re
import torch

#LETTERS = 'ABCDEFGHIJ'
LETTERS = 'AB'
VAL_SPLIT = 0.1

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Generate training data for a transformer to rewrite character sequences

    Synopsis:
      {name} [--problem copy,double_a] [--num_samples <int>] [--max_len <int>] outfile
    Description:
        Generate training and validation data for a transformer to rewrite character sequences.

        --problem: Which problem to generate data for
        --num_samples: How many samples to generate
        --max_len: Maximum length of a sample

        Output goes to <outfile>_train.txt and <outfile>_val.txt 
        Ten percent of the samples will be used for validation.

    Example:
      python {name} --problem copy --num_samples 100000 --max_len 100 samples_da

    '''
    msg += '\n '
    return msg

# -------------

def main():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('outfile', type=str)
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    args = parser.parse_args()
    args = args.__dict__
    run(**args)

def run(outfile, problem, num_samples, max_len):
    if problem == 'double_a':
        samples = gen_double_a_samples(num_samples, max_len)
    elif problem == 'copy':
        samples = gen_copy_samples(num_samples, max_len)
    else:
        raise Exception(f'Unknown problem: {problem}')
    valfile = outfile + '_val.txt'
    trainfile = outfile + '_train.txt'
    splitidx = int(len(samples) * VAL_SPLIT)
    with open(valfile, 'w') as f:
        for sample in samples[:splitidx]:
            f.write(f'{sample}\n')
    with open(trainfile, 'w') as f:
        for sample in samples[splitidx:]:
            f.write(f'{sample}\n')

def gen_double_a_samples(num_samples, max_len):
    samples = []
    for i in range(num_samples):
        samplen = torch.randint(1, max_len, (1,)).item()
        sample = torch.randint(0, len(LETTERS), (samplen,))
        sample = ''.join([LETTERS[i] for i in sample])
        sample_out = re.sub('A', 'AA', sample)
        out = f'{sample},{sample_out}'
        samples.append(out)
    return samples

def gen_copy_samples(num_samples, max_len):
    samples = []
    for i in range(num_samples):
        samplen = torch.randint(1, max_len, (1,)).item()
        sample = torch.randint(0, len(LETTERS), (samplen,))
        sample = ''.join([LETTERS[i] for i in sample])
        out = f'{sample},{sample}'
        samples.append(out)
    return samples

main()
