
import argparse
import os
import re
import torch

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Optimize hyperparameters 

    Synopsis:
      {name} [--config <json_file>] 
    Description:
        The json file contains a command to generate training and test data,
        plus a list of configurations to try.
        Results go to test_harness.out.
        Example json file:
        { 
            "data": "python gen_data.py --problem double_a --num_samples 100000 --max_len 100 samples_da.txt",
            "configs": [
            
        }
        
    Example:
      python {name} --config test_double_a.json

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
    else:
        raise Exception(f'Unknown problem: {problem}')
    with open(outfile, 'w') as f:
        for sample in samples:
            f.write(f'{sample}\n')

def gen_double_a_samples(num_samples, max_len):
    samples = []
    for i in range(num_samples):
        sample = gen_double_a_sample(max_len)
        samples.append(sample)
    return samples

def gen_double_a_sample(max_len):
    sample = ''
    samplen = torch.randint(1, max_len, (1,)).item()
    sample = torch.randint(0, len(LETTERS), (samplen,))
    sample = ''.join([LETTERS[i] for i in sample])
    sample_out = re.sub('A', 'AA', sample)
    out = f'{sample},{sample_out}'
    return out

main()
