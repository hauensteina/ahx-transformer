import argparse
import os
import re
import torch
from transformer import TransformerModel
from tokenizer import Tokenizer
from helpers import save_model, load_model, read_lines

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = '../models'
if 'SM_MODEL_DIR' in os.environ:
    CHECKPOINT_DIR = os.environ['SM_MODEL_DIR']

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Train a transformer to rewrite character sequences

    Synopsis:
      {name} [--problem <str>] 
      [--block_sz <int>] [--embed_sz <int>] [--batch_sz <int>] [--num_layers <int>] [--num_heads <int>] [--dropout <float>] 
      [--learning_rate <float>] [--eval_interval <int>] [--num_epochs <int>]  [--min_loss <int>] 
      [--model_in <file_path>] [--model_out <file_path>] --infile <str>

    Description:
        Train a transformer to rewrite character sequences. 

        --problem specifies which problem we're looking at.  One of 'copy', 'double_a'.
        This is just used to check test cases. The model is not aware of the problem.

        Training can run locally, or on sagemaker (see sagemaker_train.py).
        The input file must contain one input output pair per line.  Lines can be commented with #.
        For example, the following is a valid input file:

        # Minimal training data to get off the ground
        AB,AAB
        ABCAB,AABCAAB
        AB,AAB

        The model will be loaded from the file specified by --model_in. The file can be local or on S3.
        If --model_in is not given, the latest checkpoint is loaded (if running locally).
        Checkpoints <infile>_0001.pt, <infile>_0002.pt will be written to os.environ['SM_MODEL_DIR'] or to the models subfolder.
        Checkpoints are written every --eval_interval epochs.
        Training ends after --num_epochs, or earlier if the validation loss is less than --min_loss.
        The final model will be written to --model_out.

        Training data are taken from ../data if running locally, or from environ['SM_CHANNEL_TRAIN'] if running on SageMaker.
        The data files must be named <infile>_train.txt and <infile>_val.txt.  

    Examples:
      python {name} --problem copy --block_sz 32 --embed_sz 32 --batch_sz 32 --num_layers 1 --num_heads 2 --num_epochs 10 --learning_rate 3e-3 --infile samples_cp_small
      python {name} --problem copy --block_sz 32 --embed_sz 32 --batch_sz 32 --num_layers 1 --num_heads 2 --num_epochs 10 --learning_rate 3e-3 --infile samples_cp_small --model_out s3://my-bucket/my-model.pt

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--problem', type=str, required=True, choices=['double_a', 'copy'])
    parser.add_argument('--block_sz', type=int, default=32)
    parser.add_argument('--embed_sz', type=int, default=32)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--min_loss', type=float, default=0.04)
    parser.add_argument('--model_in', type=str)
    parser.add_argument('--model_out', type=str)
    args = parser.parse_args()
    args = args.__dict__
    if 'SM_CHANNEL_TRAIN' in os.environ:
        args['infile'] = os.path.join( os.environ['SM_CHANNEL_TRAIN'], args['infile'])
    else:
        args['infile'] = os.path.join( '../data', args['infile'])
    model = run(**args)

def run(problem, block_sz, embed_sz, batch_sz, num_layers, num_heads, dropout,
        learning_rate, eval_interval, num_epochs, min_loss, model_in, model_out, infile):
    
    def fresh_model():
        print(f'>>>> Fresh model')
        tok = Tokenizer(train_data)
        model = TransformerModel( DEVICE, tok, embed_sz, num_layers,
                                num_heads, block_sz, dropout)
        m = model.to(DEVICE)
        m.add_optimizer(learning_rate)
        return m

    print(f'>>>> Using device {DEVICE}')
    checkpoint_base = os.path.join(CHECKPOINT_DIR, os.path.split(infile)[-1])
    # Read all data into memory    
    train_data, val_data = read_data(infile)

    # Build or load model
    if model_in:
        try:
            print(f'>>>> Loading model from {model_in}')
            m = load_model( TransformerModel, DEVICE, model_in)
        except:
            print(f'>>>> Failed to load model from {model_in}. Using a fresh model instead')
            m = fresh_model()
    elif newest_checkpoint(checkpoint_base): 
        checkpoint_file = newest_checkpoint(checkpoint_base)
        print(f'>>>> Loading model from {checkpoint_file}')
        m = load_model( TransformerModel, DEVICE, checkpoint_file)
    else:
        m= fresh_model()

    tok = m.tokenizer
    train_data = [ tok.encode(x) for x in train_data ]
    val_data = [ tok.encode(x) for x in val_data ]
    print('A tokenized training sample:')
    print(train_data[0])

    print('\n>>>> Sanity checks')
    xb, yb = get_batch(tok, train_data, batch_sz, block_sz)
    logits, loss = m(xb, yb)
    print(f'Output logits shape (batch_sz*block_sz, alphabet_sz): {logits.shape}')
    print(f'Initial loss: {loss}')
    print(f'Generate something: {generate(m, "{A,")}')

    train(m, problem, num_epochs, min_loss, eval_interval, train_data, val_data, batch_sz, block_sz,
            checkpoint_base, model_out)


def train(model, problem, num_epochs, min_loss, eval_interval, train_data, val_data, batch_sz, block_sz,
          checkpoint_base, model_out):
    print('\n>>>> Start training')
    batches_per_epoch = len(train_data) // batch_sz
    tok = model.tokenizer
    for epoch_num in range(num_epochs):
        if epoch_num % eval_interval == 0:
            # Log the loss and run test cases
            losses = estimate_loss(model, tok, train_data, val_data, batch_sz, block_sz)
            print( f"\nBefore epoch {epoch_num}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
            print('First x,y in batch:')
            xb, yb = get_batch(tok, train_data, batch_sz, block_sz)
            print(tok.decode(xb[0].tolist()))
            print(tok.decode(yb[0].tolist()))
            testcases(model, problem, val_data)

            # Save model checkpoint
            fname = next_checkpoint(checkpoint_base)
            print(f'Saving model checkpoint to {fname}')
            model.save(fname)

            if losses[1] < min_loss:
                print(f'>>>> Validation loss {losses[1]:.4f} is less than {min_loss}. Stopping.')
                break

        for batch_num in range(batches_per_epoch):
            xb, yb = get_batch(tok, train_data, batch_sz, block_sz)
            logits, loss = model(xb, yb)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()

    if model_out: save_model(model, model_out)

def read_data(fname):
    """
    Read training and validation data, from two different files.
    Enclose each line in curly braces.
    """
    trainfname = fname + '_train.txt'
    valfname = fname + '_val.txt'
    trainval = [[],[]]
    for idx,fname in enumerate([trainfname, valfname]):  
        lines = read_lines(fname)
        lines = [ '{' + l + '}' for l in lines ]
        print(f'Number of samples in {fname}: ', len(lines))
        trainval[idx] = lines
    return trainval

def get_batch(tok, samps, batch_sz, block_sz):
    """
    Get one batch of training data.
    The output is the input shifted by one character.
    Output chars to the left of comma are replaced with 0.
    The rightmost output token looks one step into the future.
    """
    def get_y(x):
        """ 
        Replace anything between { up to and including , with 0 
        {AKA,AAKAA} -> {0000AAKAA}
        Then drop first char
        {0000AAKAA} -> 0000AAKAA}
        """
        ystr0 = tok.decode(x)
        def makezero(m): return '{' + '0' * (len(m.group(0)) - 1)
        ystr1 = re.sub(r'{[^,]*', makezero, ystr0)
        ystr2 = ystr1.replace(',', '0')[1:]
        return tok.encode(ystr2)

    batchx = []  # A list of lists
    batchy = []
    while len(batchx) < batch_sz:
        batchelx = []
        while len(batchelx) < block_sz + 1:
            idx = torch.randint(0, len(samps), (1,))
            batchelx += (samps[idx])
        batchely = get_y(batchelx[:block_sz+1])
        batchelx = batchelx[:block_sz]
        batchx.append(batchelx)
        batchy.append(batchely)

    batchx = torch.tensor(batchx)
    batchy = torch.tensor(batchy)
    batchx,batchy = batchx.to(DEVICE), batchy.to(DEVICE)
    return batchx, batchy  # (B,T)

@torch.no_grad()
def estimate_loss(m, tok, train_data, val_data, batch_sz, block_sz):
    """ Runs a few batches through the model and returns the average train and val loss"""
    n_batches = 100
    tv_losses = [0.0, 0.0]
    m.eval()
    for split, samps in enumerate([train_data, val_data]):
        losses = torch.zeros(n_batches)
        for k in range(n_batches):
            x, y = get_batch(tok, samps, batch_sz, block_sz)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        tv_losses[split] = losses.mean()
    m.train()
    return tv_losses

def testcases(m, problem, val_data):
    if problem == 'copy':
        testcases_cp(m, val_data)
    elif problem == 'double_a':
        testcases_da(m, val_data)
    else:
        raise ValueError(f'Unknown problem {problem}')
        
def testcases_cp(m, val_data):
    tok = m.tokenizer
    print('>>>> Test cases:')
    sampsize = 10
    for s in val_data[:sampsize]:
        prompt = tok.decode(s).split(',')[0] + ','        
        res = m.generate(prompt, stoptoken=tok.encode('}'), max_new_tokens=20)
        parts = res.split(',')
        prefix = ''
        if parts[0][1:] != parts[1][:-1]:
            prefix = 'ERROR: '
        print(f'{prefix}Prompt -> Res: {prompt} -> {res}')

def testcases_da(m, val_data):
    tok = m.tokenizer
    print('>>>> Test cases:')
    sampsize = 10
    for s in val_data[:sampsize]:
        prompt = tok.decode(s).split(',')[0] + ','        
        res = m.generate(prompt, stoptoken=tok.encode('}'), max_new_tokens=20)
        parts = res.split(',')
        prefix = ''
        if parts[1][:-1] != re.sub('A', 'AA', parts[0][1:]):
            prefix = 'ERROR: '
        print(f'{prefix}Prompt -> Res: {prompt} -> {res}')

def generate(model, prompt):
    tok = model.tokenizer
    out = model.generate(prompt,
            stoptoken=tok.encode('}'),
            max_new_tokens=20)
    return out

def newest_checkpoint(checkpoint_base):
    """
    Find the most recent checkpoint file.
    """
    base = os.path.split(checkpoint_base)[-1]
    folder = os.path.split(checkpoint_base)[0]
    files = os.listdir(CHECKPOINT_DIR)
    files = [f for f in files if f.startswith(base + '_')]
    files = [f for f in files if f.endswith('.pt')]
    if not files: return []
    files.sort()
    return os.path.join(folder, files[-1])

def next_checkpoint(checkpoint_base):
    """
    Build filename for next checkpoint.
    """
    latest = newest_checkpoint(checkpoint_base)
    if not latest: return checkpoint_base + '_0000.pt'
    latest = os.path.splitext(latest)[0]
    num = int(latest.split('_')[-1]) + 1
    out = latest.split('_')[:-1] + [f'''{num:04d}''']
    out = '_'.join(out) + '.pt'
    return out

if __name__ == '__main__':
    main()
