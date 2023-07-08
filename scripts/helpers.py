
import os
import uuid
import boto3
from tokenizer import Tokenizer

def s3parts(s3fname):
    """ Split S3 filename into bucket and key """
    path_parts = s3fname.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key

def download_s3(s3fname):
    """ Copy S3 file to local file with uuid filename """
    ext = os.path.splitext(s3fname)[1]
    bucket,key = s3parts(s3fname)
    s3 = boto3.client('s3')
    local_fname = str(uuid.uuid4()) + ext
    s3.download_file(bucket, key, local_fname)
    return local_fname

def upload_s3(localfname, s3fname):
    """ Copy local file to S3 """
    bucket,key = s3parts(s3fname)
    s3 = boto3.client('s3')
    s3.upload_file(localfname, bucket, key)

def save_model(model, fname):
    """ Save model to local file or S3 """
    if fname.startswith('s3://'):
        print(f'>>>> Saving model: {fname}')
        local_fname = str(uuid.uuid4()) + '.pt'
        model.save(local_fname)
        upload_s3(local_fname, fname)
        os.remove(local_fname) 
    else: # local file
        print(f'>>>> Saving model locally to  {fname}')
        model.save(fname)

def load_model(model_cls, device, fname):
    """ Load model from local file or S3 """
    tok = Tokenizer([])
    if fname.startswith('s3://'):
        print(f'>>>> Loading model: {fname}')
        local_fname = download_s3(fname)
        m = model_cls.load( device, tok, local_fname)
        os.remove(local_fname) 
    else: # local file
        print(f'>>>> Loading model from local file {fname}')
        m = model_cls.load( device, tok, fname)
    return m

def read_lines(fname):
    """ 
    Read all lines from a file (local or S3) into a list of trimmed strings.
    Ignore empty lines.
    Ignore lines starting with #.
    """
    def _readlines(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [ l.strip() for l in lines ]
        lines = [ l for l in lines if len(l) > 0 and not l.startswith('#') ]
        return lines
    
    if fname.startswith('s3://'):
        local_fname = download_s3(fname)
        lines = _readlines(local_fname) 
        os.remove(local_fname)
        return lines 
    else: # local file
        return _readlines(fname)
