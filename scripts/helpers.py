
import os
import uuid
import boto3
from tokenizer import Tokenizer

def save_model(model, fname):
    """ Save model to local file or S3 """
    if fname.startswith('s3://'):
        print(f'>>>> Saving model on S3: {fname}')
        local_fname = str(uuid.uuid4()) + '.pt'
        model.save(local_fname)
        path_parts = fname.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        s3 = boto3.client('s3')
        s3.upload_file(local_fname, bucket, key)
        os.remove(local_fname) 
    else: # local file
        print(f'>>>> Saving model locally to  {fname}')
        model.save(fname)

def load_model( model_cls, device, fname):
    """ Load model from local file or S3 """
    tok = Tokenizer([])
    if fname.startswith('s3://'):
        print(f'>>>> Loading model from S3: {fname}')
        path_parts = fname.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        s3 = boto3.client('s3')
        local_fname = str(uuid.uuid4()) + '.pt'
        s3.download_file(bucket, key, local_fname)
        m = model_cls.load( device, tok, local_fname)
        os.remove(local_fname) 
    else: # local file
        print(f'>>>> Loading model from local file {fname}')
        m = model_cls.load( device, tok, fname)
    return m
