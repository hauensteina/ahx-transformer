
from sagemaker.pytorch import PyTorch

# You need this under trusted entities for your Sagemaker role on AWS console:
'''
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
'''

# This should be called 'command_line_args_for_your_train_script' (train.py)
hyperparameters = {
    'infile': 'samples_da', #'samples_cp',
    'block_sz': 32, # max context length
    'embed_sz': 32,
    'batch_sz': 32,
    'num_layers': 1,
    'num_heads': 2,
    'dropout': 0.0,
    'learning_rate': 3e-3, # 3e-4,
    'eval_interval': 10, # estimate loss and save model every eval_interval epochs
    'num_epochs': 100,
    }

aws_role = 'arn:aws:iam::147785435127:role/service-role/SageMaker-ahx'

# AWS
pytorch_estimator = PyTorch(
    entry_point='train.py', # My train script 
    role=aws_role,
    instance_count=1,
    #instance_type='ml.m5.large', # CPU instance
    instance_type='ml.g4dn.xlarge', # GPU instance
    source_dir='scripts', 
    framework_version='2.0', # PyTorch version
    py_version='py310',
    hyperparameters=hyperparameters,
    output_path='s3://ahx-sagemaker/double_a_output', # The folders environ['SM_OUTPUT_DATA_DIR'] and environ['SM_MODEL_DIR'] will be copied here
)
# Fit the model
fit_parms = {
    # We keep our training and validation data in this folder on S3. 
    # Folder gets copied to os.environ['SM_CHANNEL_TRAIN'] on the instance when the job starts.
    'train': 's3://ahx-sagemaker/double_a_data', 
}
pytorch_estimator.fit( fit_parms)  

