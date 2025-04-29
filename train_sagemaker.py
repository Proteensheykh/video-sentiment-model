from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from dotenv import load_dotenv

import os

load_dotenv()

AWS_S3_URI = os.getenv('AWS_S3_URI')
SAGEMAKER_EXECUTION_ROLE_ARN = os.getenv('SAGEMAKER_EXECUTION_ROLE_ARN')

def start_training():
    print("starting training...")

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=os.path.join(AWS_S3_URI, 'tensorboard'),
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=SAGEMAKER_EXECUTION_ROLE_ARN,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fits({
        "training":os.path.join(AWS_S3_URI, "dataset", "train"),
        "validation":os.path.join(AWS_S3_URI, "dataset", "dev"),
        "test":os.path.join(AWS_S3_URI, "dataset", "test"),
    })

if __name__ == "__main__":
    start_training()
