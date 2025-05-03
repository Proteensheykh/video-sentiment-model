from sagemaker.pytorch import PyTorchModel
from dotenv import load_dotenv

import os
import sagemaker

load_dotenv()

SAGEMAKER_ENDPOINT_ROLE_ARN = os.getenv('SAGEMAKER_ENDPOINT_ROLE_ARN')
AWS_S3_URI = os.getenv('AWS_S3_URI')

def deploy_endpoint():
    sagemaker.Session()
    role = SAGEMAKER_ENDPOINT_ROLE_ARN
    model_uri = os.path.join(AWS_S3_URI, "model.tar.gz")

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )

if __name__ == "__main__":
    deploy_endpoint()