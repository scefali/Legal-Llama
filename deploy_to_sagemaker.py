import torch
import logging
import boto3
import os

from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker.local import LocalSession


from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    DEPLOYED_MODEL_ID,
    DEPLOYED_MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
)
from run_localGPT import load_model
from load_models import load_quantized_model_qptq


REPO_NAME = "legal-lama-model"
REGION = "us-west-2"
ACCOUNT_ID = 610179610581

# Configuration
image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}:latest"
role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMaker"


def deploy_model():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda"

    # 1. Download the model
    # model = load_model(device_type, model_id=DEPLOYED_MODEL_ID, model_basename=DEPLOYED_MODEL_BASENAME)
    model, tokenizer = load_quantized_model_qptq(DEPLOYED_MODEL_ID, DEPLOYED_MODEL_BASENAME, device_type, logging)

    # 2. Save the model locally
    model_dir_local = "deployed_models"

    # Ensure local directory exists
    if not os.path.exists(model_dir_local):
        os.makedirs(model_dir_local)

    model.save_pretrained(model_dir_local)
    tokenizer.save_pretrained(model_dir_local)

    model_file = f"{model_dir_local}/model.tar.gz/model.bin"
    tokenizer_file = f"{model_dir_local}/tokenizer.tar.gz/tokenizer.model"

    # 3. Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(model_file, "legal-llama-model", os.path.basename(model_file))
    s3_client.upload_file(tokenizer_file, "legal-llama-model", os.path.basename(tokenizer_file))

    # 4. Create the session
    sagemaker_session = Session()
    # sagemaker_session = LocalSession()

    model_path_in_s3 = "s3://legal-llama-model/model.bin"

    model = Model(
        image_uri=image_uri,
        role=role_arn,
        sagemaker_session=sagemaker_session,
        model_data=model_path_in_s3,
    )

    # Deploy the model. This will launch an endpoint.
    predictor = model.deploy(instance_type="ml.p3.2xlarge", initial_instance_count=1)

    return predictor


def test_endpoint(predictor):
    sample_input = "Your sample input data here"
    response = predictor.predict(sample_input)
    print(response)
    return response


def cleanup(predictor):
    predictor.delete_endpoint()


if __name__ == "__main__":
    # Deploy the model
    deployed_predictor = deploy_model()

    # Test the deployed model
    test_endpoint(deployed_predictor)

    # Optionally: Clean up the endpoint after testing
    # cleanup(deployed_predictor)
