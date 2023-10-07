# Variables
REPO_NAME = legal-lama-model
REGION = us-west-2
ACCOUNT_ID = 610179610581

# Get AWS login password and authenticate Docker
login:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com

# Build Docker image
build:
	docker build -t $(REPO_NAME) .

# Tag Docker image for Amazon ECR
tag:
	docker tag $(REPO_NAME):latest $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPO_NAME):latest

# Push Docker image to Amazon ECR
push:
	docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPO_NAME):latest

deploy:
	python deploy_to_sagemaker.py

# All-in-one command
all: login build tag push deploy

.PHONY: login build tag push all
