# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get install -y g++-11 make python3 python-is-python3 pip

# Copy just the requirements and install them first for caching
COPY ./requirements.txt .
RUN --mount=type=cache,target=/root/.cache CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --timeout 100 -r requirements.txt

# Copy the rest of the files
COPY . .

RUN ls -al


# Make sure train and serve scripts are in PATH and executable
COPY train.sh /opt/program/train
COPY serve.sh /opt/program/serve

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve


ENV device_type=cuda

# The ENTRYPOINT is not specified since SageMaker will run the container with either "train" or "serve" as the command
