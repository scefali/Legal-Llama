# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.0-base-ubuntu20.04 

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get install -y g++-11 make python3 python-is-python3 pip

# Install necessary dependencies and Python 3.10
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-distutils python3.10-venv python3.10-dev python3-pip

# Optionally set Python 3.10 as the default for "python3"
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy just the requirements and install them first for caching
COPY ./requirements.txt .
RUN --mount=type=cache,target=/root/.cache CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --timeout 100 -r requirements.txt

# Copy the rest of the files
COPY . .

RUN ls -al

ENV PATH="/opt/program:${PATH}"



# Make sure train and serve scripts are in PATH and executable
COPY train.sh /opt/program/train
COPY serve.sh /opt/program/serve

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve



ENV device_type=cuda

# The ENTRYPOINT is not specified since SageMaker will run the container with either "train" or "serve" as the command
