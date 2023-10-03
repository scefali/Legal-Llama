
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List

import boto3
from dotenv import load_dotenv
import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Configure boto3 client with credentials from .env file
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)


def load_single_document(bucket_name: str, key: str) -> Document:
    logging.info(f"Loading document: {key}")
    local_file_path = f"SOURCE_DOCUMENTS/{bucket_name}/{key}"

    # Check if the file exists locally
    if not os.path.isfile(local_file_path):
        # Get the file object from the S3 bucket
        obj = s3.get_object(Bucket=bucket_name, Key=key)

        # Read the file content
        file_content = obj['Body'].read().decode('utf-8')

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Save the file content locally for future use
        with open(local_file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)
            file.close()
            logging.info(f"Saved file locally: {local_file_path}")

    # Loads a single document from the file content
    # Make sure to get the file_extension appropriately from the key or metadata
    loader_class = DOCUMENT_MAP.get('.txt')
    if loader_class:
        loader = loader_class(local_file_path)  # Adjust this as necessary to work with content instead of file path
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_documents(bucket_name: str) -> list[Document]:
    # Loads all documents from the specified S3 bucket, including nested folders

    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Create a list of file paths (keys) for all objects in the bucket
    paths = [obj.key for obj in bucket.objects.all() if not obj.key.endswith('/')]

    # ... (the rest of your existing code which works with 'paths' variable stays the same)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames (keys)
            filepaths = paths[i: (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, bucket_name, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def load_document_batch(bucket_name: str, filepaths: List[str]):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, bucket_name, name) for name in filepaths[0:1]]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def split_documents(documents: list[Document]) -> tuple[List[Document], List[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    documents = load_documents('legal-scraper')
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,

    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
