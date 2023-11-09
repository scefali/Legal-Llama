import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from tqdm import tqdm


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

BATCH_SIZE = 10  # Choose a batch size that suits your machine's memory constraints

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Configure boto3 client with credentials from .env file
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_single_document(bucket_name: str, key: str) -> Document:
    logging.info(f"Loading document: {key}")
    local_file_path = f"SOURCE_DOCUMENTS/{bucket_name}/{key}"

    # Check if the file exists locally
    if not os.path.isfile(local_file_path):
        # Get the file object from the S3 bucket
        obj = s3.get_object(Bucket=bucket_name, Key=key)

        # Read the file content
        file_content = obj["Body"].read().decode("utf-8")

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Save the file content locally for future use
        with open(local_file_path, "w", encoding="utf-8") as file:
            file.write(file_content)
            file.close()
            logging.info(f"Saved file locally: {local_file_path}")

    # Loads a single document from the file content
    # Make sure to get the file_extension appropriately from the key or metadata
    loader_class = DOCUMENT_MAP.get(".txt")
    if loader_class:
        loader = loader_class(local_file_path)  # Adjust this as necessary to work with content instead of file path
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_documents(bucket_name: str) -> list[Document]:
    # Loads all documents from the specified S3 bucket, including nested folders

    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Create a list of file paths (keys) for all objects in the bucket
    paths = [obj.key for obj in bucket.objects.all() if not obj.key.endswith("/")]

    # Determine the size of each chunk based on the total paths and INGEST_THREADS
    chunk_size = len(paths) // INGEST_THREADS

    docs = []

    with ProcessPoolExecutor(INGEST_THREADS) as executor:
        for chunk in chunkify(paths, chunk_size):
            results = executor.map(load_single_document, [bucket_name] * len(chunk), chunk)
            docs.extend(results)

    return docs


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


def process_batch(batch_texts, embeddings):
    logging.info(f"Processing batch of {len(batch_texts)} texts")
    try:
        # This function stores embeddings for a batch of texts using Chroma.from_documents
        Chroma.from_documents(
            batch_texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
    except Exception as e:
        logging.error(f"An error occurred while processing batch: {e}")
        raise e



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
    documents = load_documents("legal-scraper")
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
    logging.info(f"Created embeddings for {EMBEDDING_MODEL_NAME}")

    # Define batch size for processing embeddings

    # Create batches of texts
    text_batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    logging.info(f"Handling {len(text_batches)} batches")


    # Use ProcessPoolExecutor to parallelize the embedding creation and storage process
    with ProcessPoolExecutor(max_workers=INGEST_THREADS) as executor:
        # Schedule the tasks and collect the futures
        futures = {executor.submit(process_batch, batch, embeddings): batch for batch in text_batches}

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(futures), total=len(text_batches), desc="Processing batches"):
            try:
                # This will raise an exception if one occurred within a thread
                future.result()
            except Exception as e:
                # Handle exceptions here if you need to do something specific on failure
                logging.error(f'Batch processing generated an exception: {e}')
                raise e

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
