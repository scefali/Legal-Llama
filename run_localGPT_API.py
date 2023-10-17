import logging
import os
import shutil
import subprocess
from asyncio import run
from langchain.globals import set_debug

import threading
from queue import Empty, Queue
import time

import torch
from flask import Flask, jsonify, stream_with_context, request, Response

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from queue import Queue
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.embeddings import HuggingFaceEmbeddings
from run_localGPT import load_model

from langchain.vectorstores import Chroma
from callback_handlers import StreamingCallbackHandler


from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME


app = Flask(__name__)
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_SOURCES = False
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

queue = Queue()  # This queue will store user prompts

set_debug(True)


# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
if os.path.exists(PERSIST_DIRECTORY):
    try:
        shutil.rmtree(PERSIST_DIRECTORY)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")
else:
    print("The directory does not exist")

run_langest_commands = ["python", "ingest.py"]
if DEVICE_TYPE == "cpu":
    run_langest_commands.append("--device_type")
    run_langest_commands.append(DEVICE_TYPE)

result = subprocess.run(run_langest_commands, capture_output=True)
if result.returncode != 0:
    raise FileNotFoundError(
        "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
    )

# load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
print("got model", LLM)

QA = RetrievalQA.from_chain_type(
    llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
)

app = Flask(__name__)


@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python", "ingest.py"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        # load the vectorstore
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()

        QA = RetrievalQA.from_chain_type(
            llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/prompt", methods=["GET", "POST"])
def prompt_route():
    global QA
    user_prompt = request.args.get("user_prompt")
    print("user prompt", user_prompt)
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        res = QA(user_prompt)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400
    


@app.route("/api/test", methods=["GET"])
def def_test_routes():
    return "Test successful"


@app.route("/stream/prompt")
def streamed_response():
    user_prompt = request.args.get("user_prompt")
    print("got user prompt", user_prompt)
    if not user_prompt:
        return "No user prompt received", 400

    token_queue = Queue()

    def generate():
        def on_token(token):
            print("on token", token)
            token_queue.put(token)

        def stream_function():
            return QA(user_prompt, callbacks=[StreamingCallbackHandler(on_token)])

        # Start the streaming in a separate thread
        stream_thread = threading.Thread(target=stream_function)
        stream_thread.start()

        # In the main thread, yield tokens from the queue:
        while stream_thread.is_alive():  # While the stream_thread is running
            try:
                token = token_queue.get(timeout=10)  # Wait for a token for up to 5 seconds
                yield token
            except Empty:
                # No token received within the timeout, but the thread is still running.
                continue

        # After the streaming thread has finished, there might be tokens left in the queue. Handle them:
        while not token_queue.empty():
            yield token_queue.get()

        stream_thread.join()

    return stream_with_context(generate())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, host='0.0.0.0', port=5110)
