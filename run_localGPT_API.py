import logging
import os
import shutil
from langchain.globals import set_debug

import threading
from queue import Empty, Queue

import torch
from flask import Flask, jsonify, stream_with_context, request

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate

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

prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Just answer the question directly.

{context}
.\n\nQuestion: {question}\n

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
if os.path.exists(PERSIST_DIRECTORY):
    try:
        shutil.rmtree(PERSIST_DIRECTORY)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")
else:
    print("The directory does not exist")

# load the vectorstore
print("loading vectorstore")
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)
print("loaded vectorstore")

RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
print("got model", LLM)

QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=SHOW_SOURCES,
)

app = Flask(__name__)


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
    app.run(debug=False, host="0.0.0.0", port=5110)
