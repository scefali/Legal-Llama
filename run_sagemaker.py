import logging

from run_localGPT import load_model
import torch


from constants import (
    MODEL_ID,
    MODEL_BASENAME,
)


# Define a function to load the model.
def model_fn(model_dir):
    """
    Load the model from the model directory.
    This is called once when the endpoint starts.
    """

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)

    # Use your load_model function here
    model_pipeline = load_model(
        device_type=device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging
    )

    # For SageMaker, we might want to return the actual model instead of the pipeline.
    # But if the pipeline is essential for your inference, you can return the pipeline.
    return model_pipeline.pipeline.model


# Define a function to preprocess request input.
def input_fn(serialized_input_data, content_type="text/plain"):
    """
    Deserialize and preprocess the input data.

    Args:
        serialized_input_data (object): The serialized input data (in this case, a byte stream).
        content_type (str): The content type of the input data. Defaults to 'text/plain'.

    Returns:
        str: The deserialized input data (text string).
    """
    logging.info("Deserializing the input data.")

    if content_type == "text/plain":
        return serialized_input_data.decode("utf-8")
    raise ValueError(f"Unsupported content type: {content_type}")


# Define a function to run prediction.
def predict_fn(input_data, model):
    """
    Make a prediction using the provided model.

    Args:
        input_data (object): The processed input data (from input_fn). In our case, it's the text string.
        model (object): The loaded model. This is the return value of the `model_fn`.

    Returns:
        object: The prediction result.
    """
    logging.info("Generating text based on the input data.")

    # Using the provided model's pipeline to generate text
    output = model.pipeline(prompt=input_data)

    return output


# Define a function to format the model's prediction as a response.
def output_fn(prediction, content_type):
    """
    Formats the prediction result into the specified content type for response.

    Args:
        prediction (object): The result of the prediction from `predict_fn`.
        content_type (str): The MIME type of the desired response format.

    Returns:
        bytearray: The formatted response.
    """
    logging.info("Formatting the prediction result.")

    # Convert prediction to JSON string if content_type is JSON
    if content_type == "application/json":
        return json.dumps(prediction)

    # Otherwise, convert prediction to plain text
    elif content_type == "text/plain":
        # Assuming prediction is a dictionary with 'generated_text' as a key.
        return prediction["generated_text"]

    # Raise an error if unsupported content type is requested
    else:
        raise ValueError(f"Unsupported content type requested: {content_type}")
