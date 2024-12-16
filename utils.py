import subprocess
import time
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, BitsAndBytesConfig

import config


def start_qdrant_server():
    """
    Start the Qdrant server using Docker in a new terminal window.
    """
    docker_command = (
        'docker run -p 6333:6333 -p 6335:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant'
    )
    try:
        subprocess.Popen(['start', 'cmd', '/c', docker_command], shell=True)
        print("Qdrant server started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while starting the Qdrant server: {e}")


def load_model_with_retry(model_name, cache_dir, retries=3, wait_time=10):
    """
    Attempt to load the model with retries in case of timeouts or other issues.

    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): The directory to cache and load models from.
        retries (int): Number of times to retry loading the model.
        wait_time (int): Time to wait between retries in seconds.

    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    for attempt in range(retries):
        try:
            if model_name in [config.CODEBERT_MODEL_NAME, config.GRAPH_CODEBERT_MODEL_NAME]:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            elif model_name == config.CODET5_MODEL_NAME:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = T5EncoderModel.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            
            print("model and tokenzier created")
            return model, tokenizer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise e
            

def load_huggingface_model(model_name, cache_dir=None):
    """
    Load a Hugging Face model and its tokenizer by model name.

    Args:
        model_name (str): The name of the model to load from the Hugging Face model hub.
        cache_dir (str, optional): Directory to cache and load models from. Defaults to None.

    Returns:
        model, tokenizer: Loaded Hugging Face model and tokenizer.
    """
    try:

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name == "microsoft/codebert-base":
            # Load the model
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name in  ["Salesforce/codet5p-2b", "Salesforce/codet5-base"]:
            # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )

        if model_name == "microsoft/graphcodebert-base":
            model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name == "Qwen/Qwen2.5-Coder-0.5B":
            # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"  # Automatically maps layers to available devices
            )

        print(f"Successfully loaded model and tokenizer: {model_name}")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")


def get_embedding_and_token_size(model, tokenizer):
    """
    Determine the embedding size and maximum token input size of the Hugging Face model.

    Args:
        model: The Hugging Face model.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        tuple: The embedding size and the maximum token input size.
    """
    # Create a dummy input
    dummy_input = tokenizer("dummy text", return_tensors="pt")
    
    # Check if the model requires both input_ids and decoder_input_ids
    if model.config.is_encoder_decoder:
        dummy_input["decoder_input_ids"] = dummy_input["input_ids"]
    
    # Pass the dummy input through the model
    outputs = model(**dummy_input)

    # Handle different output types
    if hasattr(outputs, "last_hidden_state"):  # Standard models
        embedding_size = outputs.last_hidden_state.size(-1)
    elif hasattr(outputs, "logits"):  # For masked LM models
        embedding_size = outputs.logits.size(-1)
    elif hasattr(outputs, "hidden_states"):  # Models with hidden states
        embedding_size = outputs.hidden_states[-1].size(-1)
    else:
        raise AttributeError("The model's output does not contain a known embedding attribute.")
    
    # Retrieve the maximum token input size
    max_token_input_size = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else None

    # Validate the embedding size
    if embedding_size > config.QDRANT_MAX_EMBEDDING_SIZE:
        print(
            f"Warning: Model's embedding size ({embedding_size}) exceeds the maximum allowed size ({config.QDRANT_MAX_EMBEDDING_SIZE}). "
            f"Using the maximum allowed size instead."
        )
        embedding_size = config.QDRANT_MAX_EMBEDDING_SIZE
        # embedding_size = config.DESIRED_EMBEDDING_SIZE
        config.REDUCE_EMBEDDING_SIZE = True

    return embedding_size, max_token_input_size






# def generate_code_embedding(code, model, tokenizer):
#     """
#     Create an embedding for the given code using the specified model.

#     Args:
#         code (str): The code snippet to create an embedding for.
#         model_name (str): The name of the model to use for embedding creation.
#         cache_dir (str): The directory to cache and load models from.

#     Returns:
#         numpy.ndarray: The embedding of the code snippet.
#     """
#     inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
#     return embedding




def reduce_embedding_size(embedding, target_size = 768):
    """
    Truncate the embedding to the target size if it exceeds the limit.

    Args:
        embedding (numpy.ndarray): Original high-dimensional embedding.

    Returns:
        numpy.ndarray: Truncated embedding.
    """
    return embedding[:target_size] if embedding.shape[0] > target_size else embedding





# def generate_code_embedding_generic_old(code, model, tokenizer):
#     """
#     Create an embedding for the given code using the specified model with mixed precision.

#     Args:
#         code (str): The code snippet to create an embedding for.
#         model: The Hugging Face model.
#         tokenizer: The tokenizer for the model.

#     Returns:
#         numpy.ndarray: The embedding of the code snippet.
#     """
#     # # Define the device
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # # Only move the model to the device if it is not a quantized model
#     # if not hasattr(model, "quantization_config"):
#     #     model.to(device)  # Move the model to the appropriate device

#     # # Tokenize the input code and move inputs to the device
#     # inputs = tokenizer(
#     #     code, 
#     #     return_tensors="pt", 
#     #     truncation=True, 
#     #     padding=True, 
#     #     max_length=config.MODEL_MAX_INPUT_TOKENS
#     # ).to(device)

#         # Tokenize the input code and move inputs to the device
#     inputs = tokenizer(
#         code, 
#         return_tensors="pt", 
#         truncation=True, 
#         padding=True, 
#         max_length=config.MODEL_MAX_INPUT_TOKENS
#     )

#     # Handle encoder-decoder models (e.g., T5 or Bart)
#     if model.config.is_encoder_decoder:
#         inputs["decoder_input_ids"] = inputs["input_ids"]

#     # # Enable mixed precision (handle fallback if `device_type` is unsupported)
#     # try:
#     #     with autocast(device_type="cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
#     #         outputs = model(**inputs)
#     # except TypeError as e:
#     #     print(f"Falling back: autocast failed with error: {e}")
#     #     with autocast(dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
#     #         outputs = model(**inputs)

#     outputs = model(**inputs)

#     # Extract embeddings based on model's output attributes
#     if hasattr(outputs, "last_hidden_state"):  # Standard models (e.g., CodeBERT)
#         embeddings = outputs.last_hidden_state
#     elif hasattr(outputs, "logits"):  # Models outputting logits
#         embeddings = outputs.logits
#     elif hasattr(outputs, "hidden_states"):  # Models with intermediate hidden states
#         embeddings = outputs.hidden_states[-1]  # Use the last layer
#     else:
#         raise AttributeError("Model output does not contain embeddings in a known attribute.")

#     # Compute the mean embedding for the sequence
#     embedding = embeddings.mean(dim=1).squeeze().detach().cpu().numpy()

#     # Optional: Reduce embedding size for efficiency
#     if hasattr(config, "REDUCE_EMBEDDING_SIZE") and config.REDUCE_EMBEDDING_SIZE:
#         embedding = reduce_embedding_size(embedding, 768)

#     return embedding


def generate_code_embedding_generic(code, model, tokenizer):
    """
    Create an embedding for the given code using the specified model with mixed precision.

    Args:
        code (str): The code snippet to create an embedding for.
        model: The Hugging Face model.
        tokenizer: The tokenizer for the model.

    Returns:
        numpy.ndarray: The embedding of the code snippet.
    """

    # max_length = model.config.max_position_embeddings

    # Tokenize the input code and move inputs to the device
    # inputs = tokenizer(
    #     code[:max_length],  # Truncate input if necessary 
    #     return_tensors="pt", 
    #     truncation=True, 
    #     padding=True, 
    #     max_length=config.MODEL_MAX_INPUT_TOKENS
    # )

    # Define the semantic prompt
    PROMPT = "You are a programming expert. Analyze the following code snippet semantically, focusing on its functionality and behavior. Understand its purpose, logic, and how it operates, while ignoring irrelevant details such as formatting or variable names:\nCode:\n"
    
    # Combine the prompt with the code
    full_input = PROMPT + code

    inputs = tokenizer(
        full_input,
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        max_length=config.MAX_TOKEN_INPUT_SIZE
    )

    # inputs = tokenizer(
    #     code,
    #     return_tensors="pt", 
    #     truncation=True, 
    #     padding=True
    # )

    

    # Handle encoder-decoder models (e.g., T5 or Bart)
    if model.config.is_encoder_decoder:
        inputs["decoder_input_ids"] = inputs["input_ids"]

    outputs = model(**inputs)

    # Extract embeddings based on model's output attributes
    if hasattr(outputs, "last_hidden_state"):  # Standard models (e.g., CodeBERT)
        embeddings = outputs.last_hidden_state
    elif hasattr(outputs, "logits"):  # Models outputting logits
        embeddings = outputs.logits
    elif hasattr(outputs, "hidden_states"):  # Models with intermediate hidden states
        embeddings = outputs.hidden_states[-1]  # Use the last layer
    else:
        raise AttributeError("Model output does not contain embeddings in a known attribute.")

    # Compute the mean embedding for the sequence
    embedding = embeddings.mean(dim=1).squeeze().detach().cpu().numpy()

    # Optional: Reduce embedding size for efficiency
    if hasattr(config, "REDUCE_EMBEDDING_SIZE") and config.REDUCE_EMBEDDING_SIZE:
        embedding = reduce_embedding_size(embedding, config.EMBEDDING_SIZE)

    return embedding






def clean_code(code):
    """
    Clean the code snippet by removing extra new lines and spaces.

    Args:
        code (str): The code snippet to clean.

    Returns:
        str: The cleaned code snippet.
    """
    return ' '.join(code.split())
