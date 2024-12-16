import pandas as pd
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import os
import subprocess
import torch
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchAny
import time

# Define constants
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODET5_MODEL_NAME = "Salesforce/codet5-base-multi-sum"
GRAPH_CODEBERT_MODEL_NAME = "microsoft/graphcodebert-base"
CSV_NAME = "examples_fix.csv"
TEST_CSV_NAME = "checks_fix.csv"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_CSV = os.path.join(SCRIPT_DIR, CSV_NAME)
TEST_CSV = os.path.join(SCRIPT_DIR, TEST_CSV_NAME)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "code_embeddings"
EMBEDDING_SIZE = 768

# Initialize Qdrant client
client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

def start_qdrant_server():
    """
    Start the Qdrant server using Docker in a new terminal window.
    """
    docker_command = (
        'docker run -p 6333:6333 -p 6335:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant'
    )
    try:
        # Open a new terminal and run the Docker command
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
            # Load the model based on the model name
            if model_name == CODEBERT_MODEL_NAME or model_name == GRAPH_CODEBERT_MODEL_NAME:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            elif model_name == CODET5_MODEL_NAME:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = T5EncoderModel.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            return model, tokenizer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise e
            
   

def create_embedding(code, model, tokenizer):
    """
    Create an embedding for the given code using the specified model.

    Args:
        code (str): The code snippet to create an embedding for.
        model_name (str): The name of the model to use for embedding creation.
        cache_dir (str): The directory to cache and load models from.

    Returns:
        numpy.ndarray: The embedding of the code snippet.
    """

    # create embedding using the LLM model
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    
    return embedding

def init_vector_db(embedding, index, id):
    """
    Initialize the vector database with the given embedding.

    Args:
        embedding (numpy.ndarray): The embedding to store in the vector database.
        index (str): The index of the embedding.
        id (int): The id of the embedding.
    """
    point = PointStruct(
            id=id,
            vector=embedding.tolist(),
            payload={"index": index}
        )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print(f"Initialized embedding for index {index} into the vector database.")

def clean_code(code):
    """
    Clean the code snippet by removing extra new lines and spaces.

    Args:
        code (str): The code snippet to clean.

    Returns:
        str: The cleaned code snippet.
    """
    cleaned_code = ' '.join(code.split())
    return cleaned_code

def index_examples(csv_file, model, tokenizer):
    """
    Index examples from the CSV file into the vector database.

    Args:
        csv_file (str): Path to the CSV file containing code examples.
        model_name (str): The name of the model to use for embedding creation.
        cache_dir (str): The directory to cache and load models from.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Iterate over each row in the DataFrame
        for id, row in df.iterrows():
            code = row['code']

            # Ensure the code is not NaN or None
            if pd.isna(code):
                print(f"Skipping ID {id}: code is NaN or None")
                continue
            
            # Ensure that the code is treated as a string
            code = str(code).strip()
            
            # Check if the code is still not a valid string after cleaning
            if not code:
                print(f"Skipping ID {id}: code is an empty string after cleaning")
                continue
            
            index = row['index']
            embedding = create_embedding(code, model, tokenizer)
            if embedding is not None:
                init_vector_db(embedding, index, id)  # Store in vector db
                print(f"ID {id}, Index: {index}, Embedding: {embedding}")
                
                embedding_size = embedding.shape[0]
                print(f"Embedding size: {embedding_size}")
    except Exception as e:
        print(f"Error indexing examples: {e}")
        
def clear_vector_db():
    """
    Clear the vector database by deleting the collection.
    """
    try:
        # Check if the collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME in collection_names:
            # If the collection exists, delete it
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist. Skipping deletion.")
        
        # Create a new collection with the necessary vector configuration
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_SIZE,  # Update this size according to your model's embedding size
                distance=models.Distance.COSINE
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Error clearing vector database: {e}")

def run_tests(test_csv_file, model, tokenizer, file_name):
    """
    Run tests to compare embeddings against expected results and save results to a CSV file.

    Args:
        test_csv_file (str): Path to the CSV file containing test cases.
        model_name (str): The name of the model to use for embedding creation.
    """
    try:
        print("\n")
        df_tests = pd.read_csv(test_csv_file)
        
        total_passed = 0
        total_failed = 0
        results = []

        for _, row in df_tests.iterrows():
            test_id = row['index']
            code = row['code']
            title = row['Description']
            expected_result_indexes = str(row['insperation_index_db']).split(',')  # Splitting into a list
            domain_label = row['domain_label']

            embedding = create_embedding(code, model, tokenizer)
            if embedding is not None:
                # Search for the closest embedding in Qdrant
                search_result = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=embedding,
                    limit=1,
                )
                
                closest_embedding_id = search_result[0].id
                closest_embedding_index = search_result[0].payload["index"]
                similarity_score = search_result[0].score

                # Print the results
                print(f"Test ID: {test_id}, Title: {title}")
                print(f"Closest Embedding ID: {closest_embedding_id}, Closest Embedding Index: {closest_embedding_index}, Similarity Score: {similarity_score}")
                print(f"The expected result indexes: {expected_result_indexes}")

                # Print similarity scores for expected result IDs
                print(f"Similarity scores with expected result IDs:")
                for expected_index in expected_result_indexes:
                    if expected_index:
                        search_result = client.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=embedding,
                            limit=1,
                            query_filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="index",
                                        match=MatchAny(any=[expected_index])
                                    )
                                ]
                            )
                        )
                        if search_result:
                            score = search_result[0].score
                            print(f"Expected Index: {expected_index}, Similarity Score: {score}")
                        else:
                            score = "Not found"
                            print(f"Expected Index: {expected_index}, Similarity Score: Not found")
                    else:
                        score = "Not found"
                        print(f"Expected Index: {expected_index}, Similarity Score: Not found")
                    
                    results.append({
                        "Test ID": test_id,
                        "Title": title,
                        "Domain Label": domain_label,
                        "Closest Embedding Index": closest_embedding_index,
                        "Similarity Score": similarity_score,
                        "Expected Index": expected_index,
                        "Expected Similarity Score": score
                    })
                
                # Check if the closest embedding index is among the expected results
                if closest_embedding_index in expected_result_indexes:
                    total_passed += 1
                else:
                    total_failed += 1

                print("\n")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{file_name}_output.csv", index=False)

        # Append total passed and failed counts to the CSV file
        with open(f"{file_name}_output.csv", 'a') as f:
            f.write(f"Total passed:,{total_passed}\n")
            f.write(f"Total failed:,{total_failed}\n")

        print(f"Total passed: {total_passed}")
        print(f"Total failed: {total_failed}")
    except Exception as e:
        print(f"Error running tests: {e}")

def main():
    start_time = time.time()  # Start the timer
    start_qdrant_server()
    clear_vector_db()
    
    # Choose model: CODEBERT_MODEL_NAME, CODET5_MODEL_NAME, GRAPH_CODEBERT_MODEL_NAME
    model_name = CODEBERT_MODEL_NAME
    cache_dir = "./huggingface_models"  # specify the directory to cache models
    file_name = model_name.split('/')[1]
    print(f"We are using the {model_name} model")
    
    model, tokenizer = load_model_with_retry(model_name, cache_dir)
    
    index_examples(EXAMPLES_CSV, model, tokenizer)
    start_test_time = time.time()
    run_tests(TEST_CSV, model, tokenizer, file_name)
    client.close()
    
    end_time = time.time()  # End the timer
    total_time = (end_time - start_time) / 60  # Calculate the total time taken in minutes
    print(f"Total time taken: {total_time:.2f} minutes")  # Print the total time taken
    
    total_test_time = (end_time - start_test_time) / 60 
    print(f"Total test time taken: {total_test_time:.2f} minutes") 

if __name__ == '__main__':
    main()
