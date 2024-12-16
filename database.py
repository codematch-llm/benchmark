import pandas as pd

from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from utils import generate_code_embedding_generic

import config


def initialize_client():
    try:
        client = QdrantClient(url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}")
        print("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize Qdrant client: {e}")
        


def init_vector_db(client, embedding, collection_name, index, id, metadata):
    """
    Initialize the vector database with the given embedding and metadata.

    Args:
        embedding (numpy.ndarray): The embedding to store in the vector database.
        index (str): The index of the embedding.
        id (int): The id of the embedding.
        metadata (dict): The metadata associated with the embedding.
    """
    payload = {"base_code_id": index, **metadata}
    point = PointStruct(id=id, vector=embedding.tolist(), payload=payload)
    client.upsert(collection_name=collection_name, points=[point])
    print(f"Initialized embedding for index {index} with metadata into the vector database.")


# def delete_collection(collection_name):

def create_collection(client, collection_name):
    try:
        client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=config.EMBEDDING_SIZE,
                        distance=models.Distance.COSINE
                    ),
                )
        print(f"Collection '{collection_name}' created successfully.")

    except Exception as e:
        raise Exception(f"Error creating a collection in the vector database: {e}")
        

def clear_vector_db(client, collection_names=None):
    """
    Clear the vector database by deleting the collection.
    """
    try:
        
        if collection_names is None:
            print("Clearing the whole DB...")
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]

        for collection_name in collection_names:
            if client.collection_exists(collection_name=collection_name):
                client.delete_collection(collection_name=collection_name)
                print(f"Collection '{collection_name}' deleted successfully.")
            else:
                print(f"Collection '{collection_name}' does not exist. Skipping deletion.")

    except Exception as e:
        raise Exception(f"Error clearing vector database: {e}")
        


def index_original_os_code(client, csv_file, collection_name, model, tokenizer):
    """
    Index original open source code from the CSV file into the vector database with specific metadata.

    Args:
        csv_file (str): Path to the CSV file containing code examples.
        model_name (str): The name of the model to use for embedding creation.
    """
    try:

        df = pd.read_csv(csv_file)
        for id, row in tqdm(df.iterrows(), desc="Indexing the Originals in the VectorDB...", total=len(df)):
            code = str(row['code']).strip()
            if not code:
                print(f"Skipping ID {id}: code is empty or invalid.")
                continue
            
            base_code_id = row['base_code_id']
            
            # embedding = generate_code_embedding(code, model, tokenizer)
            embedding = generate_code_embedding_generic(code, model, tokenizer)

            if embedding is not None:
                metadata = {
                    'language': row['language'],
                    'task': row['task']
                }
                
                if 'domain' in row:
                    metadata['domain'] = row['domain']
                if 'subdomain' in row:
                    metadata['subdomain'] = row['subdomain']

                init_vector_db(client, embedding, collection_name, base_code_id, id, metadata)
                print(f"ID {id}, Base Code ID: {base_code_id}")

    except Exception as e:
        raise Exception(f"Error indexing examples: {e}")
        