import config
import time
from utils import start_qdrant_server, load_model_with_retry, load_huggingface_model, get_embedding_and_token_size
from database import clear_vector_db, index_original_os_code, initialize_client, create_collection
from metrics import direct_clone_comparison_test, global_clone_search


def main():

    start_time = time.time()



    print(f"Using model: {config.MODEL_NAME}")
    # model, tokenizer = load_model_with_retry(MODEL_NAME, MODEL_CACHE_DIR)
    model, tokenizer = load_huggingface_model(config.MODEL_NAME, config.MODEL_CACHE_DIR)

    # Get the embedding size
    config.EMBEDDING_SIZE, config.MODEL_MAX_INPUT_TOKENS = get_embedding_and_token_size(model, tokenizer)
    print(f"Embedding size for model '{config.MODEL_NAME}': {config.EMBEDDING_SIZE}")
    print(f"Maximum token input size: {config.MODEL_MAX_INPUT_TOKENS}")

    start_qdrant_server()

    client = initialize_client()

    # clear_vector_db(client)

    # create_collection(client, config.BENCHMARK_COLLECTION_NAME)


    # index_original_os_code(client, config.ORIGINAL_CODE_PATH, config.BENCHMARK_COLLECTION_NAME, model, tokenizer)

    # # Run tests
    # direct_clone_comparison_test(client, config.BENCHMARK_TEST_PATH, model, tokenizer, config.MODEL_NAME.split('/')[1])
    global_clone_search(client, config.BENCHMARK_TEST_PATH, model, tokenizer, config.MODEL_NAME.split('/')[1])

    total_time = (time.time() - start_time) / 60
    print(f"Total time taken: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()
