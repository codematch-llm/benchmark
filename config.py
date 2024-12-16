import os


import warnings
warnings.filterwarnings(
    "ignore",
    message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta",
)

warnings.filterwarnings("ignore")


# import torch
# # Set device dynamically based on CUDA availability
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # DEVICE = torch.device("cpu")  # Uncomment to force CPU usage


# File paths
#----------------------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ORIGINAL_CODE_CSV = "data/evaluation-datasets/original_code_benchmark_fixed.csv"
BENCHMARK_TEST_CODE_CSV = "data/evaluation-datasets/test_code_benchmark_fixed.csv"
ORIGINAL_CODE_PATH = os.path.join(ROOT_DIR, ORIGINAL_CODE_CSV)
BENCHMARK_TEST_PATH = os.path.join(ROOT_DIR, BENCHMARK_TEST_CODE_CSV)
#----------------------------------------------------------------------------------------


# Model settings
#----------------------------------------------------------------------------------------
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "huggingface_models/")
MODEL_MAX_INPUT_TOKENS = None
#----------------------------------------------------------------------------------------


# Available Models
#----------------------------------------------------------------------------------------
AVAILABLE_MODELS = ["codebert-base", "graphcodebert-base", "codet5-base", "Qwen2.5-Coder-0.5B", "Qwen2.5-Coder-0.5B-pe"]

# MODEL_NAME = "microsoft/codebert-base"    # (Embedding size - 768)  # Default model
# Link - https://huggingface.co/microsoft/codebert-base

# MODEL_NAME = "microsoft/graphcodebert-base"   # (Embedding size - 50265)
# Link - https://huggingface.co/microsoft/graphcodebert-base

# MODEL_NAME = "Salesforce/codet5p-2b"  # (Embedding size - ?)
# Link - https://huggingface.co/Salesforce/codet5p-2b (Too heavy!! - reached 100% ssd when loading)

# MODEL_NAME = "Salesforce/codet5-base" # (Embedding size - 32100)
# Link - https://huggingface.co/Salesforce/codet5-base 

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"  # (Embedding size - 151936)
# Link - https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B
#----------------------------------------------------------------------------------------


# Qdrant settings
#----------------------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_MAX_EMBEDDING_SIZE = 65536   # Qdrants' Maximum Storable Embedding Size
BENCHMARK_COLLECTION_NAME = "originals_embeddings"
#----------------------------------------------------------------------------------------


# Embeddings settings
#----------------------------------------------------------------------------------------
EMBEDDING_SIZE = None   # (initialized later during runtime)
DESIRED_EMBEDDING_SIZE = 768
REDUCE_EMBEDDING_SIZE = False
MAX_TOKEN_INPUT_SIZE = 3000



