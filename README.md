# CodeMatch Benchmark

The Benchmark repository is the first component of the CodeMatch project, focusing on the initial steps of the workflow. These steps form the foundation for evaluating and selecting the best-performing Large Language Model (LLM) for the system. Below is the diagram highlighting the first three components of the workflow:

<img src="https://github.com/user-attachments/assets/40727442-eb19-47d3-83cd-fd44c774855c" alt="Workflow" width="300">

### Workflow Steps in Benchmark:
1. **Finding/Creating Datasets**: The first step involves gathering or creating datasets tailored for evaluating LLMs. These datasets consist of original code snippets and their clones, covering various programming languages and clone types (e.g., exact, renamed, semantic).
2. **Developing a Benchmark Mechanism**: This step focuses on evaluating multiple LLMs using the created datasets to determine their ability to detect code clones. It involves a detailed scoring mechanism based on similarity metrics.
3. **Training/Fine-Tuning the Chosen LLM**: Although part of the workflow, this step was executed using prompt engineering rather than custom fine-tuning, leveraging the selected LLM's capabilities without additional training.

Out of these three steps, **only the first two are implemented in this repository**:
- Step 1: Dataset Creation - additional information can be found in the `data` [folder](https://github.com/codematch-llm/benchmark/tree/main/data) in this repository.
- Step 2: Benchmark Development - we will delve into this here.

The **benchmark component** of this project is designed to evaluate and identify the best-performing large language model (LLM) for the task of detecting code clones. This benchmark simulates the core functionalities of the final system on a larger scale by:

- Generating embeddings for code snippets using various LLMs.
- Performing two evaluation methods to assess similarity and retrieval effectiveness.
- Producing evaluation CSVs that provide metrics to help determine the best-performing LLM through manual analysis.

### Purpose

The primary goal of the benchmark is to evaluate several LLMs and compare their performance in:

1. **Direct Clone Comparison Test**: Measuring similarity between pairs of code snippets.
2. **Global Clone Search**: Identifying the closest original code snippets from a database for given clones.

By observing the outputs in the generated CSVs, we can decide which model is best suited for the final production system.

---

## Workflow

![Benchmark Workflow](images/becnhmark_workflow.png)

The benchmark follows this general flow:

1. **Load the LLM**: Download and load the chosen Hugging Face LLM and tokenizer locally to this repository if not already available.
2. **Initialize Vector Database**: Populate a Qdrant vector database with embeddings of original code snippets.
3. **Run Evaluation Methods**: Perform the two evaluation methods on benchmark test data.
4. **Generate CSV Outputs**: Create CSV files containing metrics for each evaluation method, organized in the `output` folder. Each model has a dedicated subfolder, and within it, results for each method are separated.
5. **Log Execution Details**: Log the entire execution process for debugging and analysis.

---

## Evaluated Models

The following models have been benchmarked in this project:

- **`codebert-base`** - [Hugging Face](https://huggingface.co/microsoft/codebert-base)
- **`graphcodebert-base`** - [Hugging Face](https://huggingface.co/microsoft/graphcodebert-base)
- **`codet5-base`** - [Hugging Face](https://huggingface.co/Salesforce/codet5-base)
- **`Qwen2.5-Coder-0.5B`** - [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- **`Qwen2.5-Coder-0.5B-pe`** (prompt-engineered version)

---

## Key Features

1. **Logging**:
   - Logs are added throughout the project to capture progress, errors, and important metadata.
   - Logs are stored in a dedicated `logs` folder for detailed debugging and tracking.
2. **Multiprocessing**:
   - Utilized to run both evaluation methods in parallel, reducing runtime.
3. **Local Model Management**:
   - Automatically downloads and caches the selected model locally if not already present, ensuring smooth execution without manual intervention.

---

## Modular Design

The project is designed with modularity in mind, separating core functionalities into distinct modules:

1. **`config.py`**:
   - Handles configuration settings such as file paths, available models, and Qdrant settings.
   - Dynamically determines whether the project is running locally or in Docker to set the appropriate Qdrant host.

2. **`utils.py`**:
   - Provides utility functions for tasks like starting the Qdrant server, loading models, and generating embeddings.

3. **`database.py`**:
   - Manages interactions with the Qdrant vector database, including initializing the client, creating collections, and populating the database with embeddings and metadata.

4. **`metrics.py`**:
   - Implements the two evaluation methods: direct clone comparison and global clone search, calculating similarity scores and generating results.

5. **`main.py`**:
   - Serves as the entry point for the benchmark.
   - Coordinates the workflow, integrating functionalities from all modules to execute the benchmark process.

---

## Reproducing the Benchmark

### Prerequisites

1. **System Requirements**:

   - Docker installed (if running with Docker).
   - Python 3.9 or above (if running locally).
   - Sufficient RAM and CPU to load LLMs and process embeddings.

2. **Dependencies**:

   - See `requirements.txt` for all Python dependencies.

### Running Locally

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd benchmark
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Qdrant (required for vector database):
   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
4. Run the benchmark:
   ```bash
   python main.py
   ```

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t benchmark-image:v1 .
   ```
2. Start Qdrant in the same network:
   ```bash
   docker network create benchmark-network
   docker run --network benchmark-network --name qdrant -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
3. Run the benchmark container:
   ```bash
   docker run --network benchmark-network -it --rm benchmark-image:v1
   ```

---

## Notes for Users

1. **Editing Code**:
   - Clone the repository and edit files locally in your favorite editor (e.g., VS Code).
   - Rebuild the Docker image after making changes:
     ```bash
     docker build -t benchmark-image:v1 .
     ```
2. **Logs**:
   - Logs for the benchmark run will be printed in the console and stored in the `logs` folder for analysis.

3. **Generated CSVs**:
   - Evaluation results are saved as CSV files in a dedicated `output` folder. Each model has a subfolder, and within it, results for the two evaluation methods are stored separately.

