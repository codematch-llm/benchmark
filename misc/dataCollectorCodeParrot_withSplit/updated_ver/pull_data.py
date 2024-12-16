from datasets import load_dataset
from dotenv import load_dotenv
from splitv2_ppc import *
import os
import subprocess

# Supported languages
# SUPPORTED_LANGUAGES = ['python', 'javascript', 'java', 'php', 'go', 'ruby']
SUPPORTED_LANGUAGES = ['python']

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

# Load The Stack dataset and filter for supported languages
def load_stack_for_languages(token, n=5):
    raw_dataset = []
    for language in SUPPORTED_LANGUAGES:
        print(f"Loading data for: {language}")

        # Load the dataset in streaming mode for each language
        ds = load_dataset(
            "bigcode/the-stack", 
            data_dir=f"data/{language}",  # Use the language as the directory
            split="train", 
            streaming=True,
            token=token
        )

        # Limit to only the first `n` rows for each language
        language_data = []
        for idx, sample in enumerate(ds):
            language_data.append(sample)
            if idx + 1 >= n:
                break  # Stop after loading `n` rows for this language

        print(f"Loaded {len(language_data)} rows for {language}.\n")
        raw_dataset.extend(language_data)  # Append the n rows for this language

    return raw_dataset

# Function to create the label for each sample
def create_label(sample, part_name):
    repo_name = sample.get('max_stars_repo_name', 'unknown_repo')
    repo_path = sample.get('max_stars_repo_path', 'unknown_path')
    return f"{repo_name}/{repo_path}@{part_name}"

# Create and return a new dataset after splitting the content
def create_splitted_dataset(raw_dataset):
    splitted_dataset = []

    for sample in raw_dataset:
        content = sample["content"]
        language = (sample.get("lang") or sample.get("language")).lower()

        # Preprocess the content code by splitting it into functions/classes/modules
        processed_code_dict = preprocess_code(content, language)

        # Create new rows for each split part and add them to the splitted_dataset
        for part_name, part_code in processed_code_dict.items():
            new_sample = sample.copy()  # Copy the original sample
            new_sample["content"] = part_code  # Update content with the split part
            new_sample["part_in_code_name"] = part_name  # Add part name from processed code
            
            # Use the create_label function to generate the label
            new_sample["label"] = create_label(sample, part_name)
            
            splitted_dataset.append(new_sample)

    return splitted_dataset

# def print_dataset_headers(splitted_dataset):
#     if splitted_dataset:
#         print("Headers in the splitted_dataset:")
#         # Get the first sample in the splitted_dataset
#         first_sample = splitted_dataset[0]
        
#         # Print the keys (headers) from the first sample
#         for header in first_sample.keys():
#             print(f"- {header}")
#     else:
#         print("The splitted_dataset is empty!")

def main():
    if not HUGGING_FACE_TOKEN:
        print("Please set your Hugging Face token in the environment variable 'HUGGING_FACE_TOKEN'")
        return

    # Load and filter dataset for specific languages
    raw_dataset = load_stack_for_languages(HUGGING_FACE_TOKEN, n=1)
    
    # Print content of each sample
    for idx, sample in enumerate(raw_dataset):
        print(f"Sample {idx + 1} ({sample['lang']}):\n{sample['content']}\n\n")
    
    # Create splitted dataset by calling preprocess_code
    splitted_dataset = create_splitted_dataset(raw_dataset)
    
    # Optionally, print the first few samples from the splitted dataset
    for idx, sample in enumerate(splitted_dataset):
        print(f"Sample {idx + 1} ({sample['lang']}):\nPart name: {sample['part_in_code_name']}\nLabel: {sample['label']}\n{sample['content']}\n")
        
    # print_dataset_headers(splitted_dataset)
        
if __name__ == '__main__':
    main()