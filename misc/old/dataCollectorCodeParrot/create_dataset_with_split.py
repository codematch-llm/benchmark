import os
import csv
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from splitv2_ppc import *

# Define prompts

# PROMPT_TYPE_1 (T1) – Syntactically Identical Code Fragments with Whitespace and Comment Differences
PROMPT_TYPE_1 = "You are provided with a code fragment. Your task is to reproduce an identical copy of the code, but with changes only in whitespace and comments. Do not modify the logic, variable names, or any other syntactical elements."

# PROMPT_TYPE_2 (T2) – Syntactically Identical Code Fragments with Differences in Identifier Names and Literal Values
PROMPT_TYPE_2 = "You are a helpful assistant to make code changes. You are requested to rename the variable names to some sequence, like x1, x2, ..."

# PROMPT_TYPE_3 (T3) – Syntactically Similar Code with Statement-Level Changes
PROMPT_TYPE_3 = "You are provided with a code fragment. Your task is to create a code clone that remains logically equivalent but differs at the statement level. You may add, modify, or remove statements, but the overall functionality should remain the same."

# PROMPT_TYPE_4a (T4a) – Same Functionality, Different Syntax in the Same Programming Language
PROMPT_TYPE_4a = "You are provided with a code fragment. Your task is to create a code clone that remains logically equivalent but differs at the statement level. You may add, modify, or remove statements, but the overall functionality should remain the same."

# PROMPT_TYPE_4b (T4b) – Same Functionality, Different Programming Language
# Will be defined in the code
PROMPT_TYPE_4b = ""

# Supported languages
SUPPORTED_LANGUAGES = [
    'python', 'c', 'cpp', 'php', 'sql', 'ruby', 'javascript', 'java', 'c', 'swift',
    'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy',
    'bash', 'perl', 'r', 'lua', 'haskell', 'clojure'
]

# Load environment variables for OpenAI API key
load_dotenv()
OPENAI_KEY = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=OPENAI_KEY)

def get_codeparrot_data():
    """
    Load the CodeParrot dataset.
    הפונקציה הזו טוענת את הדאטה של קוד-פארוט
    """
    codeparrot_data = load_dataset(
        "codeparrot/github-code", 
        streaming=True, split="train", 
        trust_remote_code=True,
    )
    return codeparrot_data

def generate_types(source_code, prompt):
    """
    Use OpenAI API to generate responses based on the provided prompt and source code.
    פונקציה שמייצרת העתקים של קטעי קוד בהתאם לפרומט שהיא מקבלת בהתאמה לסוג טייפ
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_code}
            ]
        )
        response = completion.choices[0].message['content']
        return response
    except Exception as e:
        print(f"Error generating types: {e}")
        return None

def write_csv_original_code(data, csv_file='original_code.csv'):
    """
    Writes the metadata and code into the 'original_code' CSV.
    פונקציה שמטרתה להכניס את הדאטה לקובץ של קטעי הקוד המקוריים
    """
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['index', 'label', 'domain_label', 'source', 'repo_name', 'path', 'license', 'size', 'code'])
        
        # Write rows
        for item in data:
            writer.writerow([
                item['index'],
                item['label'],
                item['domain_label'],
                'codeparrot',  # Fixed source
                item['repo_name'],
                item['path'],
                item['license'],
                item['size'],
                # item.get('function_name', ''),  # Add function name if exists (todo)
                item['code']
            ])

def write_csv_benchmark_test(data, csv_file='Benchmark_Test.csv'):
    """
    Writes the benchmark test (generated code clones) into the 'Benchmark_Test' CSV.
    פונקציה שמטרתה להכניס את הדאטה לקובץ של ההעתקים שיצרנו
    """
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['test_id', 'original_index', 'title', 'clone_domain_label', 'clone_type', 'code'])
        
        # Write rows
        for item in data:
            writer.writerow([
                item['test_id'],
                item['original_index'],
                item['title'],
                item['clone_domain_label'],
                item['clone_type'],
                item['code']
            ])

def create_labels_json_batch(labels_dict, json_file='labels.json'):
    """
    Append multiple labels to a JSON file in a single operation.
    כאן אנחנו מייצרים קובץ גייסון שעושה התאמה בין הלייבל שהוא מספר לבין הנתיב של הקוד פתוח בגיט
    """
    try:
        # Load existing JSON data or initialize an empty dictionary
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                labels = json.load(file)
        else:
            labels = {}
        
        # Update labels with new data
        labels.update(labels_dict)

        # Write updated labels back to the file
        with open(json_file, 'w') as file:
            json.dump(labels, file, indent=4)

    except Exception as e:
        print(f"Error writing to JSON file: {e}")

def process_codeparrot_data(codeparrot_data, batch_size=100, max_items=None):
    """
    Process CodeParrot dataset and perform the following:
    - Remove comments from the source code
    - Split the code into parts (functions/classes/modules)
    - Generate code clones using OpenAI API
    - Store metadata and code in 'original_code' CSV
    - Store generated code clones in 'Benchmark_Test' CSV
    - Create and update labels in the 'labels.json' file
    
    כאן אנחנו עובר על כל שורה בדאטה של קוד פארוט ומריצים את הפוקנציות בהתאם
    """
    original_code_data = []
    benchmark_test_data = []
    labels_data = {}
    number = 1
         
    for idx, item in enumerate(codeparrot_data):

        # Stop if we've processed the specified number of items
        if max_items is not None and idx >= max_items:
            break

       # Preprocessing stage: remove comments and split by functions
        # source_code = remove_comments(item)
        source_code = item["code"]
        language = item['language']
        processed_code_dict = preprocess_code(source_code, language)
        
        print(source_code)
        print('\n')
        print(processed_code_dict)

        # Process each part in the code separately
        for part_name, part_code in processed_code_dict.items():
            original_code_data.append({
                'index': number,
                'label': f"{item['repo_name']}/{item['path']}@{part_name}",
                'domain_label': language,
                'repo_name': item['repo_name'],
                'path': item['path'],
                'license': item['license'],
                'size': item['size'],
                'part_in_code_name': part_name,
                'code': part_code
            })
            labels_data[number] = f"{item['repo_name']}/{item['path']}@{part_name}"
            number += 1

        continue

        # Continue with the code generation
        for code_entry in original_code_data:
            part_code = code_entry['code']
            part_name = code_entry.get('part_in_code_name', '')

            # Generate code clones based on different prompt types
            type_1_clone = generate_types(part_code, PROMPT_TYPE_1)
            type_2_clone = generate_types(part_code, PROMPT_TYPE_2)
            type_3_clone = generate_types(part_code, PROMPT_TYPE_3)
            type_4a_clone = generate_types(part_code, PROMPT_TYPE_4a)

            # Add Type 1 to 4a clones to benchmark_test_data
            if type_1_clone:
                benchmark_test_data.append({
                    'test_id': f'{number}_1',
                    'original_index': number,
                    'title': f"{item['repo_name']}/{item['path']}@{part_name}",
                    'clone_domain_label': language,
                    'clone_type': 1,
                    'code': type_1_clone
                })
            if type_2_clone:
                benchmark_test_data.append({
                    'test_id': f'{number}_2',
                    'original_index': number,
                    'title': f"{item['repo_name']}/{item['path']}@{part_name}",
                    'clone_domain_label': language,
                    'clone_type': 2,
                    'code': type_2_clone
                })
            if type_3_clone:
                benchmark_test_data.append({
                    'test_id': f'{number}_3',
                    'original_index': number,
                    'title': f"{item['repo_name']}/{item['path']}@{part_name}",
                    'clone_domain_label': language,
                    'clone_type': 3,
                    'code': type_3_clone
                })
            if type_4a_clone:
                benchmark_test_data.append({
                    'test_id': f'{number}_4a',
                    'original_index': number,
                    'title': f"{item['repo_name']}/{item['path']}@{part_name}",
                    'clone_domain_label': language,
                    'clone_type': '4a',
                    'code': type_4a_clone
                })
                
            # Generate Type-4b clones for random target languages
            for target_lang in random.sample(SUPPORTED_LANGUAGES, 3):
                if target_lang == language:
                    continue
                PROMPT_TYPE_4b = f"You are provided with a code fragment. Your task is to implement the same functionality in a different programming language. The new code should perform exactly the same task as the original but using idiomatic syntax and constructs of the target language: {target_lang}."
                type_4b_clone = generate_types(part_code, PROMPT_TYPE_4b)
                
                if type_4b_clone:
                    benchmark_test_data.append({
                        'test_id': f'{number}_4b_{target_lang}',
                        'original_index': number,
                        'title': f"{item['repo_name']}/{item['path']}@{part_name}",
                        'clone_domain_label': target_lang,
                        'clone_type': '4b',
                        'code': type_4b_clone
                    })

        # Process in batches to avoid memory issues
        if (idx + 1) % batch_size == 0:
            write_csv_original_code(original_code_data)
            write_csv_benchmark_test(benchmark_test_data)
            create_labels_json_batch(labels_data)

            # Clear data
            original_code_data.clear()
            benchmark_test_data.clear()
            labels_data.clear()

    # Write any remaining data that didn't complete a full batch
    if original_code_data:
        write_csv_original_code(original_code_data)
    if benchmark_test_data:
        write_csv_benchmark_test(benchmark_test_data)
    if labels_data:
        create_labels_json_batch(labels_data)