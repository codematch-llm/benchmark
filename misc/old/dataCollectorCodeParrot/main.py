import os
import csv
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# Define prompts
PROMPT_TYPE_1 = ""
PROMPT_TYPE_2 = "You are a helpful assistant to make code changes. You are requested to rename the variable names to some sequence, like x1, x2, ..."
PROMPT_TYPE_3 = ""
PROMPT_TYPE_4 = ""

# Load environment variables for OpenAI API key
load_dotenv()
OPENAI_KEY = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=OPENAI_KEY)

def get_codeparrot_data():
    """
    Load the CodeParrot dataset.
    הפונקציה הזו טוענת את הדאטה של קוד-פארוט
    """
    codeparrot_data = load_dataset("codeparrot/github-code", split="train")
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

def remove_comments(item):
    """
    Removes comments from source code based on the programming language.
    פונקציה שמטרתה להסיר את ההערות מהקוד במידה ויש (אולי לא נשתמש, צריך לחשוב)
    """
    source_code = item["code"].strip()
    domain_label = item["language"]
    c_style_pattern = r'//.*|/\*[\s\S]*?\*/'

    if domain_label in ['python', 'c', 'cpp', 'php', 'sql', 'ruby']:
        pattern = r"(#.*)|(\"{3}[\s\S]*?\"{3})|(\"[\s\S]*?\")|(\/\/.*)|(\/\*[\s\S]*?\*\/)"
        source_code = re.sub(pattern, '', source_code)
    elif domain_label in ['javascript', 'java', 'swift', 'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy']:
        source_code = re.sub(c_style_pattern, '', source_code, flags=re.DOTALL)
    elif domain_label in ['html', 'xml']:
        source_code = re.sub(r'<!--.*?-->', '', source_code, flags=re.DOTALL)
    elif domain_label in ['bash', 'perl', 'r']:
        source_code = re.sub(r'#.*', '', source_code)
    elif domain_label == 'lua':
        source_code = re.sub(r'--.*|(?s)--\[\[.*?]]', '', source_code)
    elif domain_label == 'haskell':
        source_code = re.sub(r'--.*|\{-[\s\S]*?-}', '', source_code)
    elif domain_label == 'clojure':
        source_code = re.sub(r';.*', '', source_code)
    
    source_code = source_code.strip()
    source_code = source_code.encode('ascii', 'ignore').decode('utf-8')
    return source_code

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

def process_codeparrot_data(codeparrot_data, batch_size=100):
    """
    Process CodeParrot dataset and perform the following:
    - Remove comments from the source code
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
        # Clean up code by removing comments
        source_code = remove_comments(item)

        # Prepare original code data
        original_code_data.append({
            'index': number,
            'label': number,
            'domain_label': item['language'],
            'repo_name': item['repo_name'],
            'path': item['path'],
            'license': item['license'],
            'size': item['size'],
            'code': item['code']
        })

        # Add to labels for JSON
        labels_data[number] = f"{item['repo_name']}/{item['path']}"

        # Generate code clones based on different prompt types
        type_1_clone = generate_types(source_code, PROMPT_TYPE_1)
        type_2_clone = generate_types(source_code, PROMPT_TYPE_2)
        type_3_clone = generate_types(source_code, PROMPT_TYPE_3)
        type_4_clone = generate_types(source_code, PROMPT_TYPE_4)

        # Prepare benchmark test data
        if type_1_clone:
            benchmark_test_data.append({
                'test_id': f'{number}_1',
                'original_index': number,
                'title': f"{item['repo_name']}/{item['path']}",
                'clone_domain_label': item['language'],
                'clone_type': 1,
                'code': type_1_clone
            })
        if type_2_clone:
            benchmark_test_data.append({
                'test_id': f'{number}_2',
                'original_index': number,
                'title': f"{item['repo_name']}/{item['path']}",
                'clone_domain_label': item['language'],
                'clone_type': 2,
                'code': type_2_clone
            })
        if type_3_clone:
            benchmark_test_data.append({
                'test_id': f'{number}_3',
                'original_index': number,
                'title': f"{item['repo_name']}/{item['path']}",
                'clone_domain_label': item['language'],
                'clone_type': 3,
                'code': type_3_clone
            })
        if type_4_clone:
            benchmark_test_data.append({
                'test_id': f'{number}_4',
                'original_index': number,
                'title': f"{item['repo_name']}/{item['path']}",
                'clone_domain_label': item['language'],
                'clone_type': 4,
                'code': type_4_clone
            })

        number += 1

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

def main():
    # Load data
    codeparrot_data = get_codeparrot_data()

    # Process data with defined batch size
    process_codeparrot_data(codeparrot_data, batch_size=100)

if __name__ == '__main__':
    main()
