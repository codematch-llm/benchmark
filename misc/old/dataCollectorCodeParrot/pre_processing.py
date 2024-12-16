import re

# Supported languages
SUPPORTED_LANGUAGES = [
    'python', 'c', 'cpp', 'php', 'sql', 'ruby', 'javascript', 'java', 'c', 'swift',
    'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy',
    'bash', 'perl', 'r', 'lua', 'haskell', 'clojure'
]

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

def extract_function_names(source_code, domain_label):
    """
    Extracts function names from source code based on the programming language.
    לוקח את שמות הפונקציות מהקוד
    """
    functions = []
    
    # Define regex patterns for different programming languages
    patterns = {
        'python': r'def\s+(\w+)\s*\(.*?\):',
        'c': r'\b\w+\s+(\w+)\s*\(.*?\)\s*\{',
        'cpp': r'\b\w+\s+(\w+)\s*\(.*?\)\s*\{',
        'java': r'\b\w+\s+(\w+)\s*\(.*?\)\s*\{',
        'javascript': (
            r'(?:function\s+(\w+)\s*\(.*?\))|'  # Function declarations
            r'(?:[\w$]+\s*=\s*function\s*\(.*?\)\s*\{)|'  # Function expressions
            r'(?:[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{)|'  # Arrow functions
            r'(?:\b(\w+)\s*\(.*?\)\s*\{)|'  # Method shorthand in objects
            r'(?:d\((?:function|gs)\s*\(.*?\)\s*\{)'  # Methods in Object.defineProperties (like add, clear, delete)
        ),
        'typescript': (
            r'(?:function\s+(\w+)\s*\(.*?\))|'  # Function declarations
            r'(?:[\w$]+\s*=\s*function\s*\(.*?\)\s*\{)|'  # Function expressions
            r'(?:[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{)|'  # Arrow functions
            r'(?:\b(\w+)\s*\(.*?\)\s*\{)'  # Method shorthand in objects
        ),
        'kotlin': r'\bfun\s+(\w+)\s*\(.*?\)\s*{',
        'scala': r'\bdef\s+(\w+)\s*\(.*?\)\s*\{',
        'swift': r'\bfunc\s+(\w+)\s*\(.*?\)\s*->',
        'go': r'\bfunc\s+(\w+)\s*\(.*?\)\s*\{',
        'rust': r'\bfn\s+(\w+)\s*\(.*?\)\s*\{',
        'dart': (
            r'(?:\b(\w+)\s*\(.*?\)\s*\{)|'  # Regular function
            r'(?:\b[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{)'  # Arrow functions
        ),
        'groovy': r'\bdef\s+(\w+)\s*\(.*?\)\s*\{',
        'php': r'function\s+(\w+)\s*\(.*?\)\s*\{',
        'sql': r'CREATE\s+(FUNCTION|PROCEDURE)\s+(\w+)\s*\(.*?\)',
        'ruby': r'def\s+(\w+)\s*\(.*?\)',
        'bash': r'(\w+)\s*\(\)\s*\{',
        'perl': r'(\w+)\s*\(\)\s*\{',
        'r': r'(\w+)\s*\(\)\s*\{',
        'lua': r'function\s+(\w+)\s*\(.*?\)',
        'haskell': r'(\w+)\s*::\s*.*',
        'clojure': r'\(defn\s+(\w+)'
    }

    # Select the appropriate pattern based on the domain_label (programming language)
    pattern = patterns.get(domain_label)

    if pattern:
        # Apply the appropriate regex pattern to extract function names
        matches = re.findall(pattern, source_code)
        functions = matches if domain_label != 'sql' else [match[1] for match in matches]  # Handle SQL case

    return functions


def split_code_by_functions(source_code, domain_label):
    """
    Splits the source code into a dictionary of {function_name: function_code}.
    Handles multiple languages by using language-specific rules.
    מחלק את הקוד למקטעי פונקציות
    """
    function_dict = {}

    # First, extract all function names using the extract_function_names function
    function_names = extract_function_names(source_code, domain_label)

    # Define patterns to capture full functions for each language
    patterns = {
        'python': r'def\s+{function_name}\s*\(.*?\):[\s\S]*?(?=\ndef|\Z)',
        'c': r'\b\w+\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'cpp': r'\b\w+\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'java': r'\b\w+\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'javascript': (
            r'(?:function\s+\w+\s*\(.*?\)[\s\S]*?\})|'  # Function declarations
            r'(?:[\w$]+\s*=\s*function\s*\(.*?\)\s*\{[\s\S]*?\})|'  # Function expressions
            r'(?:[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{[\s\S]*?\})|'  # Arrow functions
            r'(?:\b\w+\s*\(.*?\)\s*\{[\s\S]*?\})'  # Method shorthand in objects
        ),
        'typescript': (
            r'(?:function\s+{function_name}\s*\(.*?\)[\s\S]*?\})|'  # Function declarations
            r'(?:[\w$]+\s*=\s*function\s*\(.*?\)\s*\{[\s\S]*?\})|'  # Function expressions
            r'(?:[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{[\s\S]*?\})|'  # Arrow functions
            r'(?:\b{function_name}\s*\(.*?\)\s*\{[\s\S]*?\})'  # Method shorthand in objects
        ),
        'kotlin': r'fun\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'scala': r'def\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'swift': r'func\s+{function_name}\s*\(.*?\)\s*->[\s\S]*?\}',
        'go': r'func\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'rust': r'fn\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'dart': (
            r'(?:\b{function_name}\s*\(.*?\)\s*\{[\s\S]*?\})|'  # Regular function
            r'(?:\b[\w$]+\s*=\s*\(.*?\)\s*=>\s*\{[\s\S]*?\})'  # Arrow functions
        ),
        'groovy': r'def\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'php': r'function\s+{function_name}\s*\(.*?\)\s*\{[\s\S]*?\}',
        'sql': r'CREATE\s+(FUNCTION|PROCEDURE)\s+{function_name}\s*\(.*?\)[\s\S]*?END',
        'ruby': r'def\s+{function_name}\s*\(.*?\)[\s\S]*?end',
        'bash': r'{function_name}\s*\(\)\s*\{[\s\S]*?\}',
        'perl': r'{function_name}\s*\(\)\s*\{[\s\S]*?\}',
        'r': r'{function_name}\s*\(\)\s*\{[\s\S]*?\}',
        'lua': r'function\s+{function_name}\s*\(.*?\)[\s\S]*?end',
        'haskell': r'{function_name}\s*::\s*[\s\S]*?where[\s\S]*?{function_name}\s*=',
        'clojure': r'\(defn\s+{function_name}\s*\[.*?\][\s\S]*?\)'
    }

    # Select the appropriate pattern based on the domain_label (programming language)
    pattern_template = patterns.get(domain_label)

    if not pattern_template:
        # If no pattern is found for the language, return the entire source as one function
        return {None: source_code}

    # Loop over the function names and extract corresponding function code
    for function_name in function_names:
        # Fill the function name into the regex pattern
        pattern = pattern_template.format(function_name=function_name)

        # Find the function code
        match = re.search(pattern, source_code)
        if match:
            # Add the function and its code to the dictionary
            function_dict[function_name] = match.group(0)

    return function_dict