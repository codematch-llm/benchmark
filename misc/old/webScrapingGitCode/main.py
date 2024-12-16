import os
import subprocess
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.environ.get('OPENAI_KEY')
github_urls = '''
https://github.com/jossef/vmwc
'''
client = OpenAI(api_key=OPENAI_KEY)


def main():
    for url in github_urls.split():
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.call(['git', 'clone', url], cwd=temp_dir)
            
            # 2 iterate only over the source code files with os.walk
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path) as f:
                            source_code = f.read()
                            
                            completion = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant to make make code changes. You are requested to rename the variable names to some sequence, like x1, x2, ..."},
                                    {
                                        "role": "user",
                                        "content": source_code
                                    }
                                    ]
                                )
                            
                            response = completion.choices[0].message
                            print(response)

                            # TODO calculate embedding







        # 3 for each file
        # 4 use openai to make the different types of code snippets (1,2,3,4)



        # print the result
        print(url)


if __name__ == '__main__':
    main()
