from typing import List
import tiktoken
import os
import json
import re
import sys
import time

from copy import deepcopy
import xml.etree.ElementTree as ET

from .Base import BaseStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from results.Results import Results
from evaluations.func_evaluate import evaluate_io

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

# KB + Exemplars + Example Planning + Problem Planning + Code Generation + Sample IO testing + Code Improvement


class MapCoder(BaseStrategy):
    def __init__(
        self,
        k: int = 3,
        t: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def _sanitize_xml(self, text: str) -> str:
        """Clean up common XML issues from LLM output."""
        # Strip markdown fences
        if '```xml' in text:
            text = text.replace('```xml', '')
        if '```' in text:
            text = text.replace('```', '')

        # Strip any text before the first XML tag (LLM preamble)
        first_tag = re.search(r'<\w', text)
        if first_tag:
            text = text[first_tag.start():]

        # Close unclosed CDATA sections
        open_count = text.count('<![CDATA[')
        close_count = text.count(']]>')
        if open_count > close_count:
            text += ']]>' * (open_count - close_count)

        # Remove stray CDATA markers that aren't inside tags
        # Fix improperly nested CDATA
        text = re.sub(r'<!\[CDATA\[\s*<!\[CDATA\[', '<![CDATA[', text)

        return text.strip()

    def _regex_fallback_parse(self, response: str) -> dict:
        """
        Last-resort regex extraction when XML parsing completely fails.
        Extracts content between known XML tags.
        """
        result = {}

        # Extract algorithm
        algo_match = re.search(r'<algorithm>(.*?)</algorithm>', response, re.DOTALL)
        result['algorithm'] = algo_match.group(1).strip() if algo_match else "Unable to parse algorithm from response."

        # Extract problems
        problem_blocks = re.findall(r'<problem>(.*?)</problem>', response, re.DOTALL)
        if problem_blocks:
            problems = []
            for block in problem_blocks:
                desc_match = re.search(r'<description>(.*?)</description>', block, re.DOTALL)
                code_match = re.search(r'<code>(.*?)</code>', block, re.DOTALL)
                plan_match = re.search(r'<planning>(.*?)</planning>', block, re.DOTALL)
                problems.append({
                    'description': desc_match.group(1).strip() if desc_match else "No description parsed.",
                    'code': code_match.group(1).strip() if code_match else "",
                    'planning': plan_match.group(1).strip() if plan_match else "No planning parsed.",
                })
            result['problem'] = problems if len(problems) > 1 else problems[0]
        else:
            # Absolute minimum fallback
            result['problem'] = {
                'description': "Fallback: could not parse exemplar from model response.",
                'code': "",
                'planning': "Attempt a direct solution.",
            }

        return result

    def parse_xml(self, response: str) -> dict:
        sanitized = self._sanitize_xml(response)

        # Attempt 1: parse as-is
        try:
            root = ET.fromstring(sanitized)
            return self.xml_to_dict(root)
        except ET.ParseError:
            pass

        # Attempt 2: wrap in <root>
        try:
            root = ET.fromstring('<root>\n' + sanitized + '\n</root>')
            return self.xml_to_dict(root)
        except ET.ParseError:
            pass

        # Attempt 3: strip all CDATA and try again
        try:
            stripped = sanitized.replace('<![CDATA[', '').replace(']]>', '')
            root = ET.fromstring('<root>\n' + stripped + '\n</root>')
            return self.xml_to_dict(root)
        except ET.ParseError:
            pass

        # Attempt 4: regex fallback
        print("[WARNING] XML parsing failed completely. Using regex fallback.", flush=True)
        return self._regex_fallback_parse(response)

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if not code_blocks:
            # No code blocks found despite backticks being present;
            # return the raw response stripped of backtick fences
            return response.replace('```', '').strip()

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

    def run_single_pass(self, item: dict):
        print("", flush=True)

        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{self.data.get_prompt(item)}

# Exemplars:
Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {self.language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in {self.language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>

# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
""",
            },
        ]

        print("\n\n________________________")
        print("Input for knowledge base and exemplars: ")
        print(input_kb_exemplars[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=input_kb_exemplars
        )
        item['api_calls'] = item.get('api_calls', 0) + 1

        # Post processing
        response = self.trim_text(
            response, "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response, "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
        response = self.trim_text(
            response, "# Planning to solve this problem:")
        response = self.trim_text(
            response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')

        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response = self.parse_xml(response)

        try:
            algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{ response['algorithm']}"
        except (KeyError, TypeError):
            print("[WARNING] Could not extract 'algorithm' from parsed response. Using empty.", flush=True)
            algorithm_prompt = "## Relevant Algorithm: Unable to determine from model response."

        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"
        # if type(self.data) != MBPPDataset and type(self.data) != XCodeDataset else ""

        try:
            problem_data = response["problem"]
            if isinstance(problem_data, dict):
                problem_data = [problem_data]
        except (KeyError, TypeError):
            print("[WARNING] Could not extract 'problem' from parsed response. Using fallback exemplar.", flush=True)
            problem_data = [{
                "description": "Fallback: could not parse exemplar from model response.",
                "planning": "Attempt a direct solution.",
            }]

        plannings = []
        for example_no, example in enumerate(problem_data, start=1):
            example_problem = example.get("description", "No description available.")
            example_planning = example.get("planning", "Attempt a direct solution.")

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate a concrete planning to solve the problem.\n# Problem:\n{example_problem}\n# Planning:\n{example_planning}\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print(
                f"Input for our problem planning using example: {example_no}: ")
            print(input_for_problem_planning[0]['content'], flush=True)

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_problem_planning
            )
            item['api_calls'] += 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            # planning = self.parse_xml(planning)
            # planning['confidence'] = int(str(planning['confidence']).strip())

            print("\n\n________________________")
            print("Response from our problem planning: ")
            print(planning, flush=True)

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem and a plan to solve the problem in {self.language}, tell whether the plan is correct to solve this problem.\n\n# Problem:\n{self.data.get_prompt(item)}\n# Planning:\n{planning}\n\n----------------\nImportant: Your response must follow the following xml format-```\n<root>\n<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>\n<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>\n</root>\n```"
                }
            ]

            print("Input for planning verification: ")
            print(input_for_planning_verification[0]['content'], flush=True)

            verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_planning_verification
            )
            item['api_calls'] += 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            verification_res = self.replace_tag(
                verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')

            verification_res = self.parse_xml(verification_res)

            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())

            print("Response from planning verification: ")
            print(verification_res, flush=True)

            plannings.append((
                planning,
                verification_res['confidence'],
                example
            ))

            # if type(self.data) == MBPPDataset and verification_res['confidence'] == 100:
            #     break

        plannings.sort(key=lambda x: x[1], reverse=True)
        # time.sleep(1)

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        for planning_with_ex in plannings:
            planning, confidence, example = planning_with_ex

            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation: ")
            print(input_for_final_code_generation[0]['content'], flush=True)

            code, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_final_code_generation
            )
            item['api_calls'] += 1
            # time.sleep(1)

            code = self.parse_code(code)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from final code generation: ")
            print(code, flush=True)

            response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {i}")
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
                    }
                ]

                print("\n\n________________________")
                print("Input for improving code generation: ")
                print(input_for_improving_code[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                # time.sleep(1)

                code = self.parse_code(response)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n________________________")
                print("Response from improving code generation: ")
                print(response, flush=True)

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok
