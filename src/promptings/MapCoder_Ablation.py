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
from .MapCoder import MapCoder
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


# =========================================
# 1. MapCoder (w/o R, P): Debugging(V)
# =========================================
class MapCoder_wo_RP(MapCoder):
    """
    Ablation: No Retrieval, No Planning.
    Equivalent to "Reflexion": Direct code generation followed by debugging.
    """

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Initialize token counters
        pr_tok = 0
        com_tok = 0

        # Sample IO prompt setup
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        code = ""

        # Loop for k attempts (Coding Agent) without Planning/Retrieval
        for i in range(self.k):

            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation: ")
            print(input_for_final_code_generation[0]['content'], flush=True)

            code, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_final_code_generation
            )
            item['api_calls'] = item.get('api_calls', 0) + 1
            # time.sleep(1)

            code = self.parse_code(code)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from final code generation: ")
            print(code, flush=True)

            # Response structure modified to exclude Planning
            response = f"## Code:\n```\n{code}\n```"
            passed = False

            # Debugging Agent Loop
            for j in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {j}")

                # Debugging prompt modified to exclude Algorithm and Modified Planning instructions
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the {self.language} code inside ``` block to solve this problem."
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


# =========================================
# 2. MapCoder (w/o R): Planning(V), Debugging(V)
# =========================================
class MapCoder_wo_R(MapCoder):
    """
    Ablation: No Retrieval agent.
    Performs Zero-shot Planning on the problem, then Code Generation, then Debugging.
    """

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Initialize token counters
        pr_tok = 0
        com_tok = 0

        # Sample IO prompt setup
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        # Container for generated plans
        plannings = []

        # Planning Agent Loop (Iterates k times to generate plans without retrieval exemplars)
        for i in range(self.k):
            # Planning prompt modified to exclude exemplar and algorithm references
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate a concrete planning to solve the problem.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print(
                f"Input for our problem planning (Attempt {i + 1}): ")
            print(input_for_problem_planning[0]['content'], flush=True)

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_problem_planning
            )
            item['api_calls'] = item.get('api_calls', 0) + 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from our problem planning: ")
            print(planning, flush=True)

            # Confidence Generation Prompt
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
                verification_res['confidence']
            ))

        # Sort plans by confidence
        plannings.sort(key=lambda x: x[1], reverse=True)
        # time.sleep(1)

        code = ""

        # Coding and Debugging Loop (Uses the generated plans)
        for planning_data in plannings:
            planning, confidence = planning_data

            # Final Code Gen Prompt (Modified to exclude algorithm reference)
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
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

            # Debugging Agent Loop
            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {i}")

                # Debugging Prompt (Modified to exclude algorithm reference)
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
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


# =========================================
# 3. MapCoder (w/o R, D): Planning(V)
# =========================================
class MapCoder_wo_RD(MapCoder):
    """
    Ablation: No Retrieval, No Debugging.
    Equivalent to "Self-Planning": Zero-shot planning followed by code generation.
    """

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Initialize token counters
        pr_tok = 0
        com_tok = 0

        # Sample IO prompt setup
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        # Container for generated plans
        plannings = []

        # Planning Agent Loop (Iterates k times to generate plans without retrieval exemplars)
        for i in range(self.k):
            # Planning prompt modified to exclude exemplar and algorithm references
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate a concrete planning to solve the problem.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print(
                f"Input for our problem planning (Attempt {i + 1}): ")
            print(input_for_problem_planning[0]['content'], flush=True)

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_problem_planning
            )
            item['api_calls'] = item.get('api_calls', 0) + 1
            # time.sleep(1)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print("Response from our problem planning: ")
            print(planning, flush=True)

            # Confidence Generation Prompt
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
                verification_res['confidence']
            ))

        # Sort plans by confidence
        plannings.sort(key=lambda x: x[1], reverse=True)
        # time.sleep(1)

        code = ""

        # Coding Loop (Uses the generated plans, but no Debugging Loop)
        for planning_data in plannings:
            planning, confidence = planning_data

            # Final Code Gen Prompt (Modified to exclude algorithm reference)
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
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

            # Evaluate (Debugging Agent Loop Removed)
            passed, test_log = self.data.evaluate_sample_io(
                item,
                code,
                self.language
            )

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok


# =========================================
# 4. MapCoder (w/o P): Retrieval(V), Debugging(V)
# =========================================
class MapCoder_wo_P(MapCoder):
    """
    Ablation: No Planning agent.
    Retrieves examples and algorithm. Skips explicit planning step.
    Generates code conditioned on retrieved algorithm, then Debugs.
    """

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
            response,
            "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response,
            "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
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

        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        # Iterate through retrieved examples (k paths) directly, skipping Planning Agent
        for example in response["problem"]:
            example_problem = example["description"]
            example_code = example["code"]

            # Construct a prompt using the retrieved example as a reference instead of a plan
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Reference Problem:\n{example_problem}\n## Reference Code:\n{example_code}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation (using retrieved example): ")
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

            # Modified response structure for Debugging (removed Planning)
            debug_context = f"## Code:\n```\n{code}\n```"
            passed = False

            # Debugging Agent Loop
            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {i}")

                # Debugging prompt modified to remove Planning and Modified Planning
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n{debug_context}\n## Test Report:\n{test_log}\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the {self.language} code inside ``` block to solve this problem."
                    }
                ]

                print("\n\n________________________")
                print("Input for improving code generation: ")
                print(input_for_improving_code[0]['content'], flush=True)

                response_debug, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                # time.sleep(1)

                code = self.parse_code(response_debug)
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n________________________")
                print("Response from improving code generation: ")
                print(response_debug, flush=True)

                # Update debug context with the new code
                debug_context = f"## Code:\n```\n{code}\n```"

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok


# =========================================
# 5. MapCoder (w/o D): Retrieval(V), Planning(V)
# =========================================
class MapCoder_wo_D(MapCoder):
    """
    Ablation: No Debugging agent.
    """

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
            response,
            "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response,
            "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
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

        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

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

            # Debugging Agent Removed
            # Only checking if the code passed to break or continue to the next plan
            passed, test_log = self.data.evaluate_sample_io(
                item,
                code,
                self.language
            )

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok


# =========================================
# 6. MapCoder (w/o P, D): Retrieval(V)
# =========================================
class MapCoder_wo_PD(MapCoder):
    """
    Ablation: No Planning, No Debugging.
    Retrieves examples/algorithm, then generates code directly conditioned on retrieval. No specific plan, no debugging loop.
    """

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
            response,
            "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response,
            "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
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

        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        # Iterate through retrieved examples (k paths) directly, skipping Planning
        for example in response["problem"]:
            example_problem = example["description"]
            example_code = example["code"]

            # Construct a prompt using the retrieved example as a reference instead of a plan
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(item)}\n## Reference Problem:\n{example_problem}\n## Reference Code:\n{example_code}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            print("\n\n________________________")
            print("Input for final code generation (using retrieved example): ")
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

            # Debugging Agent Removed
            # Only checking if the code passed to break or continue to the next example
            passed, test_log = self.data.evaluate_sample_io(
                item,
                code,
                self.language
            )

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok