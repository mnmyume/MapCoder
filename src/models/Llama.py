import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .Base import BaseModel


class Llama(BaseModel):
    def __init__(
            self,
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            device=None,
            temperature=0.0,
            top_p=0.9,
            input_price=0.04,
            output_price=0.14,
            **kwargs):
        """
        Initialize Llama model.

        Args:
            model_name_or_path: Hugging Face model ID or local path
            device: ('cuda', 'cpu', 'mps'), auto by default
        """
        super().__init__(**kwargs)

        print(f"Loading Llama model from {model_name_or_path} to {self.device}...")

        # model params
        self.temperature = temperature
        self.top_p = top_p
        self.input_price = input_price
        self.output_price = output_price

        # load 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        # # load 8-bit quantization
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        # )

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype="auto",
        ).eval()

    def prompt(self, processed_input: list[dict]):
        """
        Generate a response from the Llama model.

        Args:
            processed_input: OpenAI style messages
                             [{"role": "user", "content": "hello"}, ...]

        Returns:
            (response_text, prompt_tokens, completion_tokens, price)
        """
        # use chat template to format input
        text = self.tokenizer.apply_chat_template(
            processed_input,
            tokenize=False,
            add_generation_prompt=True
        )

        # tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)

        input_length = model_inputs.input_ids.shape[1]

        # generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=8192,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # get generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # decode
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # calculate token number
        prompt_tokens = input_length
        completion_tokens = len(generated_ids[0])

        # calculate price
        price = (prompt_tokens * self.input_price + completion_tokens * self.output_price) / 1_000_000

        return response, prompt_tokens, completion_tokens, price
