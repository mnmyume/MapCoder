import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Base import BaseModel


class Qwen(BaseModel):
    def __init__(self, model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507", device=None, **kwargs):
        """
        Initialize Qwen model。

        Args:
            model_name_or_path: Hugging Face model ID or local path
            device: ('cuda', 'cpu', 'mps')，auto by default
        """
        super().__init__(**kwargs)

        # auto detect device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        print(f"Loading Qwen model from {model_name_or_path} to {self.device}...")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # load model
        # if low gpu memory: load_in_4bit=True (bitsandbytes needed)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype="auto"
        ).eval()

    def prompt(self, processed_input: list[dict]):
        """
        Reasoning

        Args:
            processed_input: OpenAI style message
                             [{"role": "user", "content": "hello"}, ...]

        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        # use chat template to format input
        text = self.tokenizer.apply_chat_template(
            processed_input,
            tokenize=False,
            add_generation_prompt=True
        )

        # tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        input_length = model_inputs.input_ids.shape[1]

        # generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
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

        return response, prompt_tokens, completion_tokens