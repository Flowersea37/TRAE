import argparse
import logging
import os
from transformers import AutoProcessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from random import seed
from qwen_vl_utils import process_vision_info
from typing import List, Dict
from prompts import GENERATION_PROMPT, CRITIC_PROMPT, REFLECT_PROMPT, SYSTEM_PROMPT

from config import ModelConfig, DatasetType
from dataset_load_utils import load_dataset
import json
import re
import ray
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
class PolicyModel:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        try:
            llm = LLM(
                model=self.model_config.model_name,
                max_model_len=32768,
                gpu_memory_utilization=0.95,
                max_num_seqs=64,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                enforce_eager=True,
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
        return llm

    def _load_processor(self):
        try:
            processor = AutoProcessor.from_pretrained(self.model_config.model_name)
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            raise e
        return processor
    
    def extract_tags(self,text):
        """
        提取字符串中最后一个 <response></response> 和 <critique></critique> 标签内容。
        支持内容包含换行符。
        """
        result = {}

        # 找到所有 <response>...</response>
        responses = re.findall(r"<response>([\s\S]*?)</response>", text, re.MULTILINE)
        if responses:
            result["response"] = responses[-1].strip()

        # 找到所有 <critique>...</critique>
        critiques = re.findall(r"<critique>([\s\S]*?)</critique>", text, re.MULTILINE)
        if critiques:
            result["critique"] = critiques[-1].strip()

        return result


    def generate_answers(self, dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = []
        for item in dataset:
            
            messages.append([{"role": "user", "content": GENERATION_PROMPT.format(question=item["problem"])}])
        
        prompts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,enable_thinking=False
        )
        # torch.manual_seed(100)
        # torch_seed = torch.initial_seed()
        # print(f"当前torch随机种子: {torch_seed}")
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                max_tokens=self.model_config.out_seq_length,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                n=self.model_config.num_sequence,
                # repetition_penalty=1.2,
            ),

        )
        for idx, data in enumerate(dataset):
            data["reasoning_0"] = [
                o.text
                for o in outputs[idx].outputs
            ]
            data["messages"] = [
                {"role" : "user", "content": CRITIC_PROMPT.format(question = data["problem"],solution = data["reasoning_0"][0])}
            ]
        

        return dataset
    def reflect_answers(self, dataset: List[Dict[str, str]], time) -> List[Dict[str, str]]:
        messages = []
        reflect_prompt = """The solution you need to reflect is in the previous step and enclosed in <response></response>. Your task is to generation a critique to the solution, which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the previous solution, and referred from the critique.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format:
<critique>your critique to the solution</critique>
<response>the new solution refined from the initial solution</response>
        """
        for item in dataset:
            message = item["messages"]
            if time > 0:
                message.append({
                    "role":'user',
                    'content':reflect_prompt
                })
            messages.append(message)
        
        prompts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking = False
        )

        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                max_tokens=self.model_config.out_seq_length,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                n=self.model_config.num_sequence,
                # repetition_penalty=1.2,
            ),
        )
        for idx, data in enumerate(dataset):
            data[f"reflection_{time+1}"] = [
                o.text
                for o in outputs[idx].outputs
            ]
            reasonings = []
            critiques = []
            for reflection in data[f"reflection_{time+1}"]:
                tags = self.extract_tags(reflection)
                response = tags.get("response",reflection)
                critique = tags.get("critique","")
                reasonings.append(response)
                critiques.append(critique)
            data[f"reasoning_{time+1}"] = reasonings
            data[f"critique_{time+1}"] = critiques
            # data["messages"] = [
            #     {"role" : "user", "content": CRITIC_PROMPT.format(question = data["problem"],solution = data[f"reasoning_{time+1}"][0])}
            # ]
            
            data['messages'].append({
                "role":'assistant',
                'content':outputs[idx].outputs[0].text
            })

        return dataset
    
   



class RewardModel:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer() 

    def _load_model(self):
        try:
            llm = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name,
                device_map="auto",
                num_labels =1,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=False,
            ).eval()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
        return llm

    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise e
        return tokenizer

    def reward_answers(self, data, time):
        scores = []
        for item in data[f"reasoning_{time}"]:

            content = []
            

            messages=[
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": item}
            ]

            # logger.info(data["problem"])
            # logger.info(item)
        
            prompts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )

            input_ids = self.tokenizer.encode(
                prompts, 
                return_tensors="pt", 
                add_special_tokens=False,
                max_length=16384
            ).to(self.model.device)
            if "skywork" in self.model_config.model_name.lower():
                with torch.no_grad():
                    score = self.model(input_ids).logits[0][0].item()
                scores.append(score)
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                scores.append(outputs[0][0].item())
        
        # breakpoint()

        return scores