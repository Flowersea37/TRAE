import logging
from typing import List, Dict, Optional

from datasets import load_dataset as ld
import json
from config import DatasetType, DatasetConfig
import os


logger = logging.getLogger(__name__)

def load_dataset(dataset_name):
    dataset = []
    if "math-500" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/MATH-500/test.jsonl"
        with open(dataset_path,"r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    "problem" : data["problem"],
                    "answer" : data["answer"],
                    "id" : data["unique_id"]
                })
    elif "aime25" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/aime25/test.jsonl"
        with open(dataset_path,"r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    "problem" : data["problem"],
                    "answer" : data["answer"],
                    "id" : data["id"]
                })
    elif "aime24" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/aime24"
        dataset_data = ld(dataset_path, split="train")
        dataset = []
        for data in dataset_data:
            dataset.append({
                "problem" : data["problem"],
                "answer" : data["answer"],
                "id" : data["id"]
            })
    elif "gsm8k" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/gsm8k"
        dataset_data = ld(dataset_path, "main",split="test")
        dataset = []
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["answer"].split("#### ")[-1],
                "id" : data["question"]
            })
    elif "olympiad" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/olympiad"
        dataset_data = ld(dataset_path, split="test")
        dataset = []
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["final_answer"][0],
                "id" : data["id"]
            })
    elif "amc23" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/AMC23"
        dataset_data = ld(dataset_path, split="test")
        dataset = []
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["answer"],
                "id" : data["id"]
            })
    elif "minervamath" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/MinervaMath/test.jsonl"
        with open(dataset_path,"r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    "problem" : data["question"],
                    "answer" : data["answer"],
                    "id" : data["question"]
                })
    elif "gsm8k" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/DeepMath-103K"
        dataset = []
        dataset_data = ld(dataset_path,"main")
        for data in dataset_data["test"]:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["answer"].split("####")[-1],
                "id" : data["question"],
            })
    elif "deepmath" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/DeepMath-103K"
        dataset = []
        dataset_data = ld(dataset_path, split="train")
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["final_answer"],
                "id" : data["question"],
            })
    elif "supergpqa" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/SuperGPQA"
        dataset = []
        dataset_data = ld(dataset_path, split="train")
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["answer_letter"],
                "options" : data["options"],
                "id" : data["question"],
            })
    elif "mmlupro" in dataset_name.lower():
        dataset_path = " hf_dir/hf_dataset/MMLU-Pro/data"
        dataset = []
        dataset_data = ld(dataset_path, split="test")
        for data in dataset_data:
            dataset.append({
                "problem" : data["question"],
                "answer" : data["answer"],
                "options" : data["options"],
                "id" : data["question"],
            })
    
    return dataset