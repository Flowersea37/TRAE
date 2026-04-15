import argparse
import logging
import os
from vllm import LLM
from random import seed
from typing import List, Dict
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from config import ModelConfig, DatasetType
from dataset_load_utils import load_dataset
from model_multi_adv import PolicyModel
import json
import re
import ray
import random
import gc
import torch
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class Pipeline:
    def __init__(self, policy_config: ModelConfig, reward_config: ModelConfig):
        self.policy_config = policy_config
        self.reward_config = reward_config
        self.policy = PolicyModel(self.policy_config)

    def extract_boxed_all(self, s: str):
        tag = '\\boxed{'
        out = []
        i = 0
        while True:
            start = s.find(tag, i)
            if start == -1:
                break
            j = start + len(tag)
            depth = 1
            while j < len(s) and depth > 0:
                c = s[j]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(s[start + len(tag): j - 1])
                i = j
            else:
                break
        return out

    def compare_answer(self, predict_answer, ground_answer):
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

        if type(ground_answer) != str:
            ground_answer = str(ground_answer)
        ground_truth_boxed = "\\boxed{" + ground_answer + "}"
        predict_answer_boxed = "\\boxed{" + predict_answer + "}"

        try:
            judge, _ = verify_func([ground_truth_boxed], [predict_answer_boxed])
            logger.info(judge)
            if judge == 1.0:
                judge = True
            else:
                judge = False
        except Exception:
            judge = False

        return judge

    def judge_answers(self, data, time_step: int):
        reasonings = data[f"reasoning_{time_step}"]
        results = []

        for reasoning in reasonings:
            answers = self.extract_boxed_all(reasoning)
            if len(answers) == 0:
                answers = [reasoning]
            answer = answers[-1]
            judge = self.compare_answer(answer, data["answer"])
            results.append(judge)

        return results

    def evaluate(self, dataset, times: int, mode: str):
        if len(dataset) == 0:
            return []

        if mode == "response":
            results = self.policy.generate_answers(dataset)
        else:
            results = dataset

        # 反思多轮 reasoning
        for i in range(times):
            if f"reasoning_{i + 1}" in results[0].keys():
                continue
            results = self.policy.reflect_answers(results, i)

        # 打 judge
        for data in results:
            begin = 0
            if mode != "response":
                begin += 1
            for i in range(begin, times + 1):
                judges = self.judge_answers(data, i)
                data[f"judges_{i}"] = judges

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on various mathematical datasets"
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        help="Policy model path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        help="reward model path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        help="the name of base mode",
        default="Qwen3-8B-Base",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode type: response | reflect",
        default="response",
    )
    parser.add_argument(
        "--response_path",
        type=str,
        help="response path (for reflect mode)",
        default="",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DeepMath",
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=0.6,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--generation_top_p",
        type=float,
        default=0.95,
        help="Top-p for generation",
    )
    parser.add_argument(
        "--generation_out_seq_length",
        type=int,
        default=4096,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--num_sequence",
        type=int,
        default=1,
        help="Number of sequence",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help="Number of reflection times",
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=8,
        help="Number of Ray actors (GPUs) to use",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=10,
        help="Number of samples per mini batch (one Ray task)",
    )

    args = parser.parse_args()

    policy_config = ModelConfig(
        model_name=args.policy_model,
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        out_seq_length=args.generation_out_seq_length,
        tensor_parallel_size=args.tensor_parallel_size,
        num_sequence=args.num_sequence,
    )

    reward_config = ModelConfig(
        model_name=args.reward_model,
    )

    model_name = args.policy_model.split("/")[-1]
    reward_name = args.reward_model.split("/")[-1]

    if args.mode == "response":
        dataset_name = args.dataset
        output_dir = (
            f"outputs/eval-{args.times}-"
            f"{model_name}-"
            f"max-token{policy_config.out_seq_length}/"
            f"{dataset_name}"
        )
    else:
        # mode == reflect
        dataset_name = args.response_path.split("/")[-2]
        output_dir = (
            f"outputs/eval-{args.times}-"
            f"{model_name}-"
            f"max-token{policy_config.out_seq_length}/"
            f"{dataset_name}"
        )

    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)

    eval_path = os.path.join(output_dir, "eval.json")
    exist_instance = set()

    # 读取已经评估过的 id，避免重复
    if os.path.exists(eval_path):
        logger.info(f"Dataset {dataset_name} already partially evaluated, loading existing eval.json.")
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        exist_instance.add(item["id"])
                    except Exception:
                        continue
    else:
        # 创建空文件
        with open(eval_path, "w", encoding="utf-8") as f:
            pass

    logger.info(f"Processing dataset: {dataset_name}")
    if args.mode == "response":
        dataset_data = load_dataset(args.dataset)
    else:
        dataset_data = []
        with open(args.response_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset_data.append(json.loads(line))

    logger.info(f"Total dataset size: {len(dataset_data)}")

    # 过滤掉已经评估过的样本
    tmp_dataset = []
    for data in dataset_data:
        if data["id"] not in exist_instance:
            tmp_dataset.append(data)
    dataset_data = tmp_dataset

    logger.info(f"Remaining samples to evaluate: {len(dataset_data)}")

    if len(dataset_data) == 0:
        logger.info("No new samples to evaluate. Exit.")
        return

    # 初始化 Ray
    ray.init()
    # num_actors = args.num_actors
    num_actors = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # import pdb;pdb.set_trace()
    actors = [Pipeline.remote(policy_config, reward_config) for _ in range(num_actors)]
    logger.info(f"Created {num_actors} Pipeline actors.")

    # ---- 关键改动：把整个数据集切成很多 mini-batch，然后全部作为任务提交 ----
    mini_batch_size = max(1, args.mini_batch_size)
    mini_batches = []
    current_batch = []

    for data in dataset_data:
        current_batch.append(data)
        if len(current_batch) >= mini_batch_size:
            mini_batches.append(current_batch)
            current_batch = []
    if len(current_batch) > 0:
        mini_batches.append(current_batch)

    logger.info(
        f"Total mini batches: {len(mini_batches)}, "
        f"mini_batch_size={mini_batch_size}"
    )

    # 提交所有 mini-batch 任务，采用轮询方式分配给不同 actor
    futures = []
    # import pdb;pdb.set_trace()
    # if torch.cuda.is_available():
    #     torch_seed = torch.initial_seed()
        # print(f"当前torch随机种子: {torch_seed}")

    for idx, mb in enumerate(mini_batches):
        actor = actors[idx % num_actors]
        fut = actor.evaluate.remote(mb, args.times, args.mode)
        futures.append(fut)

    # 使用 ray.wait 流式获取结果：哪个 GPU 先算完就先写到 eval.json
    pending = set(futures)
    finished_cnt = 0


    while pending:
        done, pending = ray.wait(list(pending), num_returns=1)
        result_batch = ray.get(done[0])
        finished_cnt += 1

        # 写结果到 eval.json（追加）
        with open(eval_path, "a", encoding="utf-8") as f:
            for item in result_batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if finished_cnt % 10 == 0 or finished_cnt == len(futures):
            logger.info(
                f"Finished {finished_cnt}/{len(futures)} mini batches "
                f"({finished_cnt * mini_batch_size} samples approx.)"
            )

        # 稍微 sleep 一下，避免疯狂刷日志
        time.sleep(0.1)

    logger.info("All mini batches finished.")
    ray.shutdown()


if __name__ == "__main__":
    main()
