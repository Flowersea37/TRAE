# Escaping the Echo Trap: On Credit Assignment Failure in Multi-turn LLM Self-Reflection


## Overview
## Results

## Links

- [Escaping the Echo Trap: On Credit Assignment Failure in Multi-turn LLM Self-Reflection](#escaping-the-echo-trap-on-credit-assignment-failure-in-multi-turn-llm-self-reflection)
  - [Overview](#overview)
  - [Results](#results)
  - [Links](#links)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Acknowledgement](#acknowledgement)
  - [Citations](#citations)

## Installation
```bash
### Create a new environment with python3.12
conda create -n trae python=3.12
conda activate trae

### Install Verl_0.7.0.dev0
cd verl
bash scripts/verl_install.sh
```

## Training

Train a multi-turn reflective LLM on our dataset using verl, based on Qwen3-8B-Base.

```bash
## The configuration is consistent with that used in the paper.
bash scripts/train/run_qwen3-8b-base_trae.sh
```

## Evaluation
(1) Prepare evaluation data

For each question-answer sample, it should be a dictionary containing the desired content as below:
```
dataset.append({
    "problem" : data["problem"],
    "answer" : data["answer"],
    "id" : data["id"]
})
```
Plase download the evaluation data by your own, and refer to the data prepocess code in ```evaluation/dataset_load_utils.py```

(2) Run Evaluation.
```bash
# 导出训练后模型
bash scripts/verl_merge_fsdb_2.sh

# eval
bash scripts/eval.sh

# modify the raw_path in analysis_tree.py, to see the evaluation result
python evaluation/analysis_tree.py
```
## Acknowledgement

The codebase is built upon [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [veRL](https://github.com/volcengine/verl).We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

## Citations
```bibtex

```