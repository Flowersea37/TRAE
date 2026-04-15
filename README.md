# Escaping the Echo Trap: On Credit Assignment Failure in Multi-turn LLM Self-Reflection

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>


<!-- <strong>Search-R1</strong> is a reinforcement learning framework for <em>training reasoning and searching (tool-call) interleaved LLMs</em>.  -->
<!-- We built upon [veRL](https://github.com/volcengine/verl). -->
**Search-R1** is a reinforcement learning framework designed for training **reasoning-and-searching interleaved LLMs**—language models that learn to reason and make tool calls (e.g., to search engines) in a coordinated manner.

<!-- It can be seen as an extension of <strong>DeepSeek-R1(-Zero)</strong> with interleaved search engine calling and an opensource RL training-based solution for <strong>OpenAI DeepResearch</strong>. -->
Built upon [veRL](https://github.com/volcengine/verl), Search-R1 extends the ideas of **DeepSeek-R1(-Zero)** by incorporating interleaved search engine access and provides a fully open-source RL training pipeline. It serves as an alternative and open solution to **OpenAI DeepResearch**, enabling research and development in tool-augmented LLM reasoning.

<!-- Through RL (rule-based outcome reward), the 3B **base** LLM (both Qwen2.5-3b-base and Llama3.2-3b-base) develops reasoning and search engine calling abilities all on its own. -->

We support different RL methods (e.g., PPO, GRPO, reinforce), different LLMs (e.g., llama3, Qwen2.5, etc) and different search engines (e.g., local sparse/dense retrievers and online search engines).

Paper: [link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117); Model and data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full experiment log: [prelim](https://wandb.ai/peterjin/Search-R1-open); [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2); [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3). Details about these logs and methods can be find [here](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md).


!example.png

## Links

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
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```