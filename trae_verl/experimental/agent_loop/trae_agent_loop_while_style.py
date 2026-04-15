# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import copy
import os
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    # register,
)
from .agent_loop import register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
# 树的全局变量 需要考虑如何实现
metrics = {}
question = ''

class AgentState(Enum):
    PENDING = "pending"
    REFLECTING = "reflecting"
    GENERATING = "generating"
    TERMINATED = "terminated"

class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        question: str,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        # self.tools_kwargs = tools_kwargs
        # self.interaction = interaction
        # self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.question = question

        # 存放所有的response
        self.all_responses: list[str] = []


class TreeNode:
    def __init__(
        self,
        agent_data : AgentData,
        index,
        depth,
        sampling_params
    ):
        self.agent_data = agent_data
        # self.prompt_ids = prompt_ids if prompt_ids is not None else []
        # self.reflected_times = reflected_times  # 使用参数传入的值
        # self.response_mask = None  # 添加默认
        # self.response_ids = []
        # self.response_logprobs: list[float] = []  # 这个变量是什么含义？

        self.left_child = None
        self.right_child = None
        self.index = index
        self.depth = depth
        self.sampling_params = sampling_params


@register("trae_reflect_agent_while_style")
class ReflectAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # Initialize interactions from config file
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )
        self.max_reflect_turns = 3
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing turn-level TRAE-ReflectAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls


        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

        # 定义最大的反思轮次，后面可以定义在配置文件中
        cls.max_reflect_turns = 3
    
    async def generate_response_and_update(self,Tree_node:TreeNode):
        agent_data = Tree_node.agent_data
        with simple_timer("generate_sequences", agent_data.metrics):
            # 因为generate和reflect分开了 所以这里要进行区分
            if Tree_node.agent_data.assistant_turns > 0: # 到了反思
                # 添加反思的message
                reflect_prompt = """The solution you need to reflect is in the previous step and enclosed in <response></response>. Your task is to generation a critique to the solution, which should contain the analysis to the solution, the judge to the solution and some instruction about how to correct the solution if it is wrong or make it better if it is correct and then generate a new solution refined from the previous solution, and referred from the critique.
The new solution should solve the problem step by step. The final answer must be enclosed in \\boxed{{}}.
You must output in the following format:
<critique>your critique to the solution</critique>
<response>the new solution refined from the initial solution</response>
        """
                add_messages = [{
                    "role" : "user",
                    "content" : reflect_prompt
                }]


                reflect_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        add_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                    ),
                )

                Tree_node.agent_data.response_mask += [0] * len(reflect_prompt_ids)
                Tree_node.agent_data.prompt_ids += reflect_prompt_ids
                Tree_node.agent_data.user_turns += 1

            output = await self.server_manager.generate(
                request_id=Tree_node.agent_data.request_id,
                prompt_ids=Tree_node.agent_data.prompt_ids,
                sampling_params=Tree_node.sampling_params,
            )
            
            Tree_node.agent_data.assistant_turns += 1
            Tree_node.agent_data.response_ids = output.token_ids
            Tree_node.agent_data.prompt_ids += Tree_node.agent_data.response_ids #每一次的输出都要拼在prompt_ids的后面作为提示词
            Tree_node.agent_data.response_mask += [1] * len(Tree_node.agent_data.response_ids)
            if output.log_probs:
                Tree_node.agent_data.response_logprobs += output.log_probs

            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(Tree_node.agent_data.response_ids, skip_special_tokens=True)
            )
            # breakpoint()
            reflected_response = self.format_response(assistant_message)
            Tree_node.agent_data.all_responses.append(reflected_response)


    async def get_loop_output(self,agent_data,traj_index):
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:self.prompt_length],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        while(len(agent_data.all_responses)<(self.max_reflect_turns)):
            agent_data.all_responses.append("")
        
        output.extra_fields.update({"all_responses": agent_data.all_responses,"traj_end_index": traj_index,"depth":agent_data.assistant_turns})

        return output

    async def build_sub_tree(self, Root: TreeNode, branch_per_node: int, max_depth: int):
        expand_pool = []
        current_depth = 1
        if max_depth == 0:
            return []
        else:
            expand_pool.append(Root)
            while True:
                temp = []
                for tree_node in expand_pool:
                    if tree_node.depth < max_depth:
                        # 开始扩展
                        for index in range(branch_per_node):
                            agent_data_parent = copy.deepcopy(tree_node.agent_data)
                            # [TODO] 这个地方的index到时候要修改少了一个加1
                            Node_new = TreeNode(agent_data_parent, index=branch_per_node * tree_node.index + index, depth = tree_node.depth + 1, sampling_params=tree_node.sampling_params)
                            await self.generate_response_and_update(Node_new) 
                            temp.append(Node_new)
                if len(temp) <= 0:
                    break
                expand_pool = copy.deepcopy(temp)
        return_list = []
        # breakpoint()
        for Tree_node in expand_pool:
            output = await self.get_loop_output(Tree_node.agent_data, Tree_node.index)
            return_list.append(output)
        return return_list

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        global question,metrics
        print('TRAE Agent Loop运行！！！！！')
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        interaction = None
        interaction_kwargs = {}
        initial_agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            question=kwargs["extra_info"]["question"]
        )
        initial_agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                initial_agent_data.messages,
                add_generation_prompt=True,
                tokenize=True,
                # **self.apply_chat_template_kwargs,
            ),
        )
        root = TreeNode(initial_agent_data,1,depth=0,sampling_params=sampling_params)
        # 将这两个函数合二为一，一个函数直接返回所有的叶子节点
        # await self.build_sub_tree(root)
        # outputs = await self.collect_trajs(root)
        outputs = await self.build_sub_tree(root,branch_per_node=kwargs['branch_per_node'],max_depth=kwargs['max_depth'])
        # breakpoint()
        return outputs

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

    @classmethod
    def format_response(self, model_output: str):
    
        result = re.findall(r"<critique>(.*?)</critique>", model_output, re.S)

        if len(result) == 0 or len(result)>1:
            return ""
        
        result = re.findall(r"<response>(.*?)</response>", model_output, re.S)

        if len(result) == 0 or len(result)>1:
            return ""
        
        solution = result[0]

        return solution
