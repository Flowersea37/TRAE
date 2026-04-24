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

import random
import unittest

import numpy as np
import pytest
import torch

from trae_verl.trainer.ppo import core_algos_tree_dynamic_advantage as tree_core_algos
import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def _legacy_cal_part(tree_reward, begin, end, norm_adv_by_std_in_grpo, epsilon):
    level_arr = torch.tensor(tree_reward[begin:end], dtype=torch.float32)
    for i in range(begin, end):
        if norm_adv_by_std_in_grpo:
            tree_reward[i] = (tree_reward[i] - torch.mean(level_arr)) / (torch.std(level_arr) + epsilon)
        else:
            tree_reward[i] = tree_reward[i] - torch.mean(level_arr)


def _legacy_cal_adv_based_step_reward(indexs, step_rewards, traj_end_indexs, norm_adv_by_std_in_grpo, epsilon):
    id2tree_reward = {}
    for i in range(len(indexs)):
        index = indexs[i]
        step_reward = step_rewards[i]
        traj_end_index = traj_end_indexs[i]

        if index not in id2tree_reward:
            id2tree_reward[index] = [-100 for _ in range(16)]

        step_index = -1
        while traj_end_index > 1:
            if id2tree_reward[index][traj_end_index] != -100:
                assert id2tree_reward[index][traj_end_index] == step_reward[step_index]
            else:
                id2tree_reward[index][traj_end_index] = step_reward[step_index]
            traj_end_index = traj_end_index // 2
            step_index -= 1

    for _, tree_reward in id2tree_reward.items():
        for i in range(len(tree_reward) - 1, -1, -1):
            if i < 8:
                return_future = (tree_reward[2 * i] + tree_reward[2 * i + 1]) / 2
                tree_reward[i] = tree_reward[i] + 0.5 * return_future

    for _, tree_reward in id2tree_reward.items():
        _legacy_cal_part(tree_reward, 2, 4, norm_adv_by_std_in_grpo, epsilon)
        _legacy_cal_part(tree_reward, 4, 8, norm_adv_by_std_in_grpo, epsilon)
        _legacy_cal_part(tree_reward, 8, 16, norm_adv_by_std_in_grpo, epsilon)

    return id2tree_reward


def _legacy_compute_grpo_tree_dynamic_advantage(
    token_level_rewards,
    response_mask,
    index,
    step_reward,
    traj_end_index,
    epsilon=1e-6,
    norm_adv_by_std_in_grpo=True,
):
    id2tree_adv = _legacy_cal_adv_based_step_reward(
        indexs=index,
        step_rewards=step_reward.tolist(),
        traj_end_indexs=traj_end_index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        epsilon=epsilon,
    )

    batch_size, seq_len = response_mask.shape
    return_scores = torch.zeros(batch_size, seq_len, 1, device=token_level_rewards.device)
    final_score = torch.tensor([0.0], dtype=torch.float32, device=token_level_rewards.device)
    for i in range(batch_size):
        tree_adv = id2tree_adv[index[i]]
        traj_end_index_i = traj_end_index[i]
        current_custom_advs = []
        for _ in range(3):
            current_custom_advs = [tree_adv[traj_end_index_i]] + current_custom_advs
            traj_end_index_i = traj_end_index_i // 2
        current_idx = 0
        state = "mask"

        for j in range(seq_len):
            if response_mask[i][j].eq(0).item():
                if state == "no_mask":
                    current_idx += 1
                state = "mask"
            else:
                assert current_idx < len(current_custom_advs)
                state = "no_mask"
                final_score = torch.tensor([current_custom_advs[current_idx]], dtype=torch.float32, device=token_level_rewards.device)
            return_scores[i][j] = final_score
    return_scores = return_scores.squeeze(-1)
    return return_scores, return_scores


def _make_binary_tree_step_rewards():
    node_reward = {
        2: 1.0,
        3: 2.0,
        4: 0.5,
        5: 1.5,
        6: 2.5,
        7: 3.5,
        8: -1.0,
        9: 0.0,
        10: 1.0,
        11: 2.0,
        12: -2.0,
        13: -1.0,
        14: 0.5,
        15: 1.5,
    }
    leaf_indices = np.arange(8, 16, dtype=np.int64)
    step_rewards = []
    for leaf_idx in leaf_indices:
        path = []
        current_idx = int(leaf_idx)
        while current_idx > 1:
            path.append(node_reward[current_idx])
            current_idx //= 2
        step_rewards.append(path[::-1])
    return leaf_indices, torch.tensor(step_rewards, dtype=torch.float32)


def test_tree_dynamic_advantage_matches_legacy_binary_depth3():
    leaf_indices, step_reward = _make_binary_tree_step_rewards()
    batch_size = len(leaf_indices)
    response_mask = torch.tensor([[1, 1, 0, 1, 0, 1]] * batch_size, dtype=torch.float32)
    token_level_rewards = torch.ones_like(response_mask)
    index = np.array(["tree_a"] * batch_size, dtype=object)
    branch_per_node = np.array([2] * batch_size, dtype=object)
    max_depth = np.array([3] * batch_size, dtype=object)

    legacy_adv, legacy_ret = _legacy_compute_grpo_tree_dynamic_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        step_reward=step_reward,
        traj_end_index=leaf_indices,
    )
    new_adv, new_ret = tree_core_algos.compute_grpo_tree_dynamic_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        step_reward=step_reward,
        traj_end_index=leaf_indices,
        branch_per_node=branch_per_node,
        max_depth=max_depth,
    )

    assert torch.allclose(new_adv, legacy_adv, rtol=1e-5, atol=1e-6)
    assert torch.allclose(new_ret, legacy_ret, rtol=1e-5, atol=1e-6)


def test_tree_dynamic_advantage_generalizes_to_branch3_depth4():
    branch_per_node = 3
    max_depth = 4
    leaf_start = branch_per_node**max_depth
    leaf_end = 2 * (branch_per_node**max_depth)
    leaf_indices = np.arange(leaf_start, leaf_end, dtype=np.int64)

    step_rewards = []
    for leaf_idx in leaf_indices:
        path = tree_core_algos._collect_path_to_root(int(leaf_idx), branch_per_node)
        step_rewards.append([float(node_idx % 7 - 3) for node_idx in path])
    step_reward_tensor = torch.tensor(step_rewards, dtype=torch.float32)

    tree_adv = tree_core_algos.cal_adv_based_step_reward(
        indexs=np.array(["tree_b"] * len(leaf_indices), dtype=object),
        step_rewards=step_reward_tensor.tolist(),
        traj_end_indexs=leaf_indices,
        branch_per_node=branch_per_node,
        max_depth=max_depth,
        norm_adv_by_std_in_grpo=True,
        epsilon=1e-6,
    )["tree_b"]

    for depth in range(1, max_depth + 1):
        level_start = branch_per_node**depth
        level_end = 2 * (branch_per_node**depth)
        level_values = torch.tensor(
            [tree_adv[node_idx] for node_idx in range(level_start, level_end)],
            dtype=torch.float32,
        )
        assert torch.isclose(level_values.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(level_values.std(), torch.tensor(1.0), atol=1e-4)

    response_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 1]] * len(leaf_indices), dtype=torch.float32)
    token_level_rewards = torch.ones_like(response_mask)
    advantages, returns = tree_core_algos.compute_grpo_tree_dynamic_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["tree_b"] * len(leaf_indices), dtype=object),
        step_reward=step_reward_tensor,
        traj_end_index=leaf_indices,
        branch_per_node=np.array([branch_per_node] * len(leaf_indices), dtype=object),
        max_depth=np.array([max_depth] * len(leaf_indices), dtype=object),
    )

    assert advantages.shape == returns.shape == response_mask.shape
    sample_path = tree_core_algos._collect_path_to_root(int(leaf_indices[0]), branch_per_node)
    sample_values = [tree_adv[node_idx] for node_idx in sample_path]
    sample_row = advantages[0].tolist()
    assert sample_row[0] == pytest.approx(sample_values[0], abs=1e-6)
    assert sample_row[2] == pytest.approx(sample_values[1], abs=1e-6)
    assert sample_row[4] == pytest.approx(sample_values[2], abs=1e-6)
    assert sample_row[6] == pytest.approx(sample_values[3], abs=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
