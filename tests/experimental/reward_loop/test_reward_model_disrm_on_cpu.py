# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.reward_loop.reward_loop import RewardLoopWorker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_name", "response", "expected_endpoint", "expected_payload", "expected_score"),
    [
        (
            "vllm",
            {"data": [{"probs": [0.125, 0.875]}]},
            "classify",
            {"model": "test-model", "input": "reward prompt", "use_activation": False},
            0.875,
        ),
        (
            "sglang",
            {"data": [{"embedding": [0.25, 0.75]}]},
            "v1/embeddings",
            {"model": "test-model", "input": "reward prompt"},
            0.75,
        ),
    ],
)
async def test_compute_score_disrm_returns_reward_contract(
    engine_name, response, expected_endpoint, expected_payload, expected_score
):
    worker = object.__new__(RewardLoopWorker)
    worker.config = SimpleNamespace(
        reward=SimpleNamespace(
            reward_model=SimpleNamespace(
                rollout=SimpleNamespace(name=engine_name),
                model_path="test-model",
            )
        )
    )
    worker._preprocess_reward_inputs = AsyncMock(return_value="reward prompt")
    worker._post_request = AsyncMock(return_value=response)

    result = await worker.compute_score_disrm(data=None)

    assert result == {"reward_score": expected_score, "reward_extra_info": {}}
    worker._post_request.assert_awaited_once_with(expected_payload, expected_endpoint)
