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

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int) -> float:
    """Compute soft overlong punishment in [-1, 0]."""
    if max_response_length <= 0:
        return 0.0

    if overlong_buffer_length <= 0:
        return 0.0 if response_length <= max_response_length else -1.0

    expected_len = max_response_length - overlong_buffer_length
    if expected_len < 0:
        expected_len = 0

    if response_length <= expected_len:
        return 0.0
    if response_length <= max_response_length:
        return (expected_len - response_length) / float(overlong_buffer_length)
    return -1.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    max_response_length: Optional[int] = None,
    overlong_buffer_length: int = 0,
    overlong_penalty_factor: float = 0.0,
) -> list[dict[str, float]]:
    """Score math responses using answer accuracy and format compliance."""
    scores: list[dict[str, float]] = []

    use_overlong = max_response_length is not None and float(overlong_penalty_factor) != 0.0
    max_len = int(max_response_length) if max_response_length is not None else 0
    buffer_len = int(overlong_buffer_length)
    penalty_factor = float(overlong_penalty_factor)

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = float(format_reward(response))
        accuracy_score = float(accuracy_reward(response, reward_input["ground_truth"]))

        base_overall = (1.0 - float(format_weight)) * accuracy_score + float(format_weight) * format_score

        overlong_score = 0.0
        if use_overlong:
            resp_len = int(reward_input.get("response_length", 0))
            overlong_score = float(soft_overlong_punishment(resp_len, max_len, buffer_len))

        overall = base_overall + penalty_factor * overlong_score

        scores.append(
            {
                "overall": float(overall),
                "overall_no_overlong": float(base_overall),
                "format": float(format_score),
                "accuracy": float(accuracy_score),
                "overlong": float(overlong_score),
            }
        )

    return scores
