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

from . import math_reward


def extract_boxed_answer(solution_str: str) -> str:
    text = "" if solution_str is None else str(solution_str)
    try:
        boxed_answer = math_reward.last_boxed_only_string(text)
        if boxed_answer is None:
            return ""
        pred = math_reward.remove_boxed(boxed_answer)
    except Exception:
        return ""

    try:
        return math_reward.strip_string(pred)
    except Exception:
        return str(pred).strip()


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict[str, object]:
    score = float(math_reward.compute_score("" if solution_str is None else solution_str, ground_truth))
    return {
        "score": score,
        "acc": score > 0.0,
        "pred": extract_boxed_answer(solution_str),
    }
