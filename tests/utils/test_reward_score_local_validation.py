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

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.gpqa import extract_choice


def test_math500_local_data_source_uses_boxed_math_reward():
    assert default_compute_score("MATH-500", r"The answer is \boxed{p-q}.", "p - q") == {
        "score": 1.0,
        "acc": True,
        "pred": "p-q",
    }


def test_aime_uppercase_data_source_uses_boxed_math_reward():
    assert default_compute_score("AIME", r"The answer is \boxed{33}.", "33") == {
        "score": 1.0,
        "acc": True,
        "pred": "33",
    }


def test_gpqa_boxed_choice_score():
    assert default_compute_score("GPQA", r"The answer is \boxed{\text{A}}.", "A") == {
        "score": 1.0,
        "acc": True,
        "pred": "A",
    }


def test_gpqa_boxed_choice_with_option_text():
    assert extract_choice(r"After solving it, the final answer is \boxed{B. 315}.") == "B"


def test_gpqa_answer_sentence_wrong_choice():
    result = default_compute_score("gpqa_diamond", "The answer is (C).", "A")
    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["pred"] == "C"


def test_boxed_math_wrong_answer_keeps_pred():
    result = default_compute_score("AIME", r"The answer is \boxed{34}.", "33")
    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["pred"] == "34"
