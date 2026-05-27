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

_CHOICES = {"A", "B", "C", "D"}
_TAIL_CHARS = 500


def _last_boxed_content(text: str) -> str | None:
    """Return the content of the last LaTeX \\boxed answer."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    pos = idx + len("\\boxed")
    while pos < len(text) and text[pos].isspace():
        pos += 1

    if pos >= len(text):
        return None

    if text[pos] != "{":
        tail = text[pos:].strip()
        return tail.split()[0] if tail else None

    depth = 0
    start = pos + 1
    for i in range(pos, len(text)):
        if text[i] == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def _strip_latex_wrappers(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:text|textrm|mathrm|mathbf|mathsf)\s*\{([^{}]*)\}", r"\1", text)
    return text


def normalize_choice(text: object) -> str | None:
    """Normalize a free-form multiple-choice answer to A/B/C/D."""
    if text is None:
        return None

    candidate = _strip_latex_wrappers(str(text)).strip()
    candidate = candidate.replace("$", " ")
    candidate = candidate.replace("{", " ").replace("}", " ")
    candidate = re.sub(r"\\[a-zA-Z]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-D])\s*[\)\]]?",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\]]?\s*$",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\]]?\s*[\.:,;-]",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\]]?\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, candidate, flags=re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice in _CHOICES:
                return choice

    matches = re.findall(r"\b([A-D])\b", candidate, flags=re.IGNORECASE)
    if matches:
        choice = matches[-1].upper()
        return choice if choice in _CHOICES else None

    return None


def extract_choice(solution_str: str) -> str | None:
    text = "" if solution_str is None else str(solution_str)

    boxed_content = _last_boxed_content(text)
    pred = normalize_choice(boxed_content)
    if pred is not None:
        return pred

    return normalize_choice(text[-_TAIL_CHARS:])


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict[str, object]:
    pred = extract_choice(solution_str)
    target = normalize_choice(ground_truth)
    correct = pred is not None and target is not None and pred == target
    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": pred or "",
    }
