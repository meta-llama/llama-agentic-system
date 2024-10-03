# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import json
import random
import re

import blobfile as bf
import fire
import pandas
from llama_stack_client import LlamaStackClient

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question and make the answer very simple. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}\s*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def get_dataset():
    # this will map to the Dataset type

    # url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    df = pandas.read_csv(bf.BlobFile("./evals/mmlu.csv", "r"))
    examples = [row.to_dict() for _, row in df.iterrows()]
    return examples


def preprosess(row):
    # this will map to eval/preprocess(dataset)

    content = QUERY_TEMPLATE_MULTICHOICE.format(**row)
    return {"role": "user", "content": content}


def inference(client, preprocessed):
    # this will map to eval/inference(preprocessed)
    response = client.inference.chat_completion(
        messages=[preprocessed],
        model="Llama3.1-8B-Instruct",
        stream=False,
    )
    return response


def postprocess(response):
    # this will map to eval/postprocess
    if response.startswith("data:"):
        response = response[len("data: ") :]
        json_response = json.loads(response)

    return normalize_response(json_response["completion_message"]["content"])


def score_fn(response_text, row):
    # F2: transform generation outputs
    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break

    score = 1.0 if extracted_answer and extracted_answer == row["Answer"] else 0.0
    return score


def main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"https://llama-stack.together.ai",
    )

    # Get dataset --> mapping to dataset/register
    dataset = get_dataset()[:3]
    print(len(dataset))
    print(dataset[0])
    print(dataset[1].keys())

    # F1: preprocess dataset
    preprocessed = [preprosess(row) for row in dataset]
    print(preprocessed[0])

    # Inference loop to get generation outputs --> Batch inference
    generation_outputs = [inference(client, row) for row in preprocessed]

    # F2: post process
    postprocessed = [postprocess(row) for row in generation_outputs]
    print(postprocessed[0])

    # F3: evaluate generation outputs (based on dataset & scoring function)
    scores = [
        score_fn(response_text, row)
        for response_text, row in zip(generation_outputs, dataset)
    ]

    print(scores)
    # return scores


if __name__ == "__main__":
    fire.Fire(main)
