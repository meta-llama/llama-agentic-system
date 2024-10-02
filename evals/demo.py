# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import random
import re

import blobfile as bf
import fire
import pandas
from llama_stack_client import LlamaStackClient

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def get_dataset():
    # url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    df = pandas.read_csv(bf.BlobFile("./evals/mmlu.csv", "r"))
    examples = [row.to_dict() for _, row in df.iterrows()]
    return examples


def preprosess(row):
    content = QUERY_TEMPLATE_MULTICHOICE.format(**row)
    return {"role": "user", "content": content}


def inference(client, row):
    response = client.inference.chat_completion(
        messages=[row],
        model="Llama3.1-8B-Instruct",
        stream=True,
    )
    print(response)
    return response


def main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    # Get dataset
    dataset = get_dataset()[:10]
    print(len(dataset))
    print(dataset[0])
    print(dataset[1].keys())

    # F1: preprocess dataset
    preprocessed = [preprosess(row) for row in dataset]
    print(preprocessed[0])

    # Inference loop to get generation outputs --> Batch inference
    a = inference(client, preprocessed[0])
    print(a)
    # generation_outputs = [inference(client, row) for row in preprocessed]

    # F2: transform generation outputs
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break

    # F3: evaluate generation outputs (based on scoring function)


if __name__ == "__main__":
    fire.Fire(main)
