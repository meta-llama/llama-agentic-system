# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from termcolor import cprint

from llama_stack_client.types import *  # noqa: F403
from llama_stack_client.types.agent_create_params import *  # noqa: F403
from common.client_utils import *  # noqa: F403

from .multi_turn import execute_turns, prompt_to_turn


async def run_main(host: str, port: int, disable_safety: bool = False):
    api_keys = load_api_keys_from_env()
    agent_config = await make_agent_config_with_custom_tools(
        model="Llama3.1-8B-Instruct",
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            # Enable builtin tools
            tool_definitions=[
                # web search ( search the web for current info)
                search_tool_defn(api_keys),
                # code interpreter (allowing for loading attachements and reading them)
                AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter"),
            ],
            custom_tools=[],
            # Enables file ananlysis mode ( instead of RAG mode )
            attachment_behavior="code_interpreter",
        ),
    )

    transcript_path = "https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/transcript_shorter.txt"

    await execute_turns(
        agent_config=agent_config,
        custom_tools=[],
        turn_inputs=[
            prompt_to_turn(
                "here is a podcast transcript, can you summarize it",
                attachments=[
                    Attachment(
                        content=transcript_path,
                        mime_type="text/plain",
                    ),
                ],
            ),
            prompt_to_turn("What are the top 3 salient topics that were discussed ?"),
            prompt_to_turn("Was anything related to 'H100' discussed ?"),
            prompt_to_turn(
                "While this podcast happened in April, 2024 can you provide an update from the web on what were the key developments that have happened in the last 3 months since then ?"
            ),
            prompt_to_turn(
                "Imagine these people meet again in 1 year, what might be three good follow ups to discuss ?"
            ),
            prompt_to_turn("Can you rewrite these followups in hindi ?"),
        ],
        host=host,
        port=port,
    )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
