# llama-agentic-system

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-agentic-system)](https://pypi.org/project/llama-agentic-system/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

This repo allows you to run Llama 3.1 as a system capable of performing "agentic" tasks like:

- Breaking a task down and performing multi-step reasoning.
- Ability to use tools
  - built-in: the model has built-in knowledge of tools like search or code interpreter
  - zero-shot: the model can learn to call tools using previously unseen, in-context tool definitions

Additionally, we would like to shift safety evaluation from the model level to the overall system level. This allows the underlying model to remain broadly steerable and adaptable to use cases which need varying levels of safety protection.

One of the safety protections is provided by Llama Guard. By default, Llama Guard is used for both input and output filtering. However, the system can be configured to modify this default setting. For example, it is recommended to use Llama Guard for output filtering in situations where refusals to benign prompts are frequently observed, as long as safety requirements are met for your use case.

> [!NOTE]
> The API is still evolving and may change. Feel free to build and experiment, but please don't rely on its stability just yet!


**Getting started with the Llama Stack**
========================================
As noted above, making an agentic app work needs a few components:
- ability to run inference on the underlying Llama series of models
- ability to run safety checks using the Llama-Guard series of models
- ability to execute tools, including a code execution environment, and loop using the model's multi-step reasoning process

The [Llama Stack](https://github.com/meta-llama/llama-toolchain/pull/8) defines and standardizes these components and many others that are needed to make building Generative AI applications smoother. Various implementations of these APIs are then conveniently assembled together via a Llama Stack **Distribution**.

To get started with Distributions, you need to:

- install the `llama-toolchain` python package which provides the core `llama` command line utility.
- (optionally) download llama models using the `llama download` command.
- install and configure a "distribution" using `llama distribution install`
- finally, start up a distribution server serving all the necessary APIs using `llama distribution start`.

Once started, you can then point your agentic app to the URL for this server (e.g. `http://localhost:5000`) and make magic happen.

Let's go through these steps in detail now:


**Create a Conda Environment**
-----------------------------

It is best to install python packages in isolated Conda environments. You can use `virtualenv` also (see section at the end of this document), but we recommend Conda since it has better isolation mechanisms.

```bash
# Create and activate a virtual environment
ENV=agentic_env
conda create -n $ENV python=3.10
cd <path-to-llama-agentic-system-repo>
conda activate $ENV

# Install dependencies
pip install -r requirements.txt
```

At this point, you should have `llama-toolchain` package and the `llama` CLI utility available.

You will also need `bwrap` to run the code executor as a tool as part of the agentic system. This utility might be present already on your system. If not, consult https://github.com/containers/bubblewrap for installation instructions.


**Test Installation**
--------------------

Test the installation by running the following command:
```bash
llama --help
```
This should print the CLI help message.

```bash
usage: llama [-h] {download,distribution,model} ...

Welcome to the LLama cli

options:
  -h, --help            show this help message and exit

subcommands:
  {download,distribution,model}
```

**Download Checkpoints (or use existing models)**
----------------------------------------------

Llama Stack supports the `ollama-inline` distribution which can use your local [`ollama`](https://ollama.com/) server. In that case, please consult ollama documentation for downloading necessary models.

Otherwise, you will need to download required checkpoints from either [Meta](https://llama.meta.com/llama-downloads/) or Huggingface.


#### Downloading from Meta

Download the required checkpoints using the following commands:
```bash
# download the 8B model, this can be run on a single GPU
llama download --source meta --model-id Meta-Llama3.1-8B-Instruct

# you can also get the 70B model, this will require 8 GPUs however
llama download --source meta --model-id Meta-Llama3.1-70B-Instruct

# llama-agents have safety enabled by default. For this, you will need
# safety models -- Llama-Guard and Prompt-Guard
llama download --source meta --model-id Prompt-Guard-86M
llama download --source meta --model-id Llama-Guard-3-8B
```

For all the above, you will need to provide a URL which can be obtained from https://llama.meta.com/llama-downloads/ after signing an agreement.

#### Downloading from Huggingface

```bash
llama download --source huggingface --model-id Meta-Llama3.1-8B-Instruct
```

Essentially, the same commands above work, just replace `--source meta` with `--source huggingface`.

**Important:** Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command to validate your access. You can find your token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

> **Tip:** Default for `llama download` is to run with `--ignore-patterns *.safetensors` since we use the `.pth` files in the `original` folder. For Llama Guard and Prompt Guard, however, we need safetensors. Hence, please run with `--ignore-patterns original` so that safetensors are downloaded and `.pth` files are ignored.


**Installing and Configuring Distributions**
------------------------------------

Let’s start with listing available distributions
```
$ llama distribution list

+---------------+---------------------------------------------+----------------------------------------------------------------------+
| Spec ID       | ProviderSpecs                               | Description                                                          |
+---------------+---------------------------------------------+----------------------------------------------------------------------+
| inline        | {                                           | Use code from `llama_toolchain` itself to serve all llama stack APIs |
|               |   "inference": "meta-reference",            |                                                                      |
|               |   "safety": "meta-reference",               |                                                                      |
|               |   "agentic_system": "meta-reference"        |                                                                      |
|               | }                                           |                                                                      |
+---------------+---------------------------------------------+----------------------------------------------------------------------+
| remote        | {                                           | Point to remote services for all llama stack APIs                    |
|               |   "inference": "inference-remote",          |                                                                      |
|               |   "safety": "safety-remote",                |                                                                      |
|               |   "agentic_system": "agentic_system-remote" |                                                                      |
|               | }                                           |                                                                      |
+---------------+---------------------------------------------+----------------------------------------------------------------------+
| ollama-inline | {                                           | Like local-source, but use ollama for running LLM inference          |
|               |   "inference": "meta-ollama",               |                                                                      |
|               |   "safety": "meta-reference",               |                                                                      |
|               |   "agentic_system": "meta-reference"        |                                                                      |
|               | }                                           |                                                                      |
+---------------+---------------------------------------------+----------------------------------------------------------------------+

```

As you can see above, each “spec” details the “providers” that make up that spec. For eg. The inline uses the “meta-reference” provider for inference while the ollama-inline relies on a different provider ( ollama ) for inference.

At this point, we don't recommend using the `remote` distribution since there are no remote providers supporting the Llama Stack APIs. We hope this changes imminently.

To install a distro, we run a simple command providing 2 inputs –
- **Spec Id** of the distribution that we want to install ( as obtained from the list command )
- A **Name** by which this installation will be known locally.

Let's imagine you are working with a 8B-Instruct model, so we will name our local installation as `inline-llama-8b`. The following command will both install _and_ configure the distribution. As part of the configuration, you will be asked for some inputs (model_id, max_seq_len, etc.)

```
llama distribution install --spec inline --name inline-llama-8b
```

Once it runs successfully , you should see some outputs in the form:

```
$ llama distribution install --spec inline --name inline-llama-8b
....
....
Successfully installed cfgv-3.4.0 distlib-0.3.8 identify-2.6.0 libcst-1.4.0 llama_toolchain-0.0.2 moreorless-0.4.0 nodeenv-1.9.1 pre-commit-3.8.0 stdlibs-2024.5.15 toml-0.10.2 tomlkit-0.13.0 trailrunner-1.4.0 ufmt-2.7.0 usort-1.0.8 virtualenv-20.26.3

Distribution `inline-llama-8b` (with spec inline) has been installed successfully!
```

You can re-configure this distribution by running:
```
llama distribution configure --name inline-llama-8b
```

Here is an example run of how the CLI will guide you to fill the configuration
```
$ llama distribution configure --name inline-llama-8b

Configuring API surface: inference
Enter value for model (required): Meta-Llama3.1-8B-Instruct
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (required): 4096
Enter value for max_batch_size (default: 1): 1
Configuring API surface: safety
Do you want to configure llama_guard_shield? (y/n): y
Entering sub-configuration for llama_guard_shield:
Enter value for model (required): Llama-Guard-3-8B
Enter value for excluded_categories (required): []
Enter value for disable_input_check (default: False):
Enter value for disable_output_check (default: False):
Do you want to configure prompt_guard_shield? (y/n): y
Entering sub-configuration for prompt_guard_shield:
Enter value for model (required): Prompt-Guard-86M
Configuring API surface: agentic_system
YAML configuration has been written to /home/ashwin/.llama/distributions/i0/config.yaml
```

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-8B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`

**Starting a Distribution and Testing it**
----------------------------------------------

Now let’s start the distribution using the CLI.
```
llama distribution start --name inline-llama-8b --port 5000
```
You should see the distribution start and print the APIs that it is supporting,

```
$ llama distribution start --name inline-llama-8b --port 5000

> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 19.28 seconds
NCCL version 2.20.5+cuda12.4
Finished model load YES READY
Serving POST /inference/batch_chat_completion
Serving POST /inference/batch_completion
Serving POST /inference/chat_completion
Serving POST /inference/completion
Serving POST /safety/run_shields
Serving POST /agentic_system/memory_bank/attach
Serving POST /agentic_system/create
Serving POST /agentic_system/session/create
Serving POST /agentic_system/turn/create
Serving POST /agentic_system/delete
Serving POST /agentic_system/session/delete
Serving POST /agentic_system/memory_bank/detach
Serving POST /agentic_system/session/get
Serving POST /agentic_system/step/get
Serving POST /agentic_system/turn/get
Listening on :::5000
INFO:     Started server process [453333]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```


> [!NOTE]
> Configuration is in `~/.llama/distributions/inline-llama-8b/config.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "inline" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

This server is running a Llama model locally.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

Now that the Distribution server is setup, the next thing would be to run an agentic app using AgenticSystem APIs.

We have built sample scripts, notebooks and a UI chat interface ( using [Mesop]([url](https://google.github.io/mesop/)) ! ) to help you get started.


**Add API Keys for Tools**
---------------------------------------------

If you want to use tools, you must create a `.env` file in your repo root directory and add API Keys for tools. Once you do that, you will need to restart the distribution server.

Tools that the model supports and which need API Keys --
- Brave for web search (https://api.search.brave.com/register)
- Wolfram for math operations (https://developer.wolframalpha.com/)

> **Tip** If you do not have API keys, you can still run the app without model having access to the tools.


**Start an App and Interact with the Server**
---------------------------------------------

Start an app (inline) and interact with it by running the following command:
```bash
mesop app/main.py
```
This will start a mesop app and you can go to `localhost:32123` to play with the chat interface.

<img src="demo.png" alt="Chat App" width="600"/>

Similar to this main app, you can also try other variants
- `PYTHONPATH=. mesop app/chat_with_custom_tools.py`  to showcase how custom tools are integrated
- `PYTHONPATH=. mesop app/chat_moderation_with_llama_guard.py`  to showcase how the app is modified to act as a chat moderator for safety

**Create agentic systems and interact with the Distribution server**
---------------------------------------------

NOTE: Ensure that Distribution server is still running.

```bash
cd <path-to-llama-agentic-system>
conda activate $ENV
llama distribution start --name inline-llama-8b --port 5000 # If not already started

PYTHONPATH=. python examples/scripts/vacation.py localhost 5000
```

You should see outputs to stdout of the form --
```bash
Environment: ipython
Tools: brave_search, wolfram_alpha, photogen

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024


User> I am planning a trip to Switzerland, what are the top 3 places to visit?
Final Llama Guard response shield_type=<BuiltinShield.llama_guard: 'llama_guard'> is_violation=False violation_type=None violation_return_message=None
Ran PromptGuardShield and got Scores: Embedded: 0.9999765157699585, Malicious: 1.1110752893728204e-05
StepType.shield_call> No Violation
role='user' content='I am planning a trip to Switzerland, what are the top 3 places to visit?'
StepType.inference> Switzerland is a beautiful country with a rich history, culture, and natural beauty. Here are three must-visit places to add to your itinerary: ....

```

> **Tip** You can optionally do `--disable-safety` in the scripts to avoid running safety shields all the time.


Feel free to reach out if you have questions.


**Using VirtualEnv instead of Conda**
-----------------------------
#### In Linux

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### For Windows

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # For Command Prompt
# or
.\venv\Scripts\Activate.ps1  # For PowerShell
# or
source venv\Scripts\activate  # For Git
```

The instructions thereafter (including `pip install -r requirements.txt` for installing the dependencies) remain the same.
