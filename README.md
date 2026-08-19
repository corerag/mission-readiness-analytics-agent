# Mission Readiness Analytics Agent

A learning project for Semantic Kernel (Python), connected to an Azure AI
Foundry `gpt-5-mini` deployment. `main.py` walks through the core building
blocks of the Kernel one at a time, printing the result of each step so you
can see what's actually happening.

## Prerequisites

- Python 3.12+
- An Azure AI Foundry resource with a chat-capable model deployed (this
  project was built against a `gpt-5-mini` deployment named `mini-mission`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install semantic-kernel python-dotenv   # Windows
# source .venv/bin/activate && pip install ...            # macOS/Linux

cp .env.example .env
```

Fill in `.env` with your resource's values:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

Then run it:

```bash
.venv\Scripts\python main.py
```

## A gotcha worth knowing

`AZURE_OPENAI_ENDPOINT` points at the newer `/openai/v1` unified endpoint
(`https://<resource>.services.ai.azure.com/openai/v1`), not the classic
`https://<resource>.openai.azure.com` host. Some Azure AI Foundry deployments
only support the newer surface — if yours doesn't respond to
`AzureChatCompletion` (SK's Azure-specific connector, which builds classic
`/openai/deployments/{name}/...` URLs and 404s against `/openai/v1`
resources), this project works around it by building a plain `AsyncOpenAI`
client pointed at the `/openai/v1` endpoint and handing that to
`OpenAIChatCompletion` (SK's generic, non-Azure connector) instead. See
`build_kernel()` in `main.py`.

## What `main.py` demonstrates

Each step prints its own output so you can see it work in isolation:

1. **Kernel + AI service** — a `Kernel` is a registry; it can't talk to a
   model until an AI service (`OpenAIChatCompletion`, here backed by a
   manually-configured `AsyncOpenAI` client) is registered with it.
2. **A reusable prompt function** — `confirm_status` is a prompt template
   registered once via `kernel.add_function(...)`, parameterized with
   `{{$topic}}`, and invoked with `kernel.invoke(function, KernelArguments(...))`
   instead of a hand-typed prompt string.
3. **Conversation history** — a `ChatHistory` object, seeded with a system
   persona (`SYSTEM_PERSONA`), accumulates turns so later prompts can
   reference earlier ones. Stateless calls like `kernel.invoke_prompt()`
   can't do this; driving `ChatHistory` + `get_chat_message_content`
   directly can.
4. **A native function plugin** — `ReadinessDataPlugin` (`get_unit_status`,
   `compare_units`) runs as plain Python, not an LLM call, for anything
   that must be exactly right rather than plausibly guessed.
5. **Automatic function calling** — with
   `function_choice_behavior=FunctionChoiceBehavior.Auto(...)` set on the
   execution settings, the model decides for itself whether a user's
   question requires calling a registered function, and the Kernel handles
   detecting the tool call, invoking it, and feeding the result back.
6. **Error handling** — `get_unit_status`/`compare_units` raise
   `UnitNotFoundError` for unknown units. A direct `kernel.invoke(...)` call
   wraps it in `KernelInvokeException` (caught and unwrapped in `main()`);
   the automatic function-calling loop catches it internally and feeds the
   model an error message it can respond to gracefully, with no extra code
   required on our end.

## Notes

- `.env` holds live credentials and is gitignored — never commit it.
  `.env.example` shows the expected shape with placeholder values.
- `main.py` forces `sys.stdout` to UTF-8 and raises the `semantic_kernel`
  logger to `CRITICAL`, purely for clean console output on Windows — see the
  comments at the top of the file for why.
