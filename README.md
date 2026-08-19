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

## `AsyncOpenAI` client setup

`build_kernel()` constructs the model connection in two steps rather than
letting `OpenAIChatCompletion` read environment variables itself:

```python
async_client = AsyncOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

kernel.add_service(
    OpenAIChatCompletion(
        service_id=SERVICE_ID,
        ai_model_id=AZURE_OPENAI_DEPLOYMENT,
        async_client=async_client,
    )
)
```

- **`AsyncOpenAI(base_url=..., api_key=...)`** is the plain OpenAI Python
  SDK's client, constructed by hand rather than through any Semantic Kernel
  API. `base_url` is `AZURE_OPENAI_ENDPOINT` from `.env`
  (`https://<resource>.services.ai.azure.com/openai/v1`) — see *A gotcha
  worth knowing* above for why it has to be this specific URL shape.
- **`ai_model_id=AZURE_OPENAI_DEPLOYMENT`** is the Foundry *deployment* name
  (`mini-mission`), not the underlying model name (`gpt-5-mini`) — the two
  are frequently different, and it's the deployment name the API actually
  routes on.
- **`async_client=async_client`** is what makes `OpenAIChatCompletion`
  generic rather than Azure-specific: instead of the connector building its
  own client internally from env vars (which is what happens if
  `async_client` is omitted), it just uses whatever client it's handed —
  including one already pointed at a non-standard `base_url` like Foundry's
  `/openai/v1` endpoint.
- **`AZURE_OPENAI_API_KEY` only has to flow through `AsyncOpenAI`'s
  constructor once.** `OpenAIChatCompletion` never sees `.env`'s API key
  directly — it only has the `async_client` object, which already carries
  the key internally.

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

## `ChatHistory` setup

`main.py` builds one `ChatHistory` object and threads it through the rest of
`main()` by hand, rather than calling `kernel.invoke_prompt(...)` per
question:

```python
history = ChatHistory()
history.add_system_message(SYSTEM_PERSONA)
history.add_user_message(f"In one sentence, confirm you're online and ready to help track {topic}.")
history.add_assistant_message(first_response_text)
```

- **`kernel.invoke_prompt()` is stateless** — each call is an isolated
  request with no memory of previous ones. `ChatHistory` is what memory
  actually is here: a plain list of messages that gets resent to the model
  in full on every call, so "the model remembers" just means "the client
  keeps sending the whole transcript so far."
- **`add_system_message` / `add_user_message` / `add_assistant_message` /
  `add_message`** append to that list with the corresponding role.
  `add_message` is the general form — used when appending an already-built
  `ChatMessageContent` (like a response the model just returned) rather than
  a plain string.
- **`get_chat_message_content(chat_history=history, settings=...)`** is what
  actually sends `history` to the model. `main.py` calls it repeatedly on
  the same `history` object, appending each new question with
  `add_user_message` and each reply with `add_message` before the next
  call — that accumulation is the entire mechanism multi-turn memory relies
  on.
- **`history` gets mutated by Semantic Kernel too, not just by us.** During
  automatic function calling, the auto-invoke loop appends the model's tool
  call and the function's result as their own entries directly into
  `history` — so by the end of `main.py`, `history` contains a mix of
  messages we added explicitly and ones SK added on our behalf.

## Persona setup

`SYSTEM_PERSONA` is one plain string, defined once near the top of
`main.py`:

```python
SYSTEM_PERSONA = (
    "You are the Mission Readiness Analytics Agent, an assistant that helps "
    "commanders and analysts interpret unit readiness data. You are precise, "
    "concise, and speak in a professional military-analytics tone. When you "
    "are uncertain or lack data, say so plainly instead of guessing."
)
```

A system message isn't a separate configuration channel — it's just the
first entry in whatever message list gets sent to the model, with role
`system` instead of `user`/`assistant`. That means `SYSTEM_PERSONA` has to
be injected wherever a conversation starts, and `main.py` does that in two
different places, because it drives conversations two different ways:

- **`ChatHistory`** — `history.add_system_message(SYSTEM_PERSONA)` is called
  once, right after `history = ChatHistory()`, before any user turns are
  added. Every `get_chat_message_content(...)` call that reuses `history`
  afterward automatically carries the persona along, since it's just message
  #0 in the list being sent each time.
- **`CONFIRM_STATUS_PROMPT`** — the prompt function has its *own*, separate
  conversation (rendered fresh by `kernel.invoke(...)`, not the shared
  `history` object), so the persona is embedded directly in the template via
  a `<message role="system">` tag rather than relying on external state.
  This keeps the function self-contained: it produces a correctly-personaed
  response no matter where it's invoked from, without depending on the
  caller to have set up a `ChatHistory` first.

After `confirm_status` runs, `main.py` re-adds the persona to `history`
(`history.add_system_message(SYSTEM_PERSONA)`) so the rest of the
conversation — everything from that point on — stays consistent with what
the prompt function already established.

## `CONFIRM_STATUS_PROMPT` topics

`confirm_status` (built from `CONFIRM_STATUS_PROMPT`) takes one template
variable, `{{$topic}}`, filled in via `KernelArguments(topic=...)` at
invocation time:

```python
CONFIRM_STATUS_PROMPT = (
    f'<message role="system">{SYSTEM_PERSONA}</message>\n'
    '<message role="user">In one sentence, confirm you\'re online and ready to help track {{$topic}}.</message>'
)
```

`main.py` calls it with `topic="mission readiness"`, but any short noun
phrase works — the persona and phrasing stay fixed, only the subject
changes:

- `"mission readiness"` (used in `main.py`)
- `"personnel readiness"`
- `"equipment status"`
- `"supply levels"`

This is what makes it a *reusable* `KernelFunction` rather than a one-off
string: the template is registered once in `build_kernel()`, and each call
site supplies its own `topic` without touching the function definition.

## Native function plugin registration

`ReadinessDataPlugin` is registered once, in `build_kernel()`:

```python
kernel.add_plugin(ReadinessDataPlugin(), plugin_name="ReadinessData")
```

- **`@kernel_function(name=..., description=...)`** marks a plain Python
  method as something the Kernel can invoke. `get_unit_status` and
  `compare_units` are both decorated this way in `ReadinessDataPlugin` — the
  `description` matters beyond documentation, since it's also what the model
  reads when deciding whether to call the function during automatic function
  calling.
- **`kernel.add_plugin(instance, plugin_name=...)`** is different from the
  `kernel.add_function(...)` used for `confirm_status`: instead of
  registering one function by hand, it scans the given instance for *every*
  `@kernel_function`-decorated method and registers each one under
  `plugin_name` in a single call. `_lookup` isn't decorated, so it's never
  registered or callable through the Kernel — it stays a private Python
  helper the two `@kernel_function` methods share internally.
- **This is why adding `compare_units` required no changes to
  `build_kernel()`.** It only had to be added to the `ReadinessDataPlugin`
  class body, decorated with `@kernel_function`; `kernel.add_plugin(...)`
  picked it up automatically the next time the kernel was built. Both
  functions end up addressable the same way afterward —
  `kernel.plugins["ReadinessData"]["get_unit_status"]` /
  `kernel.plugins["ReadinessData"]["compare_units"]` — and invocable
  identically via `kernel.invoke(function, KernelArguments(...))`.

## `get_unit_status` and `compare_units` topics

Unlike `confirm_status`'s free-form `{{$topic}}`, these two native functions
take a `unit` (or `unit_a`/`unit_b`) argument that's checked against a fixed
mock dataset in `ReadinessDataPlugin._MOCK_READINESS`. Matching is
case-insensitive and trims whitespace (`unit.strip().lower()`), but the name
otherwise has to match one of:

- `"Alpha Company"` — 78% ready, 2 vehicles down for maintenance
- `"Bravo Company"` — 92% ready, fully equipped
- `"Charlie Company"` — 61% ready, awaiting ammunition resupply
- `"2nd Battalion"` — 85% ready, minor equipment backlog clearing
- `"3rd Battalion"` — 70% ready, two platoons on training rotation

Anything else raises `UnitNotFoundError` — see the *Error handling* point
above for how that's caught (directly, via `kernel.invoke`) or handled
automatically (via the auto function-calling loop) depending on the call
site. `main.py`'s `"Delta Company"` calls exist specifically to exercise
that path.

## Automatic function calling settings

Automatic function calling is turned on per-call, not globally, by building
a separate `OpenAIChatPromptExecutionSettings` with
`function_choice_behavior` set:

```python
auto_settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={"included_plugins": ["ReadinessData"]}
    )
)
```

- **`FunctionChoiceBehavior.Auto(...)`** lets the model decide, per request,
  whether to call a function at all — as opposed to `.Required()` (must call
  one) or `.NoneInvoke()` (functions are visible but never auto-invoked).
- **`filters={"included_plugins": ["ReadinessData"]}`** scopes which
  functions the model is even offered. Here it's deliberately narrowed to
  just the `ReadinessData` plugin (`get_unit_status`, `compare_units`) so the
  model can't try to "call" `confirm_status` — a prompt function, not
  something meant to be invoked as a tool. Other supported filter keys:
  `excluded_plugins`, `included_functions`, `excluded_functions`.
- **`kernel=kernel` at the call site is required**, not optional — every
  `get_chat_message_content(..., settings=auto_settings, kernel=kernel)` call
  in `main.py` passes it, because the Kernel needs it to resolve and invoke
  whichever function the model chooses.

This same `auto_settings` object is reused across all four auto
function-calling calls in `main.py` (the Charlie Company question, the
unknown-unit question, the Alpha/Charlie comparison, and the battalion
comparison) — the model re-decides what to do each time based on the
current `ChatHistory`, not anything cached from the previous call.

### Inspecting what actually landed in `ChatHistory`

The printed assistant reply is only the *last* message the auto-invoke loop
produces. To see the tool call and tool result in between, record
`len(history.messages)` before the call and diff against it after:

```python
history_len_before = len(history.messages)
history.add_user_message("How ready is 2nd Battalion compared to 3rd Battalion?")
battalion_response = await chat_service.get_chat_message_content(
    chat_history=history, settings=auto_settings, kernel=kernel
)
history.add_message(battalion_response)

for message in history.messages[history_len_before:]:
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"  [{message.role}] tool call -> {item.function_name}({item.arguments})")
        elif isinstance(item, FunctionResultContent):
            print(f"  [{message.role}] tool result <- {item.result}")
```

For that question, four entries land in `history` from one
`get_chat_message_content` call:

```
[AuthorRole.USER] TextContent: How ready is 2nd Battalion compared to 3rd Battalion?
[AuthorRole.ASSISTANT] tool call -> compare_units({"unit_a":"2nd Battalion","unit_b":"3rd Battalion"})
[AuthorRole.TOOL] tool result <- 2nd Battalion (85% ready) is more mission-ready than 3rd Battalion (70% ready) by 15 percentage points.
[AuthorRole.ASSISTANT] TextContent: 2nd Battalion is 85% ready versus 3rd Battalion at 70%...
```

The `ASSISTANT` tool-call message and the `TOOL` result message are the two
entries SK's auto-invoke loop adds on its own (see *`ChatHistory` gets
mutated by Semantic Kernel too* above) — only the first `USER` message and
the final `ASSISTANT` text reply came from code we wrote directly.

## Automatic function calling vs. the older Planner classes

Semantic Kernel used to offer explicit **Planner** classes —
`SequentialPlanner`, `ActionPlanner`, `BasicPlanner`, `StepwisePlanner` —
now deprecated in favor of `FunctionChoiceBehavior`. They solved the same
problem (let an LLM decide which functions to call) in a fundamentally
different, older way:

- **Planners made a separate call to *plan*, before any execution
  happened.** You gave a planner a natural-language goal, and it used its
  own specially engineered prompt template — not the model's native
  tool-calling — to get the LLM to emit an explicit `Plan` object: a
  sequence of function calls with inputs/outputs wired between steps. Only
  after that plan existed did you execute it, either all at once
  (`SequentialPlanner`) or one step at a time with re-planning in between
  (`StepwisePlanner`, a ReAct-style loop).
- **`FunctionChoiceBehavior.Auto()` doesn't plan ahead of time at all.**
  There's no separate `Plan` object and no dedicated planning call. It rides
  directly on the model provider's own native tool-calling mechanism
  (OpenAI's `tools`/`tool_calls` fields), inline within a single
  `get_chat_message_content` call: the model returns either text or a tool
  call, and if it's a tool call, SK's auto-invoke loop (in
  `chat_completion_client_base.py`) invokes the function, appends the
  result to `ChatHistory`, and calls the model again — repeating until it
  gets a plain text response or hits `maximum_auto_invoke_attempts`. The
  `USER` → `ASSISTANT` (tool call) → `TOOL` (result) → `ASSISTANT` (text)
  sequence shown above *is* that loop, captured in `ChatHistory`.
- **The tradeoff:** Planners worked with any model, including ones with no
  native tool-calling support, because SK did the reasoning-about-function-
  use itself via prompting. `FunctionChoiceBehavior.Auto()` needs a model
  with real native tool-calling — but is simpler (a settings flag, no
  planner object to construct), and doesn't need a separate planning
  round-trip before execution can start.

## Error handling

`ReadinessDataPlugin._lookup` raises `UnitNotFoundError` (a plain
`Exception` subclass defined in `main.py`) for any unit not in
`_MOCK_READINESS`, instead of silently returning a "not found" string. Both
`get_unit_status` and `compare_units` call `_lookup` internally, so both get
this for free. Deliberately raising here — rather than returning a
soft-fallback string — matters because it's what lets a caller (our own code
or the model) recognize the failure as a distinct, structured error case,
not just another string it might mistake for real data.

That exception is handled differently depending on how the function was
invoked:

- **Direct `kernel.invoke(function, KernelArguments(...))`** — the Kernel
  doesn't let the original exception through as-is. It wraps whatever was
  raised in `KernelInvokeException`, with `UnitNotFoundError` attached as
  `__cause__`. So `main.py` catches `KernelInvokeException` and unwraps it:

  ```python
  try:
      await kernel.invoke(get_unit_status, KernelArguments(unit="Delta Company"))
  except KernelInvokeException as exc:
      if isinstance(exc.__cause__, UnitNotFoundError):
          print(f"Native function raised as expected: {exc.__cause__}")
      else:
          raise
  ```

- **Automatic function calling** — no code of ours is involved at all. SK's
  internal auto-invoke loop catches the exception itself, turns it into an
  error message, and feeds that back to the model as the tool's result. The
  model then responds to *that* — in `main.py`'s case, by asking for the
  correct unit designation instead of fabricating readiness numbers.

One side effect worth knowing about: Semantic Kernel logs every
function-invocation failure at `ERROR` level internally
(`logger.exception`/`logger.error` in `kernel.py`), regardless of whether
the calling code goes on to handle it — including our expected
`UnitNotFoundError` cases. `main.py` raises the `semantic_kernel` logger to
`CRITICAL` (see the *Notes* section below) specifically to keep that
internal noise out of the console output, without touching any of the
`try`/`except` logic above.

## Notes

- `.env` holds live credentials and is gitignored — never commit it.
  `.env.example` shows the expected shape with placeholder values.
- `main.py` forces `sys.stdout` to UTF-8 and raises the `semantic_kernel`
  logger to `CRITICAL`, purely for clean console output on Windows — see the
  comments at the top of the file for why.
