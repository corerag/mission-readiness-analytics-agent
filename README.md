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

---

# Step 2: Microsoft Agent Framework (`main_af.py`)

[Microsoft Agent Framework](https://aka.ms/agent-framework) is Microsoft's
production successor to Semantic Kernel — same company, same underlying
lineage, but a rebuilt API surface (`agent-framework` on PyPI, unrelated
import path: `agent_framework`, not `semantic_kernel`). `main_af.py` sits
next to `main.py` in this repo and recreates the exact same Mission
Readiness agent — same persona, same `ReadinessDataPlugin` mock data and
method bodies, same sequence of demonstrations — so the two files can be
read side by side. `main.py` and Semantic Kernel are untouched.

## Setup

```bash
.venv\Scripts\pip install agent-framework-core agent-framework-openai   # Windows
# source .venv/bin/activate && pip install agent-framework-core agent-framework-openai  # macOS/Linux

.venv\Scripts\python main_af.py
```

`agent-framework-core` + `agent-framework-openai` is the lean install — only
what `main_af.py` actually imports. There's also a plain `agent-framework`
meta-package (what this project's `.venv` actually has installed) that pulls
in *every* optional connector — Anthropic, Bedrock, Gemini, Azure AI Search,
A2A, DevUI, and more — which is convenient for exploring but far more than
this file needs. No new `.env` variables — `main_af.py` reads the same three
Foundry values `main.py` does.

## Concept map: SK → Agent Framework

| Semantic Kernel | Agent Framework | Relationship |
|---|---|---|
| `Kernel` | *(nothing)* | Gone. AF has no central registry object — see below. |
| `OpenAIChatCompletion` | `OpenAIChatClient` | Same job (wrap an `AsyncOpenAI` client), renamed. |
| `kernel.add_service(...)` | `OpenAIChatClient(...)` | AF's client *is* the service — no separate registration step. |
| `@kernel_function` | `@tool` | Same purpose: mark a method as model-callable. |
| `kernel.add_plugin(instance, plugin_name=...)` | *(nothing — pass the bound methods directly)* | No plugin registry; a tool is just a reference you hold. |
| `kernel.plugins["X"]["y"]` lookup | `instance.y` attribute access | You already have the reference; there's nothing to look up. |
| `kernel.invoke(function, KernelArguments(...))` | calling the tool directly, e.g. `instance.y(arg=...)` | AF tools are plain callables; no invocation wrapper needed. |
| `KernelInvokeException` (wraps your exception) | *(nothing — your exception passes through as-is)* | Direct calls don't get wrapped at all. |
| `kernel.add_function(prompt=...)` (registered prompt template) | *(no equivalent — see below)* | Genuine capability gap, not a rename. |
| `ChatHistory` | plain `list[Message]` | Same role (an ordered transcript), no container class. |
| `history.add_user_message(...)` etc. | `history.append(Message("user", [...]))` | Same idea; you build the `Message` yourself. |
| `history.add_system_message(SYSTEM_PERSONA)` | `instructions=SYSTEM_PERSONA` on the agent, set once | Persona lives on the *agent*, not re-injected into every history. |
| auto-invoke mutates `history` in place | `agent.run()` returns only the *new* messages | You `history.extend(response.messages)` yourself every time. |
| `FunctionChoiceBehavior.Auto(filters={...})` per call | tools attached to the agent are auto-invoked on every `run()` by default | Opt-out (`options={"tool_choice": "none"}`), not opt-in. |
| `FunctionCallContent` / `FunctionResultContent` (separate classes) | one `Content` class, `.type` discriminator string | `isinstance(item, X)` becomes `content.type == "function_call"`. |
| `chat_service.get_chat_message_content(...)` | `agent.run(...)` | Same "send messages, get a reply" role. |

## Walkthrough

### 1. Client + agent construction — `build_agent()`

Same Foundry `/openai/v1` workaround as `main.py`: a hand-built
`AsyncOpenAI(base_url=..., api_key=...)` client, because the deployment only
speaks the unified OpenAI-compatible surface, not classic Azure REST. That
client is handed to `OpenAIChatClient(model=..., async_client=...)` —
directly parallel to SK's `OpenAIChatCompletion(async_client=...)`.

**Genuinely new:** there's no `Kernel` at all. SK's `Kernel` is a registry —
services and plugins get added to it, then looked up by ID/name later.
`chat_client.as_agent(instructions=..., tools=[...])` builds one self-
contained `Agent` object instead: the model connection, persona, and tool
list are all just constructor arguments, not things registered into a
separate container. There's nothing analogous to `kernel.get_service(...)`
because there's no registry to fetch from.

### 2. Tools — `ReadinessDataPlugin`

`get_unit_status` and `compare_units` are byte-for-byte the same method
bodies as `main.py`. Only the decorator changed: `@tool(name=..., description=...)`
instead of `@kernel_function(name=..., description=...)` — same two
keyword arguments, same purpose (the `description` is what the model reads
when deciding whether to call it).

**Same:** the decorator can sit directly on an instance method, and
accessing it through an instance (`plugin.get_unit_status`) gives you back a
callable already bound to `self` — no different from how
`@kernel_function` works.

**Genuinely new:** SK's `kernel.add_plugin(ReadinessDataPlugin(), plugin_name="ReadinessData")`
scans an instance and auto-registers *every* decorated method under a
plugin name, which is what later lets you write
`kernel.plugins["ReadinessData"]["get_unit_status"]`. AF has no plugin
registry to scan into: `build_agent()` just puts
`[plugin.get_unit_status, plugin.compare_units]` straight into the `tools=`
list. `_lookup` stays undecorated and private either way — a genuine
carryover, not a coincidence: neither framework's decorator scan/registry
touches a method that was never marked as a tool/kernel-function.

### 3. Persona

**Genuinely new mental model, not just a rename:** in `main.py`,
`SYSTEM_PERSONA` has to be injected in two separate places, because SK's
persona isn't attached to anything reusable — `history.add_system_message(...)`
for the `ChatHistory` path, and a `<message role="system">` tag baked
directly into `CONFIRM_STATUS_PROMPT` for the prompt-function path, since
that function has its own separate conversation. In `main_af.py`,
`instructions=SYSTEM_PERSONA` is set once on the `Agent` in `build_agent()`
and is automatically prepended as a system message to *every* `run()` call
that agent ever makes — the confirm-status call, the manual `history` list,
all of it — for the rest of the agent's life. One agent, one persona,
nothing to re-inject.

### 4. The "reusable prompt" — `confirm_status_message()`

This is the one piece with **no direct equivalent at all**, not just a
renamed API. SK's `CONFIRM_STATUS_PROMPT` is a `{{$topic}}`-templated string
registered once via `kernel.add_function(...)`, becoming a first-class,
independently invocable `KernelFunction` you can look up
(`kernel.plugins["MissionReadiness"]["confirm_status"]`) and call with
`kernel.invoke(function, KernelArguments(topic=...))`. Agent Framework's
core package has no template-registry object to parallel that — no
Handlebars/Jinja `{{$var}}` templating, no "prompt function" you register
and invoke by name. `confirm_status_message(topic)` in `main_af.py` is just
a plain Python function returning an f-string; "reusable and parameterized"
survives, but "a registered, independently-invokable framework object" does
not. If you actually need templated prompt files as portable artifacts, that
lives in a separate ecosystem tool (Prompty), not in `agent-framework`
itself.

### 5. Conversation history

**Same concept, no container class.** `ChatHistory` becomes a plain
`list[Message]` you manage by hand — appending is your job either way, SK
just wraps the list in a class with `add_user_message`/`add_assistant_message`
convenience methods; AF's `Message(role, contents)` constructor is the direct
equivalent of building a `ChatMessageContent` yourself.

**Concrete gotcha found while building this file:** `Message`'s `contents`
parameter wants a **list**. `Message("user", "some text")` — a bare string —
type-checks fine, because `str` itself satisfies `Sequence[str]`, but it
silently explodes the string into one single-character text `Content` per
letter instead of one text content holding the whole string. (The model
still answers correctly regardless — providers reassemble the text contents
before sending the request — but anything that inspects `message.contents`
afterward, like this file's own final debug loop, sees garbage until you
wrap it: `Message("user", ["some text"])`.) `main_af.py` does this
correctly throughout; the comment above the first `Message(...)` call flags
it.

**Genuinely new behavior, not just a naming difference:** SK's
`get_chat_message_content(chat_history=history, ...)` *mutates `history` in
place* — the auto-invoke loop appends the model's tool calls and results
directly onto the object you passed in. `agent.run(history)` does **not**
mutate your list; `response.messages` holds only the *new* messages that
call produced, and growing the transcript (`history.extend(response.messages)`)
is explicitly your responsibility, every single call. Forgetting this is the
easiest way to end up with an agent that mysteriously has no memory.

### 6. Automatic function calling

**Genuinely new default, worth internalizing:** SK requires opting in to
tool use *per call*, via a separate `OpenAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto(...))`
object — so a given `get_chat_message_content(...)` call either has tools
on or doesn't, explicitly. In `main_af.py`, tools are attached once, to the
*agent*, in `build_agent()`; from that point on, **every** `agent.run()`
call that agent makes can invoke them if the model decides to — including
the earlier "confirm you're online" and "what did you just say" calls,
which happen to never need a tool but technically had one available the
whole time. There's no equivalent of SK's `filters={"included_plugins": [...]}`
to reach for either — not because AF simplified it away, but because there's
nothing to filter: this agent's tool list only ever contains the two
readiness functions, so there's no prompt-function or unrelated tool that
could accidentally get exposed. To suppress tool use for one specific call,
the mechanism is `options={"tool_choice": "none"}` on that `run()` call, not
a filter on which tools are visible.

### 7. Error handling

**Same outcome, simpler direct path.** `plugin.get_unit_status(unit="Delta Company")`
called directly raises `UnitNotFoundError` completely unwrapped — no
`KernelInvokeException` to catch-and-unwrap the way `main.py` has to. AF
tools are plain Python callables under the hood; calling one directly *is*
calling plain Python, so nothing wraps the exception.

**Same outcome during auto function calling, but a real default-behavior
difference underneath:** Agent Framework's internal function-invocation loop
catches a tool's exception and feeds the model back an error result, just
like SK's auto-invoke loop does. But *by default*, it only tells the model
`"Error: Function failed."` — it does **not** forward your exception's actual
message. `build_agent()` sets
`function_invocation_configuration={"include_detailed_errors": True}` on the
`OpenAIChatClient` specifically to restore SK's behavior (the real
`"No readiness data on file for 'Delta Company'."` message reaching the
model), because without it the model can only report a generic failure —
it can't ask about the specific unit name the way SK's version does. This
is a deliberate, security-conscious default in AF (avoid leaking internal
exception text to the model/end user by default) that has no counterpart in
SK's behavior at all.

Same as `main.py`: Agent Framework logs every function-invocation failure at
`WARNING` under the `"agent_framework"` logger namespace, regardless of
whether the loop goes on to handle it — including our expected
`UnitNotFoundError` cases. `main_af.py` raises that logger to `CRITICAL` for
the same reason `main.py` raises `"semantic_kernel"`'s.

### Inspecting what actually landed in the message list

Same technique as `main.py` — record `len(history)` before appending the
next question, diff after:

```python
history_len_before = len(history)
history.append(Message("user", ["How ready is 2nd Battalion compared to 3rd Battalion?"]))
battalion_response = await agent.run(history)
history.extend(battalion_response.messages)

for message in history[history_len_before:]:
    for content in message.contents:
        if content.type == "function_call":
            print(f"  [{message.role}] tool call -> {content.name}({content.arguments})")
        elif content.type == "function_result":
            print(f"  [{message.role}] tool result <- {content.result}")
```

**Genuinely new API shape:** SK gives every content kind its own class
(`FunctionCallContent`, `FunctionResultContent`, `TextContent`, ...), so
`main.py`'s loop branches on `isinstance(item, FunctionCallContent)`. AF
uses a single `Content` class for everything, with a `.type` string field
(`"function_call"`, `"function_result"`, `"text"`, `"text_reasoning"`, ...)
as the discriminator instead — the loop branches on `content.type == "..."`.
`gpt-5-mini` being a reasoning model, you'll also see `"text_reasoning"`
content entries in the output that have no SK counterpart in this demo at
all — that's a model capability showing through, not a framework concept.
