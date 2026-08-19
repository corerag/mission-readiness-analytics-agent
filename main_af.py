import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agent_framework import Agent, Message, tool
from agent_framework.openai import OpenAIChatClient

load_dotenv()

# Same Windows console encoding fix as main.py - see the comment there for why.
sys.stdout.reconfigure(encoding="utf-8")

# Agent Framework logs every function-invocation failure at WARNING level under the
# "agent_framework" namespace, the same way semantic_kernel does under its own
# namespace in main.py. Same fix, same reasoning: raise it so our own try/except
# output isn't buried under internal noise for the UnitNotFoundError cases below.
logging.getLogger("agent_framework").setLevel(logging.CRITICAL)

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

SYSTEM_PERSONA = (
    "You are the Mission Readiness Analytics Agent, an assistant that helps "
    "commanders and analysts interpret unit readiness data. You are precise, "
    "concise, and speak in a professional military-analytics tone. When you "
    "are uncertain or lack data, say so plainly instead of guessing."
)


def confirm_status_message(topic: str) -> str:
    """The user-turn text for the "confirm you're online" check-in.

    main.py's equivalent (CONFIRM_STATUS_PROMPT) was a registered, reusable
    KernelFunction with the persona baked into its own <message role="system">
    tag, because a prompt function is a standalone object that might be invoked
    from anywhere. Agent Framework has no equivalent "registered prompt
    function" object - an Agent's `instructions` already carry the persona on
    every run() call, so a parameterized prompt is just a plain Python
    function returning a string. This is a genuine capability gap versus SK,
    not just a renamed API - see the README section on this file for detail.
    """
    return f"In one sentence, confirm you're online and ready to help track {topic}."


class UnitNotFoundError(Exception):
    """Raised when a unit has no readiness data on file. Identical to main.py's."""


class ReadinessDataPlugin:
    """Native functions run as plain Python - no LLM call involved. Same class,
    same mock data, same method bodies as main.py's ReadinessDataPlugin. Only
    the decorator changes: @tool instead of @kernel_function.
    """

    _MOCK_READINESS: dict[str, dict[str, object]] = {
        "alpha company": {
            "name": "Alpha Company",
            "percent": 78,
            "notes": "2 vehicles down for maintenance, full personnel strength.",
        },
        "bravo company": {
            "name": "Bravo Company",
            "percent": 92,
            "notes": "fully equipped, no outstanding maintenance issues.",
        },
        "charlie company": {
            "name": "Charlie Company",
            "percent": 61,
            "notes": "awaiting an ammunition resupply, one squad on leave.",
        },
        "2nd battalion": {
            "name": "2nd Battalion",
            "percent": 85,
            "notes": "fully staffed, minor equipment backlog clearing this week.",
        },
        "3rd battalion": {
            "name": "3rd Battalion",
            "percent": 70,
            "notes": "reduced personnel strength, two platoons on training rotation.",
        },
    }

    def _lookup(self, unit: str) -> tuple[str, dict[str, object]]:
        key = unit.strip().lower()
        try:
            record = self._MOCK_READINESS[key]
        except KeyError:
            raise UnitNotFoundError(f"No readiness data on file for '{unit}'.") from None
        return record["name"], record

    @tool(
        name="get_unit_status",
        description="Looks up the current readiness status for a named unit.",
    )
    def get_unit_status(self, unit: str) -> str:
        _, record = self._lookup(unit)
        return f"{record['percent']}% ready - {record['notes']}"

    @tool(
        name="compare_units",
        description="Compares the readiness percentage of two named units and states which is more ready.",
    )
    def compare_units(self, unit_a: str, unit_b: str) -> str:
        name_a, record_a = self._lookup(unit_a)
        name_b, record_b = self._lookup(unit_b)
        pct_a, pct_b = record_a["percent"], record_b["percent"]

        if pct_a == pct_b:
            return f"{name_a} and {name_b} are equally ready, both at {pct_a}%."

        leader, leader_pct = (name_a, pct_a) if pct_a > pct_b else (name_b, pct_b)
        laggard, laggard_pct = (name_b, pct_b) if pct_a > pct_b else (name_a, pct_a)
        return (
            f"{leader} ({leader_pct}% ready) is more mission-ready than "
            f"{laggard} ({laggard_pct}% ready) by {leader_pct - laggard_pct} percentage points."
        )


def build_agent() -> tuple[Agent, ReadinessDataPlugin]:
    # Identical Foundry /openai/v1 workaround as main.py's build_kernel() - see
    # that function's comment and the README's "gotcha" section for why a plain
    # AsyncOpenAI client is required instead of an Azure-specific connector.
    async_client = AsyncOpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    # OpenAIChatClient is Agent Framework's equivalent of SK's OpenAIChatCompletion:
    # a generic (non-Azure) connector that accepts an already-configured
    # AsyncOpenAI client instead of building its own from env vars.
    #
    # include_detailed_errors is a genuine behavioral difference from SK worth
    # knowing up front: by default, when a tool raises during automatic function
    # calling, Agent Framework feeds the model back only a generic
    # "Error: Function failed." - not the exception's actual message. SK's
    # auto-invoke loop passes the real exception text through. Setting this
    # flag restores that SK-like behavior (see main() for where this matters).
    chat_client = OpenAIChatClient(
        model=AZURE_OPENAI_DEPLOYMENT,
        async_client=async_client,
        function_invocation_configuration={"include_detailed_errors": True},
    )

    plugin = ReadinessDataPlugin()

    # There is no add_plugin()/add_function() registry step here. A Kernel is a
    # registry you look functions up in by plugin-name/function-name string; an
    # Agent just holds a flat list of the specific callables you hand it -
    # plugin.get_unit_status and plugin.compare_units are already the
    # references you'll use directly later, not names to resolve later.
    #
    # instructions=SYSTEM_PERSONA is also doing more work here than
    # history.add_system_message() does in main.py: it's set once, on the
    # Agent object itself, and gets prepended as a system message to *every*
    # run() call this agent makes for the rest of its life - no need to
    # re-inject it into each conversation's message list by hand.
    agent = chat_client.as_agent(
        name="mission-readiness",
        instructions=SYSTEM_PERSONA,
        tools=[plugin.get_unit_status, plugin.compare_units],
    )

    return agent, plugin


async def main() -> None:
    agent, plugin = build_agent()

    # No kernel.invoke(function, KernelArguments(...)) step: confirm_status_message
    # is a plain function, so we just call it and hand the string straight to
    # agent.run(). The persona is already on the agent from build_agent(); it
    # doesn't need to be embedded in this string the way CONFIRM_STATUS_PROMPT
    # embedded it for main.py.
    topic = "mission readiness"
    first_message = confirm_status_message(topic)
    first_response = await agent.run(first_message)
    first_response_text = first_response.text
    print(f"Assistant: {first_response_text}")

    # Message is Agent Framework's equivalent of a single ChatHistory entry
    # (role + content), but there's no ChatHistory container class to hold
    # them - a plain list works, and it's ours to manage. Note there's no
    # add_system_message() call here: unlike SK's history, this list never
    # needs the persona in it, because instructions=SYSTEM_PERSONA on the
    # agent already puts it in front of every run() call automatically.
    #
    # Gotcha: Message(role, contents) wants contents as a *list*
    # (Sequence[Content | str | Mapping]). Message("user", first_message) -
    # passing the bare string - type-checks fine, because str itself satisfies
    # Sequence[str], but it silently explodes the string into one
    # single-character text Content per letter. Always wrap: [first_message].
    history: list[Message] = [
        Message("user", [first_message]),
        Message("assistant", [first_response_text]),
    ]

    # This second prompt only makes sense to answer correctly if the model can
    # see the first exchange - same test as main.py, same reason.
    history.append(Message("user", ["What did you just say, word for word?"]))
    second_response = await agent.run(history)
    print(f"Assistant: {second_response.text}")
    # agent.run() does NOT mutate `history` in place the way SK's
    # get_chat_message_content() mutates ChatHistory. response.messages holds
    # only the *new* messages this call produced, so growing the transcript is
    # our job: extend() what came back onto our own list, every time.
    history.extend(second_response.messages)

    # Direct invocation: this is where the story genuinely simplifies versus
    # SK. plugin.get_unit_status is a real Python object with a __call__ that
    # runs the wrapped function directly - so calling it *is* calling plain
    # Python, no kernel.invoke(function, KernelArguments(...)) indirection
    # required, and no result wrapper to unwrap.
    unit_result = plugin.get_unit_status(unit="Bravo Company")
    print(f"Native function result: {unit_result}")

    # And no KernelInvokeException wrapper either: since __call__ doesn't
    # catch anything, UnitNotFoundError comes through exactly as raised.
    try:
        plugin.get_unit_status(unit="Delta Company")
    except UnitNotFoundError as exc:
        print(f"Native function raised as expected: {exc}")

    # Automatic function calling: unlike SK's FunctionChoiceBehavior.Auto(),
    # which is opted into per-call via a separate execution-settings object,
    # tool use here has been "on" since build_agent() - any run() call this
    # agent makes can invoke get_unit_status/compare_units if the model
    # decides to, with no per-call settings object required. (There's also no
    # included_plugins-style filter to reach for: this agent's tool list only
    # ever contains these two functions - there's no prompt-function
    # equivalent that could accidentally get exposed as something "callable."
    # If you needed to suppress tool use for one specific call, the mechanism
    # is options={"tool_choice": "none"} on that run() call, not a filter.)
    history.append(Message("user", ["What's Charlie Company's current readiness status?"]))
    auto_response = await agent.run(history)
    print(f"Assistant (auto function calling): {auto_response.text}")
    history.extend(auto_response.messages)

    # Ask about a unit that doesn't exist. get_unit_status raises
    # UnitNotFoundError again, but this time nothing in our code catches it -
    # Agent Framework's internal function-invocation loop does, the same way
    # SK's auto-invoke loop does. The difference is what the model gets told:
    # with include_detailed_errors=True (set in build_agent()), it sees the
    # real "No readiness data on file for 'Delta Company'." message, so it can
    # respond specifically instead of just reporting a generic failure.
    history.append(Message("user", ["What's Delta Company's current readiness status?"]))
    unknown_unit_response = await agent.run(history)
    print(f"Assistant (auto function calling, unknown unit): {unknown_unit_response.text}")
    history.extend(unknown_unit_response.messages)

    # Direct invoke of the new function, same pattern as get_unit_status earlier.
    comparison_result = plugin.compare_units(unit_a="Alpha Company", unit_b="Bravo Company")
    print(f"Native function result: {comparison_result}")

    # Auto function calling again, now with two candidate tools available -
    # tests that the model picks compare_units for a comparison question
    # rather than defaulting to a lookup.
    history.append(Message("user", ["Which is more ready right now, Alpha Company or Charlie Company?"]))
    comparison_response = await agent.run(history)
    print(f"Assistant (auto function calling, comparison): {comparison_response.text}")
    history.extend(comparison_response.messages)

    # One more comparison, over units that aren't Companies, to confirm
    # compare_units gets picked generally. Recording len(history) before the
    # call isolates exactly what this run() call appended, the same technique
    # main.py uses on ChatHistory - just applied to a plain list here.
    history_len_before = len(history)
    history.append(Message("user", ["How ready is 2nd Battalion compared to 3rd Battalion?"]))
    battalion_response = await agent.run(history)
    print(f"Assistant (auto function calling, battalion comparison): {battalion_response.text}")
    history.extend(battalion_response.messages)

    # main.py's inspection loop does isinstance(item, FunctionCallContent) /
    # isinstance(item, FunctionResultContent) checks, because SK gives each
    # content kind its own class. Agent Framework uses one Content class for
    # everything, with a .type string discriminator ("function_call",
    # "function_result", "text", "text_reasoning", ...) instead - so this is
    # an if/elif on content.type rather than a chain of isinstance checks.
    print("\nMessages added by this call:")
    for message in history[history_len_before:]:
        for content in message.contents:
            if content.type == "function_call":
                print(f"  [{message.role}] tool call -> {content.name}({content.arguments})")
            elif content.type == "function_result":
                print(f"  [{message.role}] tool result <- {content.result}")
            else:
                text = getattr(content, "text", None)
                print(f"  [{message.role}] {content.type}: {text if text is not None else content}")


if __name__ == "__main__":
    asyncio.run(main())
