import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.exceptions import KernelInvokeException
from semantic_kernel.functions import KernelArguments, kernel_function

load_dotenv()

# Windows defaults stdout to the system codepage (cp1252 here), which can't
# represent characters the model may emit (em dashes, curly quotes, etc). Terminals
# on this machine expect UTF-8, so a cp1252-encoded byte for those characters shows
# up as mojibake. Forcing stdout to UTF-8 makes the encoding match what's expected.
sys.stdout.reconfigure(encoding="utf-8")

# All SK submodules log under the "semantic_kernel" namespace via
# logging.getLogger(__name__). kernel.py logs every function-invocation failure
# at ERROR level even when the caller (us) handles it — like our expected
# UnitNotFoundError cases. Raising this logger's level to CRITICAL silences that
# internal noise without touching our own try/except logic or print() output.
logging.getLogger("semantic_kernel").setLevel(logging.CRITICAL)

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

SERVICE_ID = "mission-readiness-chat"

SYSTEM_PERSONA = (
    "You are the Mission Readiness Analytics Agent, an assistant that helps "
    "commanders and analysts interpret unit readiness data. You are precise, "
    "concise, and speak in a professional military-analytics tone. When you "
    "are uncertain or lack data, say so plainly instead of guessing."
)

# {{$topic}} is a template variable: this prompt is a reusable KernelFunction, not
# a one-off string. The <message> tags let a single prompt template carry both a
# system and a user turn, so the function's persona travels with it wherever it's
# invoked from.
CONFIRM_STATUS_PROMPT = (
    f'<message role="system">{SYSTEM_PERSONA}</message>\n'
    '<message role="user">In one sentence, confirm you\'re online and ready to help track {{$topic}}.</message>'
)


class UnitNotFoundError(Exception):
    """Raised when a unit has no readiness data on file."""


class ReadinessDataPlugin:
    """Native functions run as plain Python — no LLM call involved. Use these for
    anything that must be exact (data lookups, math, API calls), since a prompt
    function can only ever produce the model's best guess at an answer.
    """

    _MOCK_READINESS: dict[str, dict[str, object]] = {
        "alpha company": {"percent": 78, "notes": "2 vehicles down for maintenance, full personnel strength."},
        "bravo company": {"percent": 92, "notes": "fully equipped, no outstanding maintenance issues."},
        "charlie company": {"percent": 61, "notes": "awaiting an ammunition resupply, one squad on leave."},
    }

    def _lookup(self, unit: str) -> tuple[str, dict[str, object]]:
        key = unit.strip().lower()
        try:
            record = self._MOCK_READINESS[key]
        except KeyError:
            raise UnitNotFoundError(f"No readiness data on file for '{unit}'.") from None
        return key.title(), record

    @kernel_function(
        name="get_unit_status",
        description="Looks up the current readiness status for a named unit.",
    )
    def get_unit_status(self, unit: str) -> str:
        _, record = self._lookup(unit)
        return f"{record['percent']}% ready - {record['notes']}"

    @kernel_function(
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


def build_kernel() -> Kernel:
    # The Foundry deployment only exposes the newer OpenAI-compatible "/openai/v1"
    # surface, not the classic Azure OpenAI REST API. So instead of AzureChatCompletion
    # (which builds classic /openai/deployments/{name}/... URLs), we build a plain
    # AsyncOpenAI client pointed at that endpoint ourselves and hand it to
    # OpenAIChatCompletion, SK's generic (non-Azure) OpenAI connector.
    async_client = AsyncOpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=SERVICE_ID,
            ai_model_id=AZURE_OPENAI_DEPLOYMENT,
            async_client=async_client,
        )
    )

    # Registering a prompt as a KernelFunction files it under kernel.plugins, so it
    # can be looked up and invoked by name from anywhere, instead of re-typing the
    # prompt string every time we want to ask this question.
    kernel.add_function(
        plugin_name="MissionReadiness",
        function_name="confirm_status",
        prompt=CONFIRM_STATUS_PROMPT,
        description="Asks the agent to confirm it's online and ready for a given readiness topic.",
    )

    # add_plugin scans the instance for @kernel_function-decorated methods and
    # registers each one under this plugin name, the same way add_function did
    # above for a single prompt function.
    kernel.add_plugin(ReadinessDataPlugin(), plugin_name="ReadinessData")

    return kernel


async def main() -> None:
    kernel = build_kernel()

    # Pulling the service straight off the kernel (by the service_id we registered
    # it under) lets us drive get_chat_message_content ourselves, instead of going
    # through kernel.invoke_prompt() — which is stateless and wouldn't demonstrate
    # multi-turn memory.
    chat_service = kernel.get_service(service_id=SERVICE_ID)
    settings = OpenAIChatPromptExecutionSettings()

    # Look up the reusable function by plugin/function name and invoke it with a
    # topic argument, rather than hand-building a ChatHistory for it.
    confirm_status = kernel.plugins["MissionReadiness"]["confirm_status"]
    topic = "mission readiness"
    first_result = await kernel.invoke(confirm_status, KernelArguments(topic=topic))
    first_response_text = str(first_result)
    print(f"Assistant: {first_response_text}")

    # kernel.invoke() rendered and ran the function in its own internal ChatHistory,
    # so we replay that exchange into our own running history here, to keep the
    # multi-turn memory test below working off a complete transcript.
    history = ChatHistory()
    history.add_system_message(SYSTEM_PERSONA)
    history.add_user_message(f"In one sentence, confirm you're online and ready to help track {topic}.")
    history.add_assistant_message(first_response_text)

    # This second prompt only makes sense to answer correctly if the model can see
    # the first exchange — proving the history (not just the connection) works.
    history.add_user_message("What did you just say, word for word?")
    second_response = await chat_service.get_chat_message_content(
        chat_history=history, settings=settings
    )
    print(f"Assistant: {second_response}")
    history.add_message(second_response)

    # Native functions are invoked through the kernel the exact same way prompt
    # functions are — kernel.invoke(function, KernelArguments(...)) — even though
    # no LLM call happens here at all; get_unit_status just runs as Python.
    get_unit_status = kernel.plugins["ReadinessData"]["get_unit_status"]
    unit_result = await kernel.invoke(get_unit_status, KernelArguments(unit="Bravo Company"))
    print(f"Native function result: {unit_result}")

    # kernel.invoke() doesn't let a native function's exception through as-is: it
    # wraps whatever was raised in a KernelInvokeException, with our original
    # UnitNotFoundError attached as __cause__. So a direct call needs to catch the
    # wrapper and unwrap it, not catch UnitNotFoundError directly.
    try:
        await kernel.invoke(get_unit_status, KernelArguments(unit="Delta Company"))
    except KernelInvokeException as exc:
        if isinstance(exc.__cause__, UnitNotFoundError):
            print(f"Native function raised as expected: {exc.__cause__}")
        else:
            raise

    # Automatic function calling: instead of us deciding to call get_unit_status,
    # the model sees it's available (scoped here to just the ReadinessData plugin,
    # so it won't try to "call" our prompt function too) and decides for itself
    # whether the user's question requires it. get_chat_message_content then runs
    # a call -> detect tool call -> invoke function -> call again loop internally,
    # and appends every step of that (tool call + tool result) into `history`.
    auto_settings = OpenAIChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto(
            filters={"included_plugins": ["ReadinessData"]}
        )
    )

    history.add_user_message("What's Charlie Company's current readiness status?")
    auto_response = await chat_service.get_chat_message_content(
        chat_history=history, settings=auto_settings, kernel=kernel
    )
    print(f"Assistant (auto function calling): {auto_response}")
    history.add_message(auto_response)

    # Ask about a unit that doesn't exist. get_unit_status will raise
    # UnitNotFoundError, but this time nothing in our code catches it — the auto
    # function calling loop in chat_completion_client_base.py does that for us,
    # turning the exception into a tool-result error message the model sees and
    # can respond to, instead of the exception propagating and crashing the script.
    history.add_user_message("What's Delta Company's current readiness status?")
    unknown_unit_response = await chat_service.get_chat_message_content(
        chat_history=history, settings=auto_settings, kernel=kernel
    )
    print(f"Assistant (auto function calling, unknown unit): {unknown_unit_response}")
    history.add_message(unknown_unit_response)

    # Direct invoke of the new function, same pattern as get_unit_status earlier.
    compare_units = kernel.plugins["ReadinessData"]["compare_units"]
    comparison_result = await kernel.invoke(
        compare_units, KernelArguments(unit_a="Alpha Company", unit_b="Bravo Company")
    )
    print(f"Native function result: {comparison_result}")

    # Now via auto function calling, with two candidate functions available
    # (get_unit_status and compare_units) — this tests that the model picks the
    # right tool for a comparison question rather than defaulting to a lookup.
    history.add_user_message("Which is more ready right now, Alpha Company or Charlie Company?")
    comparison_response = await chat_service.get_chat_message_content(
        chat_history=history, settings=auto_settings, kernel=kernel
    )
    print(f"Assistant (auto function calling, comparison): {comparison_response}")
    history.add_message(comparison_response)


if __name__ == "__main__":
    asyncio.run(main())
