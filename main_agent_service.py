import logging
import os
import sys

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, MessageRole, RunStatus, ToolSet
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv

load_dotenv()

# Same Windows console encoding fix as main.py/main_af.py - see main.py's comment for why.
sys.stdout.reconfigure(encoding="utf-8")

# Same noise-suppression pattern as main.py's "semantic_kernel" logger and
# main_af.py's "agent_framework" logger. This SDK logs
# "Error executing function ..." and "Tool outputs contain errors - retrying"
# at WARNING level for the exact Delta-Company failure this script triggers on
# purpose below - expected here, not worth the noise.
logging.getLogger("azure.ai.agents").setLevel(logging.CRITICAL)

AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
AZURE_CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]
AZURE_AI_PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

SYSTEM_PERSONA = (
    "You are the Mission Readiness Analytics Agent, an assistant that helps "
    "commanders and analysts interpret unit readiness data. You are precise, "
    "concise, and speak in a professional military-analytics tone. When you "
    "are uncertain or lack data, say so plainly instead of guessing."
)


def confirm_status_message(topic: str) -> str:
    return f"In one sentence, confirm you're online and ready to help track {topic}."


class UnitNotFoundError(Exception):
    """Raised when a unit has no readiness data on file. Identical to main.py's."""


class ReadinessDataPlugin:
    """Same mock data and method bodies as main.py/main_af.py's plugin.

    No @kernel_function or @tool decorator here, and that's not a style choice -
    FunctionTool (below) has no equivalent decorator to hang a name/description
    off of. It builds each tool's JSON schema by reflecting on the plain function:
    the description comes from the docstring's first line, and per-parameter
    descriptions come from parsing ":param name: ..." lines out of the docstring
    body. Skip the docstring and the model just sees "No description" for that
    parameter, so - unlike the other two files - the docstring here isn't
    optional documentation, it's the schema.
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

    def get_unit_status(self, unit: str) -> str:
        """Looks up the current readiness status for a named unit.

        :param unit: The name of the unit to look up, e.g. "Bravo Company".
        """
        _, record = self._lookup(unit)
        return f"{record['percent']}% ready - {record['notes']}"

    def compare_units(self, unit_a: str, unit_b: str) -> str:
        """Compares the readiness percentage of two named units and states which is more ready.

        :param unit_a: The name of the first unit to compare.
        :param unit_b: The name of the second unit to compare.
        """
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


def send_and_get_reply(client: AgentsClient, thread_id: str, agent_id: str, text: str) -> str:
    """Post a user message to the thread, run the agent, and return its reply text.

    runs.create_and_process is the blocking, poll-until-done helper: it submits
    the run, polls run.status, and - because enable_auto_function_calls() was
    called on this client for our toolset - executes any requires_action tool
    calls locally and resubmits their outputs, looping until the run reaches a
    terminal status. There's no separate "await response" step the way
    agent.run() returns one in main_af.py; by the time create_and_process
    returns, the whole exchange (including any tool calls) already happened.
    """
    client.messages.create(thread_id=thread_id, role=MessageRole.USER, content=text)
    run = client.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)
    if run.status != RunStatus.COMPLETED:
        raise RuntimeError(f"Run ended in status {run.status}: {run.last_error}")

    reply = client.messages.get_last_message_text_by_role(thread_id=thread_id, role=MessageRole.AGENT)
    if reply is None:
        raise RuntimeError("Run completed but no assistant message was found on the thread.")
    return reply.text.value


def main() -> None:
    credential = ClientSecretCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
    )
    plugin = ReadinessDataPlugin()

    with AgentsClient(endpoint=AZURE_AI_PROJECT_ENDPOINT, credential=credential) as client:
        # ToolSet + FunctionTool builds the JSON schema handed to the model.
        # enable_auto_function_calls is the separate step that lets
        # runs.create_and_process actually invoke these Python functions locally
        # when the model calls them - passing toolset to create_agent alone only
        # gives the model the tool definitions, not a way to execute them.
        toolset = ToolSet()
        toolset.add(FunctionTool(functions={plugin.get_unit_status, plugin.compare_units}))
        client.enable_auto_function_calls(toolset)

        # Unlike SK's Kernel or Agent Framework's Agent, which are purely local,
        # in-process objects that vanish when the script exits, create_agent()
        # creates a real, persistent resource in the Foundry project - it still
        # exists (and shows up in probe_foundry_agent_service.py's list_agents)
        # after this script ends. The try/finally below deletes it on the way
        # out so repeated runs don't pile up agents in the project.
        agent = client.create_agent(
            model=AZURE_OPENAI_DEPLOYMENT,
            name="mission-readiness",
            instructions=SYSTEM_PERSONA,
            toolset=toolset,
        )
        thread = client.threads.create()

        try:
            topic = "mission readiness"
            first_response = send_and_get_reply(client, thread.id, agent.id, confirm_status_message(topic))
            print(f"Assistant: {first_response}")

            # No history list to build or extend here, unlike main.py's ChatHistory
            # or main_af.py's list[Message]. The thread itself is the conversation
            # state, held server-side by the Agent Service - every message we've
            # posted and every reply the agent has given is already on it, so this
            # second question just works without us re-sending any prior turns.
            second_response = send_and_get_reply(
                client, thread.id, agent.id, "What did you just say, word for word?"
            )
            print(f"Assistant: {second_response}")

            # Direct invocation: same as the other two files, calling plain Python.
            unit_result = plugin.get_unit_status(unit="Bravo Company")
            print(f"Native function result: {unit_result}")

            try:
                plugin.get_unit_status(unit="Delta Company")
            except UnitNotFoundError as exc:
                print(f"Native function raised as expected: {exc}")

            # Auto function calling.
            charlie_response = send_and_get_reply(
                client, thread.id, agent.id, "What's Charlie Company's current readiness status?"
            )
            print(f"Assistant (auto function calling): {charlie_response}")

            # Ask about a unit that doesn't exist. Here's the real behavioral
            # difference from main_af.py's include_detailed_errors flag: there's
            # no equivalent toggle. FunctionTool.execute() unconditionally catches
            # every exception - including our UnitNotFoundError - and feeds the
            # model back a JSON string like {"error": "Error executing function
            # 'get_unit_status': No readiness data on file for 'Delta Company'."}
            # rather than either a generic failure or a bare re-raise. The real
            # message reaches the model either way; it just isn't something this
            # script can choose to suppress or restore per call.
            delta_response = send_and_get_reply(
                client, thread.id, agent.id, "What's Delta Company's current readiness status?"
            )
            print(f"Assistant (auto function calling, unknown unit): {delta_response}")

            comparison_result = plugin.compare_units(unit_a="Alpha Company", unit_b="Bravo Company")
            print(f"Native function result: {comparison_result}")

            alpha_charlie_response = send_and_get_reply(
                client, thread.id, agent.id, "Which is more ready right now, Alpha Company or Charlie Company?"
            )
            print(f"Assistant (auto function calling, comparison): {alpha_charlie_response}")

            battalion_response = send_and_get_reply(
                client, thread.id, agent.id, "How ready is 2nd Battalion compared to 3rd Battalion?"
            )
            print(f"Assistant (auto function calling, battalion comparison): {battalion_response}")

            # Run steps are this file's equivalent of main_af.py's content.type
            # inspection loop, but scoped to a single run rather than a message
            # range - there's no shared "messages added by this call" list to
            # slice, since the thread itself is the only record of what happened.
            # Note what's missing versus AF's function_result content: this API's
            # RunStepFunctionToolCallDetails exposes the tool call's name and
            # arguments, but not its output - the executed result only shows up
            # indirectly, in the assistant's final reply text above.
            last_run = list(client.runs.list(thread_id=thread.id, limit=1))[0]
            print("\nTool calls made during the last run:")
            for step in client.run_steps.list(thread_id=thread.id, run_id=last_run.id):
                if step.step_details.type == "tool_calls":
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == "function":
                            print(f"  tool call -> {tool_call.function.name}({tool_call.function.arguments})")
        finally:
            client.threads.delete(thread.id)
            client.delete_agent(agent.id)


if __name__ == "__main__":
    main()
