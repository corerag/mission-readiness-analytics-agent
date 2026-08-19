"""Connection probe for Azure AI Foundry Agent Service.

Standalone sanity check to run before building the real agent: confirms the
service-principal credentials in .env (AZURE_CLIENT_ID/SECRET/TENANT_ID) can
mint a token and that AZURE_AI_PROJECT_ENDPOINT actually points at a live
Foundry project with the Agent Service enabled. Read-only - it only lists
agents, it never creates one.

Usage: python probe_foundry_agent_service.py
"""

import os
import sys

from azure.ai.agents import AgentsClient
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.identity import ClientSecretCredential, CredentialUnavailableError
from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = [
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_AI_PROJECT_ENDPOINT",
]


def main() -> int:
    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        print(f"Missing .env values: {', '.join(missing)}")
        return 1

    endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]

    print(f"Endpoint:  {endpoint}")
    print(f"Tenant:    {os.environ['AZURE_TENANT_ID']}")
    print(f"Client ID: {os.environ['AZURE_CLIENT_ID']}")

    print("\n[1/3] Acquiring token via ClientSecretCredential...")
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    try:
        token = credential.get_token("https://ai.azure.com/.default")
        print(f"    OK - token acquired, expires_on={token.expires_on}")
    except (ClientAuthenticationError, CredentialUnavailableError) as exc:
        print(f"    FAILED - could not authenticate: {exc}")
        return 1

    print("\n[2/3] Constructing AgentsClient...")
    client = AgentsClient(endpoint=endpoint, credential=credential)
    print("    OK - client constructed")

    print("\n[3/3] Listing agents on the project (read-only)...")
    try:
        agents = list(client.list_agents(limit=1))
    except HttpResponseError as exc:
        print(f"    FAILED - request to Agent Service failed: {exc.status_code} {exc.message}")
        if exc.status_code == 404:
            print(
                "\n    AZURE_AI_PROJECT_ENDPOINT is likely missing the project path.\n"
                "    The Agent Service needs the project-scoped endpoint, shaped like:\n"
                "      https://<account>.services.ai.azure.com/api/projects/<project-name>\n"
                "    not just the bare resource host. Find the exact value on the Foundry\n"
                "    portal's Project Overview page and update .env."
            )
        elif exc.status_code == 401:
            print(
                "\n    The service principal authenticated fine but lacks an RBAC role\n"
                "    on this resource. In the Azure portal, go to the Foundry resource ->\n"
                "    Access control (IAM) -> Add role assignment, and grant it 'Azure AI\n"
                "    User' (or 'Cognitive Services User') scoped to the resource/project.\n"
                "    See https://aka.ms/FoundryPermissions"
            )
        return 1
    finally:
        client.close()

    print(f"    OK - request succeeded, {len(agents)} agent(s) returned in this page")
    if agents:
        print(f"    Example agent: id={agents[0].id} name={agents[0].name}")

    print("\nConnection to Foundry Agent Service verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
