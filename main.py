
import json
import logging
import os
import sys
import uuid

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

from agent_orchestrator import (
    build_inventory_agent,
    build_refund_agent,
    build_policy_agent,
    build_communication_agent,
    build_orchestrator_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

MEMORY_ID = "udacity_agentcore_memory-0BaJn3FBpI"


@app.entrypoint
def invoke(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise ValueError("The request payload must be a JSON object.")

    session_id = str(
        payload.get("session_id")
        or payload.get("sessionId")
        or uuid.uuid4()
    )
    customer_id = str(
        payload.get("customer_id")
        or payload.get("customerId")
        or "anonymous"
    )
    user_message = (
        payload.get("user_message")
        or payload.get("message")
        or payload.get("prompt")
    )

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError(
            "Provide a non-empty user_message, message, or prompt."
        )

    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=customer_id,
        batch_size=1,
    )
    memory_manager = AgentCoreMemorySessionManager(
        memory_config,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    try:
        inventory = build_inventory_agent()
        refund = build_refund_agent()
        policy = build_policy_agent()
        communication = build_communication_agent()

        orchestrator = build_orchestrator_agent(
            inventory,
            refund,
            policy,
            communication,
            session_manager=memory_manager,
        )

        enriched_message = (
            f"[Session ID: {session_id}] "
            f"[Customer ID: {customer_id}] "
            f"{user_message.strip()}"
        )

        result = str(orchestrator(enriched_message))

        return {
            "response": result,
            "session_id": session_id,
        }
    finally:
        memory_manager.close()


if __name__ == "__main__":
    app.run()
