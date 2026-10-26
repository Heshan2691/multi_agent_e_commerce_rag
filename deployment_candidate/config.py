import os

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "udacity-agentcore")
ACCOUNT_ID = os.environ.get("ACCOUNT_ID", "024278212493")

ORCHESTRATOR_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
WORKER_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

ORDERS_TABLE = os.environ.get("ORDERS_TABLE", "udacity-agentcore-orders")
CUSTOMERS_TABLE = os.environ.get("CUSTOMERS_TABLE", "udacity-agentcore-customers")
WORKFLOW_STATE_TABLE = os.environ.get(
    "WORKFLOW_STATE_TABLE", "udacity-agentcore-workflow-state"
)

POLICY_BUCKET = os.environ.get("POLICY_BUCKET", "")
VECTOR_STORE_BUCKET = os.environ.get("VECTOR_STORE_BUCKET", "")
AGENTCORE_ROLE_ARN = os.environ.get("AGENTCORE_ROLE_ARN", "")
AGENT_LOG_GROUP = os.environ.get(
    "AGENT_LOG_GROUP", "/aws/bedrock/agentcore/udacity-agentcore"
)

RETURNS_KB_ID = os.environ.get("RETURNS_KB_ID", "")
SHIPPING_KB_ID = os.environ.get("SHIPPING_KB_ID", "")
WARRANTY_KB_ID = os.environ.get("WARRANTY_KB_ID", "")

AGENTCORE_RUNTIME_ARN = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")
GUARDRAIL_NAME = f"{PROJECT_NAME}-guardrail"

GUARDRAIL_BLOCKED_TOPICS = [
    "competitor products",
    "pricing negotiations",
    "legal threats",
]

MEMORY_NAMESPACE = f"{PROJECT_NAME}-memory"
