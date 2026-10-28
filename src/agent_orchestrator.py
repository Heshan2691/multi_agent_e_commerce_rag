"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime, timezone
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────


def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Retrieves order and customer information from DynamoDB.
    It gathers facts and does not make refund eligibility decisions.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    system_prompt = """
    You are the Inventory Agent for NovaMart customer support.

    Your responsibilities are to retrieve accurate customer and order
    information using your available tools.

    Rules:
    - Use tools to retrieve facts instead of inventing information.
    - For an individual order, provide both customer_id and order_id.
    - Report missing customers or orders clearly.
    - If a requested order cannot be found for the supplied customer_id,
      report only that the order could not be verified for that customer.
    - Do not list other orders when a requested order is missing or mismatched.
    - Do not disclose customer names, profile details, or order counts unless
      explicitly required to answer the request.
    - For order-status requests, return only the requested status and
      essential supporting details.
    - Do not make refund eligibility decisions.
    - Do not initiate refunds or modify order records.
    - Return concise, factual, minimum-necessary results to the orchestrator.
    """

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Retrieve an individual order using its customer ID and order ID.

        Args:
            customer_id: The customer's unique identifier.
            order_id: The order's unique identifier.

        Returns:
            The order details, or an error if the order is not found.
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                }
            )

            order = response.get("Item")
            if not order:
                return {
                    "success": False,
                    "message": "Order not found for this customer.",
                    "customer_id": customer_id,
                    "order_id": order_id,
                }

            return {
                "success": True,
                "order": order,
            }

        except Exception as exc:
            logger.exception("Failed to retrieve order")
            return {
                "success": False,
                "message": "Unable to retrieve order information. Please try again later.",
            }

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier and account details.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            Customer profile including tier and account details.
        """
        try:
            table = dynamodb.Table(config.CUSTOMERS_TABLE)
            response = table.get_item(
                Key={"customer_id": customer_id}
            )

            customer = response.get("Item")
            if not customer:
                return {
                    "success": False,
                    "message": "Customer not found.",
                    "customer_id": customer_id,
                }

            # Return only fields required by downstream support decisions.
            return {
                "success": True,
                "customer": {
                    "customer_id": customer.get("customer_id", customer_id),
                    "tier": customer.get("tier"),
                },
            }

        except Exception as exc:
            logger.exception("Failed to retrieve customer")
            return {
                "success": False,
                "message": "Unable to retrieve customer information. Please try again later.",
            }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders belonging to a customer.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            The customer's orders and the total number of orders found.
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.query(
                KeyConditionExpression=Key("customer_id").eq(customer_id)
            )

            orders = response.get("Items", [])
            safe_orders = [
                {
                    key: order.get(key)
                    for key in (
                        "order_id", "status", "delivered_at", "delivery_date",
                        "estimated_delivery", "created_at", "total", "currency",
                    )
                    if key in order
                }
                for order in orders
            ]

            return {
                "success": True,
                "customer_id": customer_id,
                "count": len(safe_orders),
                "orders": safe_orders,
            }

        except Exception as exc:
            logger.exception("Failed to list customer orders")
            return {
                "success": False,
                "message": "Unable to retrieve customer orders. Please try again later.",
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )



# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────


def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Checks order status, customer tier, and the applicable return window
    before recording a return request in DynamoDB.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    system_prompt = """
    You are the Refund Agent for NovaMart customer support.

    Responsibilities:
    - Read order and customer information from WorkflowState.
    - Check the order status and applicable return window.
    - Standard customers have a 30-day return window.
    - Premium customers have a 60-day return window.
    - Use the initiate_refund tool only when the customer explicitly
      requests a return and the required facts have been verified.
    - Never invent order details, customer tiers, or return references.
    - Never claim money has been refunded merely because a return
      request was recorded.
    - If an order is missing, not delivered, outside the return window,
      or already has a return request, explain the reason.
    - Product condition and packaging requirements must still be met.
      Do not claim these requirements were verified if the data does
      not contain that information.
    - Do not invent a delivery date. If only an estimated delivery date
      is available, explain that it is being used as an estimate.
    """

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read WorkflowState to retrieve inventory-agent results.
        """
        try:
            state = _read_workflow_state(session_id)

            if state is None:
                return {
                    "success": False,
                    "message": "WorkflowState not found for this session.",
                }

            inventory_result = state.get("inventory_agent")

            if not inventory_result:
                return {
                    "success": False,
                    "message": (
                        "Inventory information is not available yet. "
                        "Run the Inventory Agent first."
                    ),
                }

            return {
                "success": True,
                "session_id": session_id,
                "customer_id": state.get("customer_id"),
                "inventory_agent": inventory_result,
            }

        except Exception:
            logger.exception("Failed to read refund inventory context")
            return {
                "success": False,
                "message": "Unable to retrieve inventory information.",
            }

    @tool
    def initiate_refund(
        customer_id: str,
        order_id: str,
        reason: str,
    ) -> dict:
        """
        Record a return request after checking order and customer data.

        This records a return request, not a completed monetary refund.
        """
        if not customer_id or not order_id or not reason.strip():
            return {
                "success": False,
                "message": (
                    "Customer ID, order ID, and a return reason are required."
                ),
            }

        try:
            orders_table = dynamodb.Table(config.ORDERS_TABLE)
            customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

            # Retrieve the order using its composite primary key.
            order_response = orders_table.get_item(
                Key={
                    "customer_id": customer_id,
                    "order_id": order_id,
                }
            )
            order = order_response.get("Item")

            if not order:
                return {
                    "success": False,
                    "message": "Order not found for this customer.",
                    "order_id": order_id,
                }

            # Only delivered orders can start a normal return.
            status = str(order.get("status", "")).lower()

            if status != "delivered":
                return {
                    "success": False,
                    "message": (
                        f"Cannot initiate a return for an order "
                        f"with status '{status or 'unknown'}'."
                    ),
                    "order_id": order_id,
                }

            # Retrieve the customer's actual tier.
            customer_response = customers_table.get_item(
                Key={"customer_id": customer_id}
            )
            customer = customer_response.get("Item")

            if not customer:
                return {
                    "success": False,
                    "message": "Customer profile not found.",
                    "customer_id": customer_id,
                }

            tier = str(customer.get("tier", "")).strip().lower()

            if tier == "premium":
                return_window_days = 60
            elif tier == "standard":
                return_window_days = 30
            else:
                return {
                    "success": False,
                    "message": (
                        "Customer tier is missing or unrecognized. "
                        "Return eligibility cannot be verified."
                    ),
                }

            # Return eligibility must use a verified actual delivery date.
            # An estimate must never be treated as proof that delivery occurred.
            delivery_date_value = order.get("delivered_at") or order.get("delivery_date")

            if not delivery_date_value:
                return {
                    "success": False,
                    "message": (
                        "A verified actual delivery date is unavailable. "
                        "Return eligibility cannot be verified; please verify "
                        "delivery before submitting a return request."
                    ),
                }

            try:
                delivery_date = datetime.strptime(
                    str(delivery_date_value)[:10],
                    "%Y-%m-%d",
                ).date()
            except ValueError:
                return {
                    "success": False,
                    "message": "The recorded delivery date is invalid.",
                }

            today = datetime.now(timezone.utc).date()
            days_since_delivery = (today - delivery_date).days

            if days_since_delivery < 0:
                return {
                    "success": False,
                    "message": (
                        "The recorded delivery date is in the future. "
                        "Return eligibility needs verification."
                    ),
                }

            if days_since_delivery > return_window_days:
                return {
                    "success": False,
                    "message": (
                        f"The order is outside the {return_window_days}-day "
                        f"return window for a {tier.title()} customer."
                    ),
                    "days_since_delivery": days_since_delivery,
                    "return_window_days": return_window_days,
                }

            # A return reference is created only after basic eligibility
            # checks pass. The conditional update prevents duplicate
            # requests and protects against a concurrent status change.
            return_reference = (
                f"RET-{uuid.uuid4().hex[:10].upper()}"
            )
            requested_at = datetime.now(timezone.utc).isoformat()

            try:
                orders_table.update_item(
                    Key={
                        "customer_id": customer_id,
                        "order_id": order_id,
                    },
                    UpdateExpression=(
                        "SET #status = :new_status, "
                        "return_reference = :reference, "
                        "return_reason = :reason, "
                        "return_requested_at = :requested_at"
                    ),
                    ConditionExpression=(
                        "#status = :delivered "
                        "AND attribute_not_exists(return_reference)"
                    ),
                    ExpressionAttributeNames={
                        "#status": "status",
                    },
                    ExpressionAttributeValues={
                        ":new_status": "return_requested",
                        ":delivered": "delivered",
                        ":reference": return_reference,
                        ":reason": reason.strip(),
                        ":requested_at": requested_at,
                    },
                )

            except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                return {
                    "success": False,
                    "message": (
                        "The order status has changed or a return "
                        "request already exists. Please check the order."
                    ),
                    "order_id": order_id,
                }

            return {
                "success": True,
                "message": "Return request recorded successfully.",
                "customer_id": customer_id,
                "order_id": order_id,
                "return_reference": return_reference,
                "return_status": "return_requested",
                "return_window_days": return_window_days,
                "days_since_delivery": days_since_delivery,
                "instructions": (
                    "Follow the return instructions provided by "
                    "customer support. A return request does not mean "
                    "the monetary refund has been completed."
                ),
            }

        except Exception:
            logger.exception("Failed to initiate return request")
            return {
                "success": False,
                "message": (
                    "Unable to record the return request. "
                    "Please try again or contact customer support."
                ),
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            get_inventory_context,
            initiate_refund,
        ],
    )



# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────


def build_policy_agent() -> Agent:
    """
    Build the Policy Agent with three specialized policy retrievers
    and a coordinator that searches them in parallel.
    """

    worker_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    # Returns policy retriever
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        if not config.RETURNS_KB_ID:
            return "Returns policy knowledge base is not configured."

        results = retrieve_from_knowledge_base(
            config.RETURNS_KB_ID, query
        )
        return format_kb_results(results)

    returns_retriever = Agent(
        model=worker_model,
        system_prompt=(
            "You retrieve information about product returns and refunds. "
            "Use the retrieval tool and report only information supported "
            "by the retrieved policy documents. Do not invent policy rules."
        ),
        tools=[retrieve_returns_policy],
    )

    # Shipping policy retriever
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        if not config.SHIPPING_KB_ID:
            return "Shipping policy knowledge base is not configured."

        results = retrieve_from_knowledge_base(
            config.SHIPPING_KB_ID, query
        )
        return format_kb_results(results)

    shipping_retriever = Agent(
        model=worker_model,
        system_prompt=(
            "You retrieve information about shipping and delivery. "
            "Use the retrieval tool and report only information supported "
            "by the retrieved policy documents. Do not invent policy rules."
        ),
        tools=[retrieve_shipping_policy],
    )

    # Warranty policy retriever
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        if not config.WARRANTY_KB_ID:
            return "Warranty policy knowledge base is not configured."

        results = retrieve_from_knowledge_base(
            config.WARRANTY_KB_ID, query
        )
        return format_kb_results(results)

    warranty_retriever = Agent(
        model=worker_model,
        system_prompt=(
            "You retrieve information about product warranties. "
            "Use the retrieval tool and report only information supported "
            "by the retrieved policy documents. Do not invent policy rules."
        ),
        tools=[retrieve_warranty_policy],
    )

    # Coordinator tool: query all policy retrievers concurrently
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query the Returns, Shipping, and Warranty policy retrievers
        in parallel and combine their findings.
        """
        knowledge_bases = {
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        }

        # Avoid unnecessary model calls when no knowledge bases are configured.
        if not any(knowledge_bases.values()):
            return (
                "Policy knowledge bases are not configured yet. "
                "No verified policy information is available."
            )

        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }

        results = {}

        def run_retriever(domain: str, agent, question: str):
            try:
                response = agent(question)
                return domain, str(response)
            except Exception:
                logger.exception("%s policy retrieval failed", domain)
                return domain, f"{domain} policy retrieval failed."

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_retriever, domain, agent, query)
                for domain, agent in retrievers.items()
                if knowledge_bases[domain]
            ]

            for future in as_completed(futures):
                domain, response = future.result()
                results[domain] = response

        # Keep the output order consistent
        sections = []
        for domain in ("Returns", "Shipping", "Warranty"):
            if domain in results:
                sections.append(f"--- {domain} Policy ---\n{results[domain]}")
            elif not knowledge_bases[domain]:
                sections.append(
                    f"--- {domain} Policy ---\nKnowledge base not configured."
                )

        return "\n\n".join(sections)

    # Policy coordinator
    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    system_prompt = """
You are the Policy Agent for a customer support system.

Use search_all_policies to investigate customer questions about returns,
shipping, delivery, and warranties.

Base answers only on retrieved policy information. Never invent policy
rules, deadlines, eligibility conditions, or exceptions.

If policy knowledge bases are unavailable or the retrieved information
does not answer the question, clearly explain that the policy cannot
currently be verified. Distinguish confirmed information from uncertainty.
"""

    return Agent(
        model=coordinator_model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )



# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────


def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Reads WorkflowState and drafts the final customer-facing response.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    system_prompt = """
    You are the Communication Agent for NovaMart customer support.

    Your job is to write a clear, friendly, professional response to
    the customer using the information recorded by the other agents.

    Rules:
    - Use get_full_workflow_context to retrieve the current session's results.
    - Use only facts supported by the workflow context.
    - Acknowledge the customer's request and explain the outcome clearly.
    - Never invent order details, policies, refund approvals, or reference numbers.
    - Never claim a refund or return was completed unless the recorded
      results confirm that it was.
    - If information is missing, explain that clearly and do not guess.
    - Do not expose internal system details or unnecessary technical data.
    - Return the final customer-facing response, not an internal analysis.
    """

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Retrieve all recorded results for the current customer session.
        """
        try:
            state = _read_workflow_state(session_id)

            if state is None:
                return {
                    "success": False,
                    "message": "WorkflowState not found for this session.",
                    "session_id": session_id,
                }

            return {
                "success": True,
                "workflow_state": state,
            }

        except Exception:
            logger.exception("Failed to read WorkflowState")
            return {
                "success": False,
                "message": "Unable to retrieve workflow information.",
            }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )



# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────


def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent,
    session_manager=None,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    orchestrator_model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        guardrail_id=config.GUARDRAIL_ID,
        guardrail_version=str(config.GUARDRAIL_VERSION),
        guardrail_trace="enabled",
    )

    system_prompt = """
    You are the Orchestrator Agent for a customer support system.

    Your responsibility is to coordinate specialized agents and ensure
    that their results are stored in WorkflowState.

    Follow these rules:
    1. Always call initialize_session first for each customer request.
    2. For order status, order history, and return requests, call
       route_to_inventory_agent first to gather order facts.
    3. For questions about returns, shipping, or warranties, call
       route_to_policy_agent to obtain verified policy information.
    4. For refund or return requests, call route_to_refund_agent only
       after inventory information has been gathered.
    5. Call route_to_communication_agent last, after all relevant
       agents have completed their work.
    6. Use the session_id and customer_id supplied with the request.
    7. Never invent order details, policies, or refund decisions.
    8. Provide the final response returned by the Communication Agent.
    """

    def save_agent_result(session_id: str, field: str, result: str) -> None:
        """Save an agent result to WorkflowState."""
        state = _read_workflow_state(session_id)

        if state is None:
            raise ValueError(
                f"WorkflowState not found for session {session_id}. "
                "Initialize the session first."
            )

        _update_workflow_state(
            session_id,
            {field: result},
            state.get("version", 0),
        )

    @tool
    def route_to_inventory_agent(
        session_id: str,
        customer_id: str,
        request: str,
    ) -> str:
        """Gather order and customer facts before making return decisions."""
        try:
            prompt = (
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n\n"
                "Retrieve the relevant order and customer facts. "
                "Report the verified information without making "
                "refund eligibility decisions."
            )

            result = str(inventory_agent(prompt))
            save_agent_result(session_id, "inventory_agent", result)
            return result

        except Exception as exc:
            logger.exception("Inventory Agent routing failed")
            return f"Unable to retrieve inventory information: {exc}"

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """Retrieve relevant returns, shipping, or warranty policy information."""
        try:
            prompt = (
                f"Session ID: {session_id}\n"
                f"Policy question: {request}\n\n"
                "Retrieve relevant policy information. Clearly distinguish "
                "verified policy facts from unavailable information."
            )

            result = str(policy_agent(prompt))
            save_agent_result(session_id, "policy_agent", result)
            return result

        except Exception as exc:
            logger.exception("Policy Agent routing failed")
            return f"Unable to retrieve policy information: {exc}"

    @tool
    def route_to_refund_agent(
        session_id: str,
        customer_id: str,
        request: str,
    ) -> str:
        """Evaluate a return or refund request using gathered order facts."""
        try:
            state = _read_workflow_state(session_id)

            if not state or not state.get("inventory_agent"):
                return (
                    "Refund processing requires inventory information first. "
                    "Call route_to_inventory_agent before this tool."
                )

            prompt = (
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Customer request: {request}\n"
                f"Inventory findings: {state['inventory_agent']}\n\n"
                "Evaluate the request using verified order facts and "
                "applicable policy information. Do not invent eligibility "
                "rules or claim a refund was initiated unless confirmed."
            )

            if state.get("policy_agent"):
                prompt += (
                    f"\nPolicy findings: {state['policy_agent']}\n"
                )

            result = str(refund_agent(prompt))
            save_agent_result(session_id, "refund_agent", result)
            return result

        except Exception as exc:
            logger.exception("Refund Agent routing failed")
            return f"Unable to process the refund request: {exc}"

    @tool
    def route_to_communication_agent(
        session_id: str,
        customer_id: str,
        original_request: str,
    ) -> str:
        """Compose the final customer-facing response using workflow results."""
        try:
            state = _read_workflow_state(session_id)

            if not state:
                return (
                    "Cannot prepare the final response because the session "
                    "has not been initialized."
                )

            communication_context = {
                "session_id": session_id,
                "customer_id": customer_id,
                "inventory_findings": state.get("inventory_agent"),
                "policy_findings": state.get("policy_agent"),
                "refund_findings": state.get("refund_agent"),
            }

            prompt = (
                f"Session ID: {session_id}\n"
                f"Customer ID: {customer_id}\n"
                f"Original customer request: {original_request}\n"
                f"Relevant workflow findings: "
                f"{json.dumps(communication_context, default=str)}\n\n"
                "Compose a clear, friendly response based only on the "
                "relevant recorded findings. Apply data minimization: "
                "disclose only information needed to answer the customer's "
                "request. Never list unrelated orders, customer profile "
                "details, or total order counts. If an order cannot be "
                "verified for this customer, state that it could not be "
                "verified and do not disclose other order information. "
                "Do not claim an action was completed unless the workflow "
                "results confirm it."
            )

            result = str(communication_agent(prompt))
            save_agent_result(session_id, "communication_agent", result)
            return result

        except Exception as exc:
            logger.exception("Communication Agent routing failed")
            return f"Unable to prepare the final response: {exc}"

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """Create the initial WorkflowState record for a customer session."""
        try:
            existing_state = _read_workflow_state(session_id)

            if existing_state:
                if existing_state.get("customer_id") != customer_id:
                    return (
                        "Session initialization failed: the session ID "
                        "belongs to a different customer."
                    )

                return f"Session {session_id} is already initialized."

            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized successfully."

        except Exception as exc:
            logger.exception("Session initialization failed")
            return f"Unable to initialize the session: {exc}"

    return Agent(
    model=orchestrator_model,
    system_prompt=system_prompt,
    tools=[
        initialize_session,
        route_to_inventory_agent,
        route_to_policy_agent,
        route_to_refund_agent,
        route_to_communication_agent,
    ],
    session_manager=session_manager,
)



# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create and publish the NovaMart Bedrock Guardrail.
    Returns (guardrail_id, numbered_guardrail_version).
    """
    bedrock_client = boto3.client("bedrock", region_name=config.AWS_REGION)

    # Reuse the existing guardrail if it has already been created.
    response = bedrock_client.list_guardrails()
    existing = next(
        (
            item for item in response.get("guardrails", [])
            if item.get("name") == config.GUARDRAIL_NAME
        ),
        None,
    )

    if existing:
        guardrail_id = existing["id"]
        existing_version = existing.get("version", "DRAFT")

        if existing_version != "DRAFT":
            print(
                f"Guardrail already exists: {guardrail_id} "
                f"(version: {existing_version})"
            )
            return guardrail_id, existing_version

        print(f"Found existing draft guardrail: {guardrail_id}")

    else:
        guardrail_response = bedrock_client.create_guardrail(
            name=config.GUARDRAIL_NAME,
            description=(
                "NovaMart customer-support safety policy. "
                "Protects personal information and blocks harmful "
                "or unrelated customer-support requests."
            ),
            blockedInputMessaging=(
                "I'm sorry, but I can't help with that request. "
                "I can help with NovaMart orders, returns, shipping, "
                "warranty policies, and general calculations."
            ),
            blockedOutputsMessaging=(
                "I'm sorry, but I can't provide that response. "
                "I can help with NovaMart customer-support questions."
            ),
            crossRegionConfig={
                "guardrailProfileIdentifier": "us.guardrail.v1:0"
            },
            contentPolicyConfig={
                "filtersConfig": [
                    {
                        "type": "SEXUAL",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                    {
                        "type": "VIOLENCE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                    {
                        "type": "HATE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                    {
                        "type": "INSULTS",
                        "inputStrength": "MEDIUM",
                        "outputStrength": "MEDIUM",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                    {
                        "type": "MISCONDUCT",
                        "inputStrength": "MEDIUM",
                        "outputStrength": "MEDIUM",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                ],
                "tierConfig": {"tierName": "STANDARD"},
            },
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {
                        "type": "CREDIT_DEBIT_CARD_NUMBER",
                        "action": "BLOCK",
                    },
                    {
                        "type": "US_SOCIAL_SECURITY_NUMBER",
                        "action": "BLOCK",
                    },
                    {
                        "type": "EMAIL",
                        "action": "ANONYMIZE",
                    },
                    {
                        "type": "PHONE",
                        "action": "ANONYMIZE",
                    },
                ],
            },
            topicPolicyConfig={
                "topicsConfig": [
                    {
                        "name": "competitor_products",
                        "definition": (
                            "Requests to recommend, compare, or provide "
                            "product-specific information about competing "
                            "retailers or their products instead of NovaMart. "
                            "Ordinary NovaMart product and order questions "
                            "are not in this category."
                        ),
                        "examples": [
                            "Tell me which competing retailer has better products.",
                            "Recommend products from another retailer.",
                        ],
                        "type": "DENY",
                    },
                    {
                        "name": "pricing_negotiations",
                        "definition": (
                            "Requests to bargain with NovaMart, change an "
                            "advertised price, demand a special unapproved "
                            "price, or negotiate a price exception. Do not "
                            "classify ordinary arithmetic using a supplied "
                            "price, quantity, or discount as negotiation."
                        ),
                        "examples": [
                            "Negotiate a lower advertised price for me.",
                            "Change the listed price to $10 just for me.",
                            "Give me an unapproved special price.",
                        ],
                        "type": "DENY",
                    },
                    {
                        "name": "legal_threats",
                        "definition": (
                            "Requests centered on making or escalating legal "
                            "threats against NovaMart, such as threatening "
                            "litigation to force a particular outcome. "
                            "Ordinary requests for return or warranty "
                            "information are not legal threats."
                        ),
                        "examples": [
                            "Write a legal threat to force NovaMart to refund me.",
                            "Help me threaten NovaMart with a lawsuit.",
                        ],
                        "type": "DENY",
                    },
                ],
                "tierConfig": {"tierName": "STANDARD"},
            },
            wordPolicyConfig={
                "managedWordListsConfig": [
                    {
                        "type": "PROFANITY",
                    },
                ],
            },
        )

        guardrail_id = guardrail_response["guardrailId"]
        print(f"Created guardrail draft: {guardrail_id}")

    # Publish the draft so runtime deployment can use a numbered version.
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Published NovaMart customer-support guardrail",
    )
    guardrail_version = version_response.get("version")

    if not guardrail_version or guardrail_version == "DRAFT":
        raise RuntimeError(
            "Bedrock did not return a numbered guardrail version. "
            f"Response: {version_response}"
        )

    print(
        f"Guardrail published successfully: {guardrail_id} "
        f"(version: {guardrail_version})"
    )
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """Return an existing AgentCore Runtime ARN, or fail clearly.

    This project does not yet package the complete application into a valid
    AgentCore Runtime artifact. Deliberately do not upload a placeholder ZIP
    or pretend a runtime was deployed. Existing runtimes can still be reused.
    """
    runtime_name = f"{config.PROJECT_NAME}-runtime".replace('-', '_')

    try:
        response = agentcore_control.list_agent_runtimes()
    except Exception as exc:
        logger.exception("Unable to list AgentCore runtimes")
        raise RuntimeError(
            "Could not check for an existing AgentCore Runtime. Verify AWS "
            "credentials, region, and bedrock-agentcore permissions."
        ) from exc

    for runtime in response.get('agentRuntimes', []):
        if runtime.get('agentRuntimeName') == runtime_name:
            runtime_arn = runtime.get('agentRuntimeArn')
            if runtime_arn:
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn

    raise RuntimeError(
        f"No existing AgentCore Runtime named '{runtime_name}' was found. "
        "Automatic creation is not implemented: the current source only "
        "builds a placeholder main.py, which cannot run the multi-agent app. "
        "Deploy a complete runtime artifact and configure its entry point "
        "before retrying. No placeholder artifact was uploaded."
    )


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        clientToken=str(uuid.uuid4()),
        name=memory_name,
        description="Session-scoped conversational memory for NovaMart.",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "session-summary",
                    "description": "Summarize customer-support conversations.",
                }
            }
        ],
    )

    # SDK/API response shapes can differ by version. Accept only an ARN
    # explicitly returned by the service; never fabricate one.
    memory = response.get("memory") or response.get("Memory") or response
    memory_arn = memory.get("arn") or memory.get("memoryArn") or memory.get("memoryARN")

    if not memory_arn:
        raise RuntimeError(
            f"AgentCore Memory creation returned no ARN: {response}"
        )

    print(f"AgentCore Memory created: {memory_arn}")
    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    runtime_id = runtime_arn.split('/')[-1]

    # TODO: Configure observability
    # NOTE: AgentCore API — control plane logging.
    # put_agent_runtime_logging_configuration may not be available in all
    # SDK versions — wrap the call in try/except and fall back gracefully.
    # Use agentcore_control.put_agent_runtime_logging_configuration() with:
    #   - agentRuntimeId (runtime_id)
    #   - loggingConfiguration containing:
    #     - cloudWatchConfig (logGroupName: config.AGENT_LOG_GROUP, logLevel: INFO, enabled: True)
    #     - xRayConfig (enabled: True, samplingRate: 1.0)
    # On success: print the CloudWatch log group and X-Ray sampling rate.
    # On exception: print "[Note] Logging config skipped (SDK version mismatch): <e>"

    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,
        },
    }

    try:
        agentcore_control.put_agent_runtime_logging_configuration(
            agentRuntimeId=runtime_id,
            loggingConfiguration=logging_configuration,
        )
        print(f"  CloudWatch Logs: {config.AGENT_LOG_GROUP} (INFO)")
        print(f"  X-Ray sampling:  {logging_configuration['xRayConfig']['samplingRate']:.0%}")
    except Exception as e:
        print(f"  [Note] Logging config skipped (SDK version mismatch): {e}")

    # AgentCore Observability is backed by CloudWatch Transaction Search:
    # make sure the log group exists and X-Ray indexes 100% of spans.
    try:
        logs_client.create_log_group(logGroupName=config.AGENT_LOG_GROUP)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass
    except Exception as e:
        print(f"  [Note] Could not create log group: {e}")

    try:
        xray_client = boto3.client('xray', region_name=config.AWS_REGION)
        if xray_client.get_trace_segment_destination().get('Destination') != 'CloudWatchLogs':
            xray_client.update_trace_segment_destination(Destination='CloudWatchLogs')
        sampling_percentage = logging_configuration['xRayConfig']['samplingRate'] * 100
        xray_client.update_indexing_rule(
            Name='Default',
            Rule={'Probabilistic': {'DesiredSamplingPercentage': sampling_percentage}},
        )
        print(f"  Transaction Search: enabled, indexing {sampling_percentage:.0f}% of spans")
    except Exception as e:
        print(f"  [Note] Transaction Search not updated: {e}")



# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    failures = []
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as exc:
            logger.exception("Gateway target registration failed: %s", t['name'])
            failures.append(t['name'])
            print(f"    [Failed] {t['name']}: check configuration and logs")

    status = 'TARGETS_FAILED' if failures else 'TARGETS_SUBMITTED'
    return {
        'gateway_id': gateway_id,
        'gateway_url': gateway_url,
        'status': status,
        'failed_targets': failures,
        'target_count': len(targets),
        'registered_target_count': len(targets) - len(failures),
    }


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
