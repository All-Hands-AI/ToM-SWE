#!/usr/bin/env python3
"""
Main API server for the ToM Agent

This module provides a RESTful API for bidirectional communication between
the ToM Agent and SWE agents, supporting both instruction improvement and
next action suggestions.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tom_swe.api.models import (
    ConversationState,
    ConversationStatusResponse,
    ErrorResponse,
    HealthResponse,
    MessageType,
    ProposeInstructionsResponse,
    SendMessageRequest,
    SendMessageResponse,
    SuggestNextActionsResponse,
)
from tom_swe.tom_agent import ToMAgent, create_tom_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
conversation_states: Dict[str, ConversationState] = {}


async def initialize_tom_agent() -> ToMAgent | None:
    """Initialize the ToM agent."""
    processed_data_dir = os.getenv("TOM_PROCESSED_DATA_DIR", "./data/processed_data")
    user_model_dir = os.getenv("TOM_USER_MODEL_DIR", "./data/user_model")
    enable_rag = os.getenv("TOM_ENABLE_RAG", "true").lower() in ("true", "1", "yes")

    try:
        agent = await create_tom_agent(
            processed_data_dir=processed_data_dir,
            user_model_dir=user_model_dir,
            enable_rag=enable_rag,
        )
        logger.info("ToM Agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize ToM Agent: {e}")
        # Continue startup even if ToM agent fails to initialize
        # This allows the health endpoint to report the issue
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting ToM Agent API server...")
    app.state.tom_agent = await initialize_tom_agent()

    yield

    # Shutdown
    logger.info("Shutting down ToM Agent API server...")
    conversation_states.clear()


# Create FastAPI app
app = FastAPI(
    title="ToM Agent API",
    description="RESTful API for bidirectional communication with SWE Agent",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG") else None,
        ).model_dump(),
    )


def get_or_create_conversation_state(user_id: str) -> ConversationState:
    """Get existing conversation state or create a new one."""
    if user_id not in conversation_states:
        conversation_states[user_id] = ConversationState(
            user_id=user_id,
            last_updated=datetime.now(),
        )
    return conversation_states[user_id]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        tom_agent_ready=hasattr(app.state, "tom_agent") and app.state.tom_agent is not None,
        active_conversations=len(conversation_states),
    )


@app.post("/send_msg", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest) -> SendMessageResponse:
    """
    Send a message to the ToM agent and trigger appropriate processing.

    Flow:
    - User message (message_type="user_message") → Triggers instruction improvement calculation
    - Agent response (message_type="agent_response") → Triggers next actions calculation

    The actual results are cached and retrieved via GET endpoints.
    """
    if not hasattr(app.state, "tom_agent") or app.state.tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )

    try:
        logger.info(f"Received {request.message_type} from user {request.user_id}")

        # Get or create conversation state
        conv_state = get_or_create_conversation_state(request.user_id)
        conv_state.last_updated = datetime.now()

        processing_type = ""

        if request.message_type == MessageType.USER_MESSAGE:
            # User sent a message → Calculate improved instructions
            logger.info("Processing user message for instruction improvement")

            # Update conversation state
            conv_state.last_user_message = request.message
            if request.original_instruction:
                conv_state.last_user_instruction = request.original_instruction

            # Analyze user context and generate improved instructions
            user_context = await app.state.tom_agent.analyze_user_context(
                user_id=request.user_id,
                current_query=request.original_instruction or request.message,
            )

            recommendations = await app.state.tom_agent.propose_instructions(
                user_context=user_context,
                original_instruction=request.original_instruction or request.message,
                domain_context=request.domain_context,
            )

            # Cache the results
            conv_state.pending_instructions = recommendations
            conv_state.instructions_calculated_at = datetime.now()
            processing_type = "instructions"

        elif request.message_type == MessageType.AGENT_RESPONSE:
            # Agent sent a response → Calculate next actions
            logger.info("Processing agent response for next actions")

            # Update conversation state with agent actions
            conv_state.last_agent_actions.append(request.message)
            # Keep only last 5 agent actions
            conv_state.last_agent_actions = conv_state.last_agent_actions[-5:]

            # Analyze user context and generate next actions
            user_context = await app.state.tom_agent.analyze_user_context(
                user_id=request.user_id,
                current_query=request.task_context,
            )

            suggestions = await app.state.tom_agent.suggest_next_actions(
                user_context=user_context,
                current_task_context=request.task_context,
            )

            # Cache the results
            conv_state.pending_next_actions = suggestions
            conv_state.next_actions_calculated_at = datetime.now()
            processing_type = "next_actions"

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid message_type: {request.message_type}. Must be 'user_message' or 'agent_response'",
            )

        return SendMessageResponse(
            user_id=request.user_id,
            message_received=True,
            processing_type=processing_type,
            success=True,
            message=f"Message processed successfully. Use GET endpoint to retrieve {processing_type}.",
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error processing message for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {e!s}",
        ) from e


@app.get("/propose_instructions", response_model=ProposeInstructionsResponse)
async def get_propose_instructions(user_id: str) -> ProposeInstructionsResponse:
    """
    Get cached instruction improvements for a user.

    Returns the instruction recommendations that were calculated when the user
    sent their message via POST /send_msg with message_type="user_message".
    """
    if not hasattr(app.state, "tom_agent") or app.state.tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )

    try:
        logger.info(f"Retrieving instruction recommendations for user {user_id}")

        conv_state = conversation_states.get(user_id)
        if not conv_state or not conv_state.pending_instructions:
            return ProposeInstructionsResponse(
                user_id=user_id,
                original_instruction=None,
                recommendations=[],
                calculated_at=datetime.now(),
                success=False,
                message="No pending instruction improvements found. Send a user message first via POST /send_msg.",
            )

        # Return cached recommendations
        return ProposeInstructionsResponse(
            user_id=user_id,
            original_instruction=conv_state.last_user_instruction,
            recommendations=conv_state.pending_instructions,
            calculated_at=conv_state.instructions_calculated_at or datetime.now(),
            success=True,
            message=f"Retrieved {len(conv_state.pending_instructions)} instruction recommendations",
        )

    except Exception as e:
        logger.error(f"Error retrieving instructions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve instructions: {e!s}",
        ) from e


@app.get("/suggest_next_actions", response_model=SuggestNextActionsResponse)
async def get_suggest_next_actions(user_id: str) -> SuggestNextActionsResponse:
    """
    Get cached next action suggestions for a user.

    Returns the action suggestions that were calculated when the SWE agent
    sent their response via POST /send_msg with message_type="agent_response".
    """
    if not hasattr(app.state, "tom_agent") or app.state.tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )

    try:
        logger.info(f"Retrieving next action suggestions for user {user_id}")

        conv_state = conversation_states.get(user_id)
        if not conv_state or not conv_state.pending_next_actions:
            return SuggestNextActionsResponse(
                user_id=user_id,
                suggestions=[],
                based_on_context="No agent responses processed yet",
                calculated_at=datetime.now(),
                success=False,
                message="No pending next action suggestions found. Send an agent response first via POST /send_msg.",
            )

        # Describe what the suggestions are based on
        context_description = (
            f"Recent agent actions: {', '.join(conv_state.last_agent_actions[-3:])}"
        )

        return SuggestNextActionsResponse(
            user_id=user_id,
            suggestions=conv_state.pending_next_actions,
            based_on_context=context_description,
            calculated_at=conv_state.next_actions_calculated_at or datetime.now(),
            success=True,
            message=f"Retrieved {len(conv_state.pending_next_actions)} next action suggestions",
        )

    except Exception as e:
        logger.error(f"Error retrieving next actions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve next actions: {e!s}",
        ) from e


@app.get("/conversation_status", response_model=ConversationStatusResponse)
async def get_conversation_status(user_id: str) -> ConversationStatusResponse:
    """Get the current conversation status for a user."""
    try:
        conv_state = conversation_states.get(user_id)

        if not conv_state:
            return ConversationStatusResponse(
                user_id=user_id,
                has_pending_instructions=False,
                has_pending_next_actions=False,
                last_activity=None,
                success=True,
            )

        return ConversationStatusResponse(
            user_id=user_id,
            has_pending_instructions=conv_state.pending_instructions is not None,
            has_pending_next_actions=conv_state.pending_next_actions is not None,
            last_activity=conv_state.last_updated,
            success=True,
        )

    except Exception as e:
        logger.error(f"Error getting conversation status for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation status: {e!s}",
        ) from e


# Utility endpoints for debugging and monitoring


@app.get("/active_conversations")
async def get_active_conversations() -> Dict[str, Any]:
    """Get information about all active conversations (debug endpoint)."""
    return {
        "total_conversations": len(conversation_states),
        "user_ids": list(conversation_states.keys()),
        "conversation_summary": {
            user_id: {
                "last_updated": conv.last_updated.isoformat(),
                "has_pending_instructions": conv.pending_instructions is not None,
                "has_pending_next_actions": conv.pending_next_actions is not None,
                "agent_actions_count": len(conv.last_agent_actions),
            }
            for user_id, conv in conversation_states.items()
        },
    }


@app.delete("/conversation/{user_id}")
async def clear_conversation_state(user_id: str) -> Dict[str, str]:
    """Clear conversation state for a specific user (debug endpoint)."""
    if user_id in conversation_states:
        del conversation_states[user_id]
        return {"message": f"Conversation state cleared for user {user_id}"}
    else:
        return {"message": f"No conversation state found for user {user_id}"}


if __name__ == "__main__":
    import uvicorn

    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    logger.info(f"Starting ToM Agent API server on {host}:{port}")
    uvicorn.run(
        "tom_swe.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
