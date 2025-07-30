#!/usr/bin/env python3
"""
Main API server for the ToM Agent

This module provides a RESTful API for personalized instruction improvement
and next action suggestions for SWE agents.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tom_swe.api.models import (
    ConversationStatusResponse,
    ErrorResponse,
    HealthResponse,
    ProposeInstructionsRequest,
    ProposeInstructionsResponse,
)
from tom_swe.tom_agent import ToMAgent, create_tom_agent

# Configure logging - use environment variable for level
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Ensure all tom_swe loggers use the same level
for logger_name in [
    "tom_swe.generation_utils.generate",
    "tom_swe.tom_agent",
    "tom_swe.rag_module",
]:
    logging.getLogger(logger_name).setLevel(getattr(logging, log_level, logging.INFO))


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


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        tom_agent_ready=hasattr(app.state, "tom_agent")
        and app.state.tom_agent is not None,
    )


@app.post("/propose_instructions", response_model=ProposeInstructionsResponse)
async def propose_instructions(
    request: ProposeInstructionsRequest,
) -> ProposeInstructionsResponse:
    """
    Get improved, personalized instructions based on user context and preferences.
    """
    if not hasattr(app.state, "tom_agent") or app.state.tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )

    try:
        logger.info(f"Generating instruction improvements for user {request.user_id}")

        # Generate improved instructions
        recommendation = app.state.tom_agent.propose_instructions(
            user_id=request.user_id,
            original_instruction=request.original_instruction,
            user_msg_context=request.context,
        )

        return ProposeInstructionsResponse(
            user_id=request.user_id,
            original_instruction=request.original_instruction,
            recommendations=[recommendation],
            success=True,
            message="Generated instruction recommendation",
        )

    except Exception as e:
        logger.error(f"Error generating instructions for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate instructions: {e!s}",
        ) from e


@app.get("/conversation_status", response_model=ConversationStatusResponse)
async def get_conversation_status(user_id: str) -> ConversationStatusResponse:
    """Get the current conversation status for a user."""
    try:
        return ConversationStatusResponse(
            user_id=user_id,
            success=True,
            message=f"API is ready to serve requests for user {user_id}",
        )

    except Exception as e:
        logger.error(f"Error getting conversation status for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation status: {e!s}",
        ) from e


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
