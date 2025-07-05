"""
FastAPI application for the ToM Agent REST API.

This module provides a RESTful API server for the Theory of Mind (ToM) agent,
offering endpoints for instruction improvement, next action suggestions, and
comprehensive personalized guidance.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..tom_agent import ToMAgent, create_tom_agent
from .models import (
    ErrorResponse,
    HealthResponse,
    ProposeInstructionsRequest,
    ProposeInstructionsResponse,
    SendMessageRequest,
    SendMessageResponse,
    SuggestNextActionsRequest,
    SuggestNextActionsResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ToM agent instance
tom_agent: ToMAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI app."""
    global tom_agent
    
    # Startup
    logger.info("Starting ToM Agent API server...")
    
    # Initialize ToM agent
    processed_data_dir = os.getenv("TOM_PROCESSED_DATA_DIR", "./data/processed_data")
    user_model_dir = os.getenv("TOM_USER_MODEL_DIR", "./data/user_model")
    
    try:
        tom_agent = await create_tom_agent(
            processed_data_dir=processed_data_dir,
            user_model_dir=user_model_dir,
        )
        logger.info("ToM Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ToM Agent: {e}")
        # Continue startup even if ToM agent fails to initialize
        # This allows the health endpoint to report the issue
    
    yield
    
    # Shutdown
    logger.info("Shutting down ToM Agent API server...")


# Create FastAPI app
app = FastAPI(
    title="ToM Agent API",
    description="RESTful API for the Theory of Mind (ToM) Agent",
    version="1.0.0",
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
async def global_exception_handler(request, exc):
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
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        tom_agent_ready=tom_agent is not None,
    )


@app.post("/api/v1/suggest_next_actions", response_model=SuggestNextActionsResponse)
async def suggest_next_actions(request: SuggestNextActionsRequest):
    """
    Suggest next actions for a user based on their context and current task.
    
    This endpoint analyzes the user's mental state, preferences, and behavior patterns
    to provide personalized next action recommendations.
    """
    if tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )
    
    try:
        logger.info(f"Suggesting next actions for user {request.user_id}")
        
        # Analyze user context
        user_context = await tom_agent.analyze_user_context(
            user_id=request.user_id,
            current_query=request.current_task_context,
        )
        
        # Get next action suggestions
        suggestions = await tom_agent.suggest_next_actions(
            user_context=user_context,
            current_task_context=request.current_task_context,
        )
        
        return SuggestNextActionsResponse(
            user_id=request.user_id,
            suggestions=suggestions,
            success=True,
            message=f"Generated {len(suggestions)} action suggestions",
        )
        
    except Exception as e:
        logger.error(f"Error suggesting next actions for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest next actions: {str(e)}",
        )


@app.post("/api/v1/propose_instructions", response_model=ProposeInstructionsResponse)
async def propose_instructions(request: ProposeInstructionsRequest):
    """
    Propose improved instructions based on user context and preferences.
    
    This endpoint takes an original instruction and personalizes it based on the
    user's mental state, preferences, and behavior patterns.
    """
    if tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )
    
    try:
        logger.info(f"Proposing instructions for user {request.user_id}")
        
        # Analyze user context
        user_context = await tom_agent.analyze_user_context(
            user_id=request.user_id,
            current_query=request.original_instruction,
        )
        
        # Get instruction recommendations
        recommendations = await tom_agent.propose_instructions(
            user_context=user_context,
            original_instruction=request.original_instruction,
            domain_context=request.domain_context,
        )
        
        return ProposeInstructionsResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            success=True,
            message=f"Generated {len(recommendations)} instruction recommendations",
        )
        
    except Exception as e:
        logger.error(f"Error proposing instructions for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to propose instructions: {str(e)}",
        )


@app.post("/api/v1/send_message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """
    Send a message to the ToM agent and get comprehensive personalized guidance.
    
    This endpoint provides the most complete interaction with the ToM agent,
    combining user context analysis, instruction improvement (if provided),
    and next action suggestions into a single comprehensive response.
    """
    if tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )
    
    try:
        logger.info(f"Processing message from user {request.user_id}")
        
        # Get comprehensive personalized guidance
        guidance = await tom_agent.get_personalized_guidance(
            user_id=request.user_id,
            instruction=request.instruction,
            current_task=request.current_task,
            domain_context=request.domain_context,
        )
        
        return SendMessageResponse(
            user_id=request.user_id,
            guidance=guidance,
            success=True,
            message="Generated personalized guidance successfully",
        )
        
    except Exception as e:
        logger.error(f"Error processing message for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )


# Additional utility endpoints

@app.get("/api/v1/users/{user_id}/context")
async def get_user_context(user_id: str):
    """Get the current context for a specific user."""
    if tom_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ToM Agent is not available",
        )
    
    try:
        user_context = await tom_agent.analyze_user_context(user_id=user_id)
        return {
            "user_id": user_id,
            "context": user_context,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Error getting context for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user context: {str(e)}",
        )


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