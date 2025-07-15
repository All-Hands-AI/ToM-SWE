#!/usr/bin/env python3
"""
CLI script to run the ToM Agent API server.

This script provides a convenient way to start the FastAPI server
with appropriate configuration.
"""

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="ToM Agent API Server")
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "false").lower() == "true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--processed-data-dir",
        default=os.getenv("TOM_PROCESSED_DATA_DIR", "./data/processed_data"),
        help="Directory containing processed user data",
    )
    parser.add_argument(
        "--user-model-dir",
        default=os.getenv("TOM_USER_MODEL_DIR", "./data/user_model"),
        help="Directory containing user model data",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Set environment variables for the ToM agent
    os.environ["TOM_PROCESSED_DATA_DIR"] = args.processed_data_dir
    os.environ["TOM_USER_MODEL_DIR"] = args.user_model_dir

    # Verify data directories exist
    processed_data_path = Path(args.processed_data_dir)
    user_model_path = Path(args.user_model_dir)

    if not processed_data_path.exists():
        print(f"Warning: Processed data directory does not exist: {processed_data_path}")
        print("Creating directory...")
        processed_data_path.mkdir(parents=True, exist_ok=True)

    if not user_model_path.exists():
        print(f"Warning: User model directory does not exist: {user_model_path}")
        print("Creating directory...")
        user_model_path.mkdir(parents=True, exist_ok=True)

    print("Starting ToM Agent API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Processed data dir: {args.processed_data_dir}")
    print(f"User model dir: {args.user_model_dir}")
    print(f"Log level: {args.log_level}")
    print()

    try:
        uvicorn.run(
            "tom_swe.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
