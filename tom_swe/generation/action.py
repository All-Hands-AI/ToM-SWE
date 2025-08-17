"""
Action space definitions and execution engine for the ToM Agent workflow controller.

This module defines the available actions, response models, and execution engine
that the agent can use to interact with files, process sessions, and manage user models.
"""

import asyncio
import json
import logging
from typing import Any, Optional
from datetime import datetime

from tom_swe.generation.dataclass import (
    ActionType,
    ReadFileParams,
    SearchFileParams,
    UpdateJsonFieldParams,
    AnalyzeSessionParams,
    InitializeUserProfileParams,
    RagSearchParams,
    CompleteTaskParams,
    SessionAnalysis,
)
from tom_swe.memory.locations import (
    get_overall_user_model_filename,
    get_cleaned_session_filename,
    get_session_model_filename,
    get_session_models_dir,
)
from tom_swe.memory.store import FileStore
from tom_swe.memory.local import LocalFileStore

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes actions for the ToM Agent workflow controller."""

    def __init__(
        self,
        agent_context: Optional[Any] = None,
        file_store: Optional[FileStore] = None,
    ):
        """
        Initialize the action executor.

        Args:
            agent_context: Reference to the ToM agent for accessing its methods and state
            file_store: FileStore for I/O operations
        """
        self.agent_context = agent_context
        self.file_store = file_store or LocalFileStore(root="~/.openhands")

    def execute_action(self, action: ActionType, parameters: Any) -> str:
        """
        Execute a specific action with given parameters.

        Args:
            action: The action to execute
            parameters: Parameters for the action

        Returns:
            Result of the action execution
        """
        logger.info(f"🎯 Executing action: {action.value}")
        logger.info(f"📋 Parameters: {parameters}")

        if action == ActionType.READ_FILE:
            return self._action_read_file(parameters)
        elif action == ActionType.SEARCH_FILE:
            return self._action_search_file(parameters)
        elif action == ActionType.UPDATE_JSON_FIELD:
            return self._action_update_json_field(parameters)
        elif action == ActionType.ANALYZE_SESSION:
            return self._action_analyze_session(parameters)
        elif action == ActionType.INITIALIZE_USER_PROFILE:
            return self._action_initialize_user_profile(parameters)
        elif action == ActionType.RAG_SEARCH:
            return self._action_rag_search(parameters)
        elif action == ActionType.COMPLETE_TASK:
            return self._action_complete_task(parameters)
        else:
            return f"Action {action.value} not implemented yet"

    # Action implementations
    def _action_read_file(self, params: ReadFileParams) -> str:
        """Read a file."""
        try:
            content = self.file_store.read(params.file_path)
            return content
        except Exception as e:
            return f"Error reading {params.file_path}: {str(e)}"

    def _action_search_file(self, params: SearchFileParams) -> str:
        """Search within files."""
        try:
            if params.search_scope == "cleaned_sessions":
                files = self.file_store.list(
                    get_cleaned_session_filename(params.user_id)
                )
            elif params.search_scope == "session_analyses":
                files = self.file_store.list(get_session_models_dir(params.user_id))
            elif params.search_scope == "user_profiles":
                files = [get_overall_user_model_filename(params.user_id)]
            else:
                files = []

            # Sort files by last_updated if latest_first
            if params.latest_first:
                file_times = []
                for file_path in files:
                    try:
                        content = self.file_store.read(file_path)
                        data = json.loads(content)
                        last_updated = data.get("last_updated", "1970-01-01")
                        file_times.append((file_path, last_updated))
                    except Exception:
                        file_times.append((file_path, "1970-01-01"))
                file_times.sort(key=lambda x: str(x[1]), reverse=True)
                files = [f[0] for f in file_times]

            results = []
            for file_path in files[:50]:
                try:
                    content = self.file_store.read(file_path)
                    if params.query.lower() in content.lower():
                        lines = [
                            line.strip()
                            for line in content.split("\n")
                            if params.query.lower() in line.lower()
                        ][:2]
                        if lines:
                            results.append(f"{file_path}:\n" + "\n".join(lines))
                            if len(results) >= params.max_results:
                                break
                except Exception:
                    continue

            return (
                f"Found {len(results)} files:\n\n" + "\n\n".join(results)
                if results
                else f"No files found containing '{params.query}'"
            )
        except Exception as e:
            return f"Search error: {str(e)}"

    def _action_update_json_field(self, params: UpdateJsonFieldParams) -> str:
        """Update a specific JSON field."""
        try:
            # Read existing data or create new
            try:
                data = json.loads(self.file_store.read(params.file_path))
            except Exception:
                if params.create_if_missing:
                    data = {}
                else:
                    return f"File not found: {params.file_path}"

            # Navigate to field using dot notation
            current = data
            field_parts = params.field_path.split(".")

            for part in field_parts[:-1]:
                if part not in current:
                    if params.create_if_missing:
                        current[part] = {}
                    else:
                        return f"Field path not found: {part}"
                current = current[part]

            # Handle list operations
            final_field = field_parts[-1]
            old_value = current.get(final_field)

            if isinstance(old_value, list) and params.list_operation in [
                "append",
                "remove",
            ]:
                if params.list_operation == "append":
                    if params.new_value not in old_value:  # Avoid duplicates
                        old_value.append(params.new_value)
                elif params.list_operation == "remove":
                    if isinstance(params.new_value, int):
                        # Remove by index
                        if 0 <= params.new_value < len(old_value):
                            old_value.pop(params.new_value)
                    else:
                        # Remove by value
                        if params.new_value in old_value:
                            old_value.remove(params.new_value)
                result_value = old_value
            else:
                # Default: replace the field
                current[final_field] = params.new_value
                result_value = params.new_value

            # Write back
            data["last_updated"] = datetime.now().isoformat()
            self.file_store.write(params.file_path, json.dumps(data, indent=2))

            return f"Updated {params.field_path}: {old_value} → {result_value}"

        except Exception as e:
            return f"Update error: {str(e)}"

    def _action_analyze_session(self, params: AnalyzeSessionParams) -> str:
        """Process a batch of sessions using ToM analyzer."""
        user_id = params.user_id
        session_batch = params.session_batch

        if not session_batch:
            return "Error: session_batch parameter is required"

        logger.info(
            f"🧠 Processing batch of {len(session_batch)} sessions for user {user_id}"
        )

        # Get the ToM analyzer from agent context
        if not self.agent_context or not hasattr(self.agent_context, "tom_analyzer"):
            return "Error: ToM analyzer not available in agent context"

        tom_analyzer = self.agent_context.tom_analyzer

        # Load all session data first
        session_data_list = []
        for session_id in session_batch:
            try:
                session_file = get_cleaned_session_filename(
                    session_id, user_id if user_id else None
                )
                content = self.file_store.read(session_file)
                session_data = json.loads(content)
                session_data_list.append(session_data)
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                continue

        async def _analyze() -> Any:
            return await tom_analyzer.process_session_batch(session_data_list)

        session_summaries = asyncio.run(_analyze())
        # Save session analyses and prepare result
        session_dumps = []
        for session_analysis in session_summaries:
            session_dump = session_analysis.model_dump()
            session_dumps.append(session_dump)
            session_file = get_session_model_filename(
                session_analysis.session_id, user_id
            )
            self.file_store.write(session_file, json.dumps(session_dump, indent=2))

        return json.dumps(session_dumps, indent=2)

    def _action_initialize_user_profile(
        self, params: InitializeUserProfileParams
    ) -> str:
        """Initialize and save user analysis using tom_module."""
        user_id = params.user_id
        if not self.agent_context or not hasattr(self.agent_context, "tom_analyzer"):
            return "Error: ToM analyzer not available in agent context"
        tom_analyzer = self.agent_context.tom_analyzer

        # Load all session analyses from files
        session_analyses = []

        for filename in self.file_store.list(get_session_models_dir(user_id)):
            if filename.endswith(".json"):
                content = self.file_store.read(filename)
                session_data = json.loads(content)
                # Convert back to SessionAnalysis object
                session_analysis = SessionAnalysis(**session_data)
                session_analyses.append(session_analysis)

        # Call ToM analyzer to initialize user analysis
        async def _initialize() -> Any:
            return await tom_analyzer.initialize_user_analysis(session_analyses)

        user_analysis = asyncio.run(_initialize())

        # Save user analysis to file and prepare result
        user_analysis_dump = user_analysis.model_dump()
        user_model_file = get_overall_user_model_filename(user_id)
        self.file_store.write(user_model_file, json.dumps(user_analysis_dump, indent=2))

        return json.dumps(user_analysis_dump, indent=2)

    def _action_rag_search(self, params: RagSearchParams) -> str:
        """Search for relevant context using RAG."""
        # TODO: Implement RAG functionality
        return f"RAG search for '{params.query}' (k={params.k}) - not implemented yet"

    def _action_complete_task(self, params: CompleteTaskParams) -> str:
        """Mark task as complete."""
        logger.info("✅ Task marked as complete")
        return params.result
