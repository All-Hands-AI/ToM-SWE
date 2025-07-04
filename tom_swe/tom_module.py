#!/usr/bin/env python3
"""
Theory of Mind (ToM) Module for User Behavior Analysis

This module analyzes user interaction data to understand their mental states,
predict their intentions, and anticipate their next actions based on their
typed messages and interaction patterns.

The module processes user data from processed_data/*.json files and generates
comprehensive mental state models saved to data/user_model/.
"""

# Standard library imports
import asyncio
import csv
import glob
import json
import os
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiofiles  # type: ignore
import litellm

# Third-party imports
from dotenv import load_dotenv
from litellm import acompletion
from rich import print

from .database import (
    OverallUserAnalysis,
    SessionSummary,
    UserDescriptionAndPreferences,
    UserMessageAnalysis,
    UserProfile,
)
from .utils import PydanticOutputParser

# Local imports

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure litellm for better error handling
litellm.set_verbose = False

# Configure LLM proxy settings
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514")

# Constants
MIN_CLARIFICATION_SEQUENCE = 2


class UserMentalStateAnalyzer:
    """
    Analyzes user mental states and predicts intentions from their interactions using LLMs.

    This class provides a complete pipeline for analyzing user behavior:
    1. Load user interaction data
    2. Analyze individual messages using LLM
    3. Process sessions and create summaries
    4. Manage user profiles over time
    5. Handle all file I/O operations
    """

    def __init__(
        self,
        processed_data_dir: str = "./data/processed_data",
        model: Optional[str] = None,
        session_batch_size: int = 3,
    ) -> None:
        """Initialize the analyzer with configuration and validate setup."""
        self.processed_data_dir = processed_data_dir
        self.model = model or DEFAULT_LLM_MODEL
        self.session_batch_size = session_batch_size
        self._validate_and_setup_configuration()

    # ===================================================================
    # INITIALIZATION & CONFIGURATION GROUP
    # ===================================================================

    def _validate_and_setup_configuration(self) -> None:
        """Validate and setup LLM configuration."""
        if not LITELLM_API_KEY:
            print("Warning: LITELLM_API_KEY not set. LLM analysis will fail.")
            print("Please set LITELLM_API_KEY environment variable or create a .env file.")

        # Set up litellm configuration if we have the required settings
        if LITELLM_API_KEY and LITELLM_BASE_URL:
            os.environ["LITELLM_API_KEY"] = LITELLM_API_KEY
            os.environ["LITELLM_BASE_URL"] = LITELLM_BASE_URL

    # ===================================================================
    # DATA LOADING GROUP
    # ===================================================================

    async def load_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load complete user data from processed_data/{user_id}.json"""
        user_file = os.path.join(self.processed_data_dir, f"{user_id}.json")

        if not os.path.exists(user_file):
            return None

        try:
            async with aiofiles.open(user_file, encoding="utf-8") as f:
                content = await f.read()
                data: Dict[str, Any] = json.loads(content)
                return data
        except Exception as e:
            print(f"Error loading {user_file}: {e}")
            return None

    async def get_user_session_ids(self, user_id: str) -> List[str]:
        """Get all session IDs for a user from processed_data/{user_id}.json"""
        user_data = await self.load_user_data(user_id)
        return list(user_data.keys()) if user_data else []

    async def load_session_data(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Load specific session data for a user."""
        user_data = await self.load_user_data(user_id)
        if not user_data:
            return None
        return user_data.get(session_id)

    def load_studio_results_metadata(self, data_dir: str = "./data") -> Dict[str, Dict[str, Any]]:
        """
        Load metadata from studio_results_*.csv files.
        Returns a dictionary with conversation_id as key and metadata as value.
        """
        metadata: Dict[str, Dict[str, Any]] = {}

        # Find all studio_results CSV files
        csv_pattern = os.path.join(data_dir, "studio_results_*.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"No studio_results CSV files found in {data_dir}")
            return metadata

        for csv_file in csv_files:
            try:
                with open(csv_file, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        conversation_id = row.get("conversation_id", "")
                        if conversation_id:
                            metadata[conversation_id] = {
                                "conversation_id": conversation_id,
                                "github_user_id": row.get("github_user_id", ""),
                                "selected_repository": row.get("selected_repository", ""),
                                "title": row.get("title", ""),
                                "last_updated_at": row.get("last_updated_at", ""),
                                "created_at": row.get("created_at", ""),
                                "selected_branch": row.get("selected_branch", ""),
                                "user_id": row.get("user_id", ""),
                                "accumulated_cost": row.get("accumulated_cost", ""),
                                "prompt_tokens": row.get("prompt_tokens", ""),
                                "completion_tokens": row.get("completion_tokens", ""),
                                "total_tokens": row.get("total_tokens", ""),
                                "trigger": row.get("trigger", ""),
                                "pr_number": row.get("pr_number", ""),
                            }
                print(f"Loaded {len(metadata)} conversation metadata from {csv_file}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        return metadata

    def extract_user_typed_messages_from_session(
        self, session_data: Dict[str, Any]
    ) -> List[Tuple[str, List[str]]]:
        """
        Extract user-typed messages from session data with their context.
        Returns list of (message_content, session_context) tuples.
        """
        if not session_data or "convo_events" not in session_data:
            return []

        session_context = []
        user_typed_messages = []

        for event in session_data["convo_events"]:
            session_context.append(str(event))
            if event.get("source") == "user":
                content = event.get("content", "")
                if self.categorize_message_type(content) == "user_typed":
                    user_typed_messages.append((content, session_context.copy()))

        return user_typed_messages

    def categorize_message_type(self, content: str) -> str:
        """
        Categorize user messages into typed vs micro-agent triggered.
        (Reusing logic from user_interaction_analysis.py)
        """
        system_tags = [
            "<REPOSITORY_INFO>",
            "<REPOSITORY_INSTRUCTIONS>",
            "<EXTRA_INFO>",
            "<RUNTIME_INFORMATION>",
            "<WORKSPACE_FILES>",
            "<CURSOR_POSITION>",
            "<DIAGNOSTICS>",
            "<SEARCH_RESULTS>",
        ]

        for tag in system_tags:
            if tag in content:
                return "micro_agent_triggered"

        return "user_typed"

    # ===================================================================
    # LLM SERVICE GROUP
    # ===================================================================

    async def call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Call the LLM with error handling and retries using async.
        """
        if not LITELLM_API_KEY:
            print("LLM API key not configured. Skipping LLM analysis.")
            return None

        try:
            # Prepare the completion call with proxy configuration
            completion_args = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": response_format,
            }

            # Add max_tokens if specified
            if max_tokens:
                completion_args["max_tokens"] = max_tokens

            # Add API key and base URL if available
            if LITELLM_API_KEY:
                completion_args["api_key"] = LITELLM_API_KEY
            if LITELLM_BASE_URL:
                completion_args["api_base"] = LITELLM_BASE_URL

            # Use native async completion
            response = await acompletion(**completion_args)
            content: str = response.choices[0].message.content
            return content.strip()

        except Exception as e:
            print(f"Error calling LLM ({self.model}): {e}")
            return None

    def build_message_analysis_prompt(
        self, message: str, session_context: Optional[List[str]] = None
    ) -> str:
        """Build the prompt for analyzing a single user message."""
        parser: PydanticOutputParser[UserMessageAnalysis] = PydanticOutputParser(
            pydantic_object=UserMessageAnalysis
        )
        format_instructions = parser.get_format_instructions()

        # Build context from previous messages in session
        context_str = "\n".join(session_context or [])
        context_info = f"""
Session context (recent messages):
{context_str}
"""

        return f"""
Analyze this user message to a coding assistant across four dimensions. Provide your analysis in JSON format.

{context_info}Current user message: "{message}"

{format_instructions}
"""

    async def generate_overall_user_description(
        self, analysis: OverallUserAnalysis
    ) -> UserDescriptionAndPreferences:
        """Generate an overall description of the user and preference summary using LLM based on session summaries"""
        summaries = analysis.session_summaries
        profile = analysis.user_profile

        if not summaries:
            return UserDescriptionAndPreferences(
                description="New user with no session history yet.", preferences=[]
            )

        # Collect user modeling insights from all sessions
        user_insights = []
        for summary in summaries:
            if summary.user_modeling_summary:
                user_insights.append(summary.user_modeling_summary)

        # Collect all preferences from session summaries
        all_preferences = []
        for summary in summaries:
            all_preferences.extend(summary.key_preferences)

        # Prepare data for LLM prompt
        insights_text = (
            "; ".join(user_insights) if user_insights else "No specific insights available"
        )
        intent_summary = ", ".join(
            [f"{intent}: {count}" for intent, count in profile.intent_distribution.items()]
        )
        emotion_summary = ", ".join(
            [f"{emotion}: {count}" for emotion, count in profile.emotion_distribution.items()]
        )
        all_preferences_text = (
            ", ".join(set(all_preferences))
            if all_preferences
            else "No specific preferences identified"
        )

        # Use Pydantic parser for structured output
        parser: PydanticOutputParser[UserDescriptionAndPreferences] = PydanticOutputParser(
            pydantic_object=UserDescriptionAndPreferences
        )
        format_instructions = parser.get_format_instructions()

        # Single LLM call for both description and preferences
        combined_prompt = f"""
Based on the following user interaction data, analyze this user and provide both an overall description and key preferences:

User ID: {profile.user_id}
Total Sessions: {profile.total_sessions}
Session Insights: {insights_text}
Intent Distribution: {intent_summary}
Emotion Distribution: {emotion_summary}
Observed Preferences: {all_preferences_text}

{format_instructions}
"""

        # Make single LLM call with structured response format
        result = await self.call_llm(combined_prompt, response_format=parser.pydantic_object)
        if not result:
            return UserDescriptionAndPreferences(
                description="New user with no session history yet.", preferences=[]
            )
        parsed_result: UserDescriptionAndPreferences = parser.parse(result)
        return parsed_result

    # ===================================================================
    # MESSAGE ANALYSIS GROUP
    # ===================================================================

    async def analyze_single_message(
        self, message: str, session_context: Optional[List[str]] = None
    ) -> UserMessageAnalysis:
        """
        Comprehensive analysis of a user message to extract intent, emotions, preferences, and constraints.
        """
        parser: PydanticOutputParser[UserMessageAnalysis] = PydanticOutputParser(
            pydantic_object=UserMessageAnalysis
        )
        prompt = self.build_message_analysis_prompt(message, session_context)

        print("\n[bold blue]Analyzing user message:[/bold blue]")
        print(f"[yellow]{message}[/yellow]")

        result = await self.call_llm(prompt, response_format=parser.pydantic_object)

        if not result:
            return UserMessageAnalysis(
                intent="general",
                emotions=["neutral"],
                preference=["minimal"],
                user_modeling="",
                should_ask_clarification=False,
            )

        structured_result: UserMessageAnalysis = parser.parse(result)
        print("\n[bold green]Result:[/bold green]")
        print(structured_result)
        return structured_result

    async def analyze_all_session_messages(
        self, user_id: str, session_id: str
    ) -> Optional[Tuple[List[UserMessageAnalysis], List[str]]]:
        """
        Analyze all user messages in a session.
        Returns tuple of (analyses_list, user_messages_list) or None if no data found.
        """
        session_data = await self.load_session_data(user_id, session_id)
        if not session_data:
            return None

        user_typed_messages = self.extract_user_typed_messages_from_session(session_data)
        if not user_typed_messages:
            return [], []

        user_analyses: List[UserMessageAnalysis] = []
        user_messages: List[str] = []

        # Process messages sequentially to maintain time dependencies
        for content, context in user_typed_messages:
            analysis = await self.analyze_single_message(content, context)
            user_analyses.append(analysis)
            user_messages.append(content)

        return user_analyses, user_messages

    # ===================================================================
    # SESSION PROCESSING GROUP
    # ===================================================================

    def create_session_summary_from_analyses(
        self,
        session_id: str,
        analyses_list: List[UserMessageAnalysis],
        session_metadata: Dict[str, Any],
    ) -> Optional[SessionSummary]:
        """Create session summary from list of UserMessageAnalysis objects"""
        if not analyses_list:
            return None

        # Count intents and emotions
        intent_counter = Counter(a.intent for a in analyses_list)
        all_emotions = [emotion for a in analyses_list for emotion in a.emotions]
        emotion_counter = Counter(all_emotions)

        # Collect preferences
        all_preferences = []
        for a in analyses_list:
            if a.preference and a.preference != ["none"]:
                all_preferences.extend(a.preference)

        # Count clarification requests
        clarification_count = sum(1 for a in analyses_list if a.should_ask_clarification)

        # Aggregate user modeling insights
        user_insights = [a.user_modeling for a in analyses_list if a.user_modeling.strip()]
        user_modeling_summary = "; ".join(user_insights[:3])  # Take first 3

        return SessionSummary(
            session_id=session_id,
            timestamp=datetime.now(),
            message_count=len(analyses_list),
            intent_distribution=dict(intent_counter),
            emotion_distribution=dict(emotion_counter),
            key_preferences=list(set(all_preferences)),
            user_modeling_summary=user_modeling_summary,
            clarification_requests=clarification_count,
            session_start=session_metadata.get("convo_start", ""),
            session_end=session_metadata.get("convo_end", ""),
        )

    def calculate_session_metrics(self, analyses_list: List[UserMessageAnalysis]) -> Dict[str, Any]:
        """Calculate various metrics for a session."""
        if not analyses_list:
            return {}

        return {
            "total_messages": len(analyses_list),
            "intent_distribution": dict(Counter(a.intent for a in analyses_list)),
            "emotion_distribution": dict(
                Counter(emotion for a in analyses_list for emotion in a.emotions)
            ),
            "clarification_ratio": sum(1 for a in analyses_list if a.should_ask_clarification)
            / len(analyses_list),
            "unique_preferences": len(
                {pref for a in analyses_list for pref in a.preference if pref != "none"}
            ),
        }

    # ===================================================================
    # USER PROFILE MANAGEMENT GROUP
    # ===================================================================

    async def load_existing_user_profile(
        self, user_id: str, base_dir: str = "./data/user_model"
    ) -> Optional[OverallUserAnalysis]:
        """Load existing user profile or return None if not found."""
        overall_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")

        if not os.path.exists(overall_path):
            return None

        try:
            async with aiofiles.open(overall_path, encoding="utf-8") as f:
                content = await f.read()
                existing_data = json.loads(content)
            return OverallUserAnalysis(**existing_data)
        except Exception as e:
            print(f"Error loading existing profile for {user_id}: {e}")
            return None

    def create_new_user_profile(self, user_id: str) -> OverallUserAnalysis:
        """Create new OverallUserAnalysis"""
        user_profile = UserProfile(
            user_id=user_id,
            total_sessions=0,
            overall_description="",
            intent_distribution={},
            emotion_distribution={},
            preference_summary=[],
        )

        return OverallUserAnalysis(
            user_profile=user_profile, session_summaries=[], last_updated=datetime.now()
        )

    def update_user_profile_with_session(self, analysis: OverallUserAnalysis) -> None:
        """Update user profile based on session summaries"""
        summaries = analysis.session_summaries
        profile = analysis.user_profile

        # Update basic stats
        profile.total_sessions = len(summaries)

        # Aggregate intent distribution
        intent_counts: Counter[str] = Counter()
        for summary in summaries:
            for intent, count in summary.intent_distribution.items():
                intent_counts[intent] += count
        profile.intent_distribution = dict(intent_counts)

        # Aggregate emotion distribution
        emotion_counts: Counter[str] = Counter()
        for summary in summaries:
            for emotion, count in summary.emotion_distribution.items():
                emotion_counts[emotion] += count
        profile.emotion_distribution = dict(emotion_counts)

        # Note: overall_description will be generated by LLM in async method

    async def save_updated_user_profile(
        self,
        user_id: str,
        new_session_summaries: List[SessionSummary],
        base_dir: str = "./data/user_model",
    ) -> None:
        """Load existing analysis, add new sessions, update user profile and save."""
        if not new_session_summaries:
            print(f"No session summaries to update for user {user_id}")
            return

        self.ensure_directory_structure(base_dir)

        # Load existing or create new
        analysis = await self.load_existing_user_profile(user_id, base_dir)
        if not analysis:
            analysis = self.create_new_user_profile(user_id)

        # Add all new session summaries
        analysis.session_summaries.extend(new_session_summaries)
        analysis.last_updated = datetime.now()

        # Update user profile
        self.update_user_profile_with_session(analysis)

        # Generate overall description using LLM
        description_and_preferences = await self.generate_overall_user_description(analysis)
        analysis.user_profile.overall_description = description_and_preferences.description
        analysis.user_profile.preference_summary = description_and_preferences.preferences

        # Save updated analysis
        overall_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")
        try:
            async with aiofiles.open(overall_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(analysis.model_dump(), indent=2, default=str))
            print(
                f"Updated user profile for {user_id} with {len(new_session_summaries)} new sessions"
            )
        except Exception as e:
            print(f"Error saving user profile: {e}")

    # ===================================================================
    # FILE MANAGEMENT GROUP
    # ===================================================================

    def ensure_directory_structure(self, base_dir: str = "./data/user_model") -> None:
        """Create necessary directory structure"""
        detailed_dir = os.path.join(base_dir, "user_model_detailed")
        overall_dir = os.path.join(base_dir, "user_model_overall")

        os.makedirs(detailed_dir, exist_ok=True)
        os.makedirs(overall_dir, exist_ok=True)

    async def save_session_analyses_to_json(
        self,
        user_id: str,
        session_id: str,
        analysis_data: Tuple[List[UserMessageAnalysis], List[str]],
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save per-message analyses to JSON file with original user messages and metadata"""
        base_dir = "./data/user_model"  # Use default base directory
        analyses_list, user_messages = analysis_data
        self.ensure_directory_structure(base_dir)

        user_dir = os.path.join(base_dir, "user_model_detailed", user_id)
        os.makedirs(user_dir, exist_ok=True)

        json_path = os.path.join(user_dir, f"{session_id}.json")

        # Prepare message analyses as a list
        message_analyses = []
        for i, (analysis, user_message) in enumerate(zip(analyses_list, user_messages)):
            message_data = {
                "message_index": i,
                "timestamp": datetime.now().isoformat(),
                "original_message": user_message,
                "analysis": analysis.model_dump(),
            }
            message_analyses.append(message_data)

        # Prepare final JSON structure with metadata and analyses
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "metadata": session_metadata or {},
            "message_analyses": message_analyses,
            "summary": {
                "total_messages": len(analyses_list),
                "created_at": datetime.now().isoformat(),
            },
        }

        try:
            async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(session_data, indent=2, default=str))
            print(f"Saved {len(analyses_list)} message analyses to {json_path}")
        except Exception as e:
            print(f"Error saving analyses to {json_path}: {e}")

    # ===================================================================
    # ORCHESTRATION GROUP - Main Pipeline Methods
    # ===================================================================

    async def _process_session_batch(
        self,
        user_id: str,
        session_batch: List[str],
        studio_metadata: Dict[str, Dict[str, Any]],
        base_dir: str,
    ) -> List[SessionSummary]:
        """
        Process a batch of sessions concurrently.
        Returns list of successfully processed session summaries.
        """

        async def process_single_session_with_metadata(session_id: str) -> Optional[SessionSummary]:
            session_metadata = studio_metadata.get(session_id, {})
            if session_metadata:
                print(
                    f"Found metadata for session {session_id}: {session_metadata.get('title', 'No title')}"
                )
            else:
                print(f"No metadata found for session {session_id}")

            return await self.process_and_save_single_session(
                user_id, session_id, session_metadata, base_dir
            )

        # Process all sessions in the batch concurrently
        tasks = [process_single_session_with_metadata(session_id) for session_id in session_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful session summaries
        session_summaries: List[SessionSummary] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing session {session_batch[i]}: {result}")
            elif isinstance(result, SessionSummary):
                session_summaries.append(result)

        return session_summaries

    async def process_and_save_single_session(
        self,
        user_id: str,
        session_id: str,
        session_metadata: Optional[Dict[str, Any]] = None,
        base_dir: str = "./data/user_model",
    ) -> Optional[SessionSummary]:
        """
        Complete analysis pipeline for one session, excluding user profile update.
        Returns SessionSummary if successful, None otherwise.
        """
        print(f"\nProcessing session {session_id} for user {user_id}")

        # Step 1: Get per-message analyses
        result = await self.analyze_all_session_messages(user_id, session_id)
        if not result:
            print(f"No analyses found for session {session_id}")
            return None

        analyses, user_messages = result
        if not analyses:
            print(f"No analyses found for session {session_id}")
            return None

        # Get session metadata for summary
        session_data = await self.load_session_data(user_id, session_id)
        session_summary_metadata = session_data or {}

        # Step 2: Save to JSON (Tier 1) with studio_results metadata
        await self.save_session_analyses_to_json(
            user_id, session_id, (analyses, user_messages), session_metadata
        )

        # Step 3: Create session summary (Tier 2)
        session_summary = self.create_session_summary_from_analyses(
            session_id, analyses, session_summary_metadata
        )
        if not session_summary:
            print(f"Could not create session summary for {session_id}")
            return None

        print(f"✅ Completed analysis for session {session_id}")
        return session_summary
