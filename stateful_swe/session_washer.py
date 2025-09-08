#!/usr/bin/env python3
"""
Session Washer for Stateful SWE Benchmark

Simplified version that modifies user raw sessions to inject user preference signals
based on realistic power user profiles. Uses LLM-based message cleaning and natural
preference signal injection.
"""

import json
import random
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tom-SWE imports
try:
    from tom_swe.generation.generate import LLMClient, LLMConfig

    TOM_SWE_AVAILABLE = True
except ImportError:
    logger.warning(
        "tom_swe generation module not available. LLM-based message processing will be disabled."
    )
    TOM_SWE_AVAILABLE = False
    LLMClient = None  # type: ignore
    LLMConfig = None  # type: ignore

# Environment configuration for LLM
DEFAULT_LLM_MODEL = "gpt-5-mini-2025-08-07"

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class MessageAction(Enum):
    """Available actions for message washing."""

    KEEP_UNCHANGED = "keep_unchanged"
    REPLACE_SENTENCE = "replace_sentence"
    APPEND_PREFERENCE = "append_preference"
    DELETE_CONTENT = "delete_content"


class MessageModification(BaseModel):
    """A single modification action for a message."""

    action: MessageAction = Field(description="The type of modification to perform")
    target_text: Optional[str] = Field(
        default=None,
        description="The specific text to modify (for replace/delete actions)",
    )
    replacement_text: Optional[str] = Field(
        default=None, description="New text to use (for replace/append actions)"
    )
    reasoning: str = Field(
        description="Brief explanation for why this modification improves profile alignment"
    )


class MessageWashingResponse(BaseModel):
    """Structured response for LLM-based message washing."""

    alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the original message aligns with user profile (0.0 = poor alignment, 1.0 = perfect alignment)",
    )
    requires_modification: bool = Field(
        description="Whether the message needs any modifications to better match the user profile"
    )
    modifications: List[MessageModification] = Field(
        default_factory=list,
        description="List of specific modifications to apply to the message",
    )


def count_tokens(session_collection: List[Dict[str, Any]]) -> int:
    """Count total tokens in a collection of sessions using tiktoken.

    Args:
        session_collection: List of session dictionaries

    Returns:
        Total token count across all sessions
    """
    total_tokens = 0

    # Initialize tiktoken encoder
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to simple counting if tiktoken fails
            enc = None
    else:
        enc = None

    for session in session_collection:
        conversation_messages = session.get("conversation_messages", [])
        for message in conversation_messages:
            content = message.get("content", "")
            if content:
                if enc is not None:
                    try:
                        # Use tiktoken for accurate token counting
                        # Allow special tokens and disable disallowed token checks
                        tokens = len(enc.encode(content, disallowed_special=()))
                    except Exception as e:
                        # Fallback to simple counting if tiktoken fails for any reason
                        logger.warning(
                            f"Tiktoken encoding failed: {e}. Using fallback counting."
                        )
                        tokens = len(content.split())
                else:
                    # Fallback to simple whitespace-based counting
                    tokens = len(content.split())
                total_tokens += tokens

    return total_tokens


@dataclass
class WasherConfig:
    """Configuration for SessionWasher behavior."""

    modification_probability: float = 0.4
    random_seed: int = 42
    batch_size: int = 100


class SessionWasher:
    """Simplified session washer that injects user preference signals."""

    def __init__(self, config: Optional[WasherConfig] = None):
        self.config = config or WasherConfig()
        random.seed(self.config.random_seed)

    def load_profiles(self, profiles_path: str) -> List[Dict[str, Any]]:
        """Load user profiles from JSONL file."""
        profiles = []

        with open(profiles_path, "r") as f:
            for line in f:
                if line.strip():
                    profile = json.loads(line.strip())
                    profiles.append(profile)

        print(f"👤 Loaded {len(profiles)} user profiles from {profiles_path}")
        return profiles

    def load_raw_sessions(
        self, sessions_dir: str = "data/stateful_swe/user_raw_sessions"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load raw session files from directory, organized by file."""
        sessions_by_file = {}
        sessions_path = Path(sessions_dir)
        total_conversations = 0

        for session_file in sessions_path.glob("*.json"):
            try:
                file_id = session_file.stem  # Get filename without extension
                file_sessions = []

                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    # Extract individual conversations from the session data
                    for convo_id, convo_data in session_data.items():
                        convo_events = convo_data.get("convo_events", [])

                        # Check if conversation has user messages (filter out system-only conversations)
                        user_messages = [
                            msg for msg in convo_events if msg.get("source") == "user"
                        ]

                        if user_messages:  # Only add conversations with user messages
                            file_sessions.append(
                                {
                                    "original_session_id": convo_id,
                                    "convo_start": convo_data.get("convo_start"),
                                    "convo_end": convo_data.get("convo_end"),
                                    "convo_events": convo_events,  # Keep full conversation
                                }
                            )

                sessions_by_file[file_id] = file_sessions
                total_conversations += len(file_sessions)
                print(f"  📄 {file_id}: {len(file_sessions)} conversations")

            except Exception as e:
                print(f"⚠️  Warning: Could not load {session_file}: {e}")

        print(
            f"📁 Loaded {total_conversations} conversations from {len(sessions_by_file)} files in {sessions_dir}"
        )
        return sessions_by_file

    async def process_user_message_async(
        self,
        message_content: str,
        profile: Dict[str, Any],
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Process and wash user message content using LLM-based analysis with structured actions.

        Args:
            message_content: Original user message content
            profile: User profile containing preferences and roleplay prompt
            conversation_context: Previous messages in the conversation for context

        Returns:
            Processed message content that reflects the user's profile
        """
        if not message_content.strip():
            return ""

        # Skip processing if tom_swe not available or no API key

        try:
            # Initialize LLM client
            config = LLMConfig(
                model=DEFAULT_LLM_MODEL,
            )
            client = LLMClient(config)

            # Create system prompt with profile context
            system_prompt = self._create_message_washing_prompt(profile)

            # Build full prompt for async call
            user_prompt = ""

            # Add conversation context if available
            if conversation_context:
                context_str = "CONVERSATION CONTEXT (previous messages for better understanding):\n\n"
                for i, msg in enumerate(
                    conversation_context[-10:]
                ):  # Last 10 messages for context
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    context_str += f"{i+1}. {role.upper()}: {content}\n"
                context_str += "\n" + "=" * 50 + "\n\n"

                user_prompt = (
                    context_str
                    + f"CURRENT USER MESSAGE to analyze and potentially modify:\n\n{message_content}"
                )
            else:
                user_prompt = f"Original user message to analyze and potentially modify:\n\n{message_content}"

            # Combine system and user prompts for structured async call
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Call LLM with structured output
            if random.random() < self.config.modification_probability:
                response = await client.call_structured_async(
                    prompt=full_prompt, output_type=MessageWashingResponse
                )

                # Apply modifications
                if response.requires_modification and response.modifications:
                    logger.info(
                        f"Applying {len(response.modifications)} modifications for profile {profile['profile_id']}"
                    )
                    # Apply the structured modifications to the original message
                    modified_message = self._apply_modifications(
                        message_content, response.modifications
                    )
                    return modified_message
                else:
                    logger.debug(
                        f"No modifications needed for profile {profile['profile_id']}"
                    )
                    return message_content
            else:
                logger.debug(
                    f"Skipping modification due to probability for profile {profile['profile_id']}"
                )
                return message_content

        except Exception as e:
            logger.warning(f"Error processing user message with LLM: {e}")
            # Fallback to original message if LLM fails
            return message_content

    def _apply_modifications(
        self, original_message: str, modifications: List[MessageModification]
    ) -> str:
        """Apply structured modifications to the original message using string operations.

        Args:
            original_message: The original user message
            modifications: List of modifications to apply

        Returns:
            Modified message after applying all operations
        """
        modified_message = original_message

        for modification in modifications:
            try:
                if modification.action == MessageAction.REPLACE_SENTENCE:
                    if modification.target_text and modification.replacement_text:
                        modified_message = modified_message.replace(
                            modification.target_text, modification.replacement_text
                        )
                        logger.debug(
                            f"Replaced: '{modification.target_text[:50]}...' -> '{modification.replacement_text[:50]}...'"
                        )

                elif modification.action == MessageAction.DELETE_CONTENT:
                    if modification.target_text:
                        modified_message = modified_message.replace(
                            modification.target_text, ""
                        )
                        # Clean up extra whitespace after deletion
                        modified_message = " ".join(modified_message.split())
                        logger.debug(f"Deleted: '{modification.target_text[:50]}...'")

                elif modification.action == MessageAction.APPEND_PREFERENCE:
                    if modification.replacement_text:
                        # Add at the end with appropriate spacing
                        current = modified_message.strip()
                        if current.endswith("."):
                            modified_message = (
                                current + " " + modification.replacement_text
                            )
                        elif current.endswith("?") or current.endswith("!"):
                            modified_message = (
                                current[:-1] + ". " + modification.replacement_text
                            )
                        else:
                            modified_message = (
                                current + ". " + modification.replacement_text
                            )
                        logger.debug(
                            f"Appended: '{modification.replacement_text[:50]}...'"
                        )

                elif modification.action == MessageAction.KEEP_UNCHANGED:
                    logger.debug("Keeping message unchanged as requested")

            except Exception as e:
                logger.warning(
                    f"Failed to apply modification {modification.action}: {e}"
                )
                continue

        return modified_message.strip()

    async def process_user_messages_batch(
        self,
        messages_with_context: List[
            Tuple[str, Dict[str, Any], Optional[List[Dict[str, str]]]]
        ],
        desc: str = "Processing messages",
    ) -> List[str]:
        """Process a batch of user messages asynchronously with progress tracking.

        Args:
            messages_with_context: List of (message_content, profile, conversation_context) tuples
            desc: Description for progress bar

        Returns:
            List of processed messages in the same order
        """
        if not messages_with_context:
            return []

        # Split into batches to avoid overwhelming the API
        batch_size = self.config.batch_size
        processed_messages = []

        # Process in batches with progress bar

        with tqdm(
            total=len(messages_with_context), desc=desc, unit="msg", leave=False
        ) as pbar:
            for i in range(0, len(messages_with_context), batch_size):
                batch = messages_with_context[i : i + batch_size]

                # Create async tasks for this batch
                tasks = []
                for message_content, profile, context in batch:
                    task = self.process_user_message_async(
                        message_content, profile, context
                    )
                    tasks.append(task)

                # Execute batch concurrently
                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle results and exceptions
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to process message {i+j}: {result}")
                            # Use original message as fallback
                            processed_messages.append(batch[j][0])
                        else:
                            processed_messages.append(result)

                        pbar.update(1)

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Fallback: use original messages
                    for message_content, _, _ in batch:
                        processed_messages.append(message_content)
                        pbar.update(1)

        return processed_messages

    def _create_message_washing_prompt(self, profile: Dict[str, Any]) -> str:
        """Create a system prompt for LLM-based message washing.

        Args:
            profile: User profile containing interaction preferences and coding standards

        Returns:
            System prompt for message analysis and modification
        """
        user_roleplay_prompt = profile.get("user_roleplay_prompt", "")

        return f"""You are analyzing and potentially modifying user messages to inject authentic interactional preferences and communication patterns while preserving their core technical intent.

DEVELOPER PROFILE (here's the description of the developer in a second person perspective narrative):
{user_roleplay_prompt}

For INTERACTIONAL PREFERENCES - Your primary goal is to inject this developer's communication patterns (make sure to inject diverse phrases to express the preferences):

    For CONCISE developers (want brief responses from agent):
    - Vary phrases (Randomly choose one from the list): "Keep it short", "Brief response please", "Just the essentials", "Don't over-explain", "Get straight to the point", "Minimal explanation needed", "Keep responses tight", "No lengthy details", "Quick summary only"
    - Examples: "Fix this bug. Keep it concise." / "Implement auth. Brief explanation only." / "Debug this issue - no need for verbose output."

    For VERBOSE developers (want detailed responses from agent):
    - Vary phrases (Randomly choose one from the list): "Explain in detail", "I want thorough explanations", "Walk me through everything", "Provide comprehensive details", "Give me the full breakdown", "I need detailed reasoning", "Elaborate on each step", "Include all context", "Don't skip any details"
    - Examples: "Fix this bug and explain the root cause thoroughly." / "Implement auth with detailed security considerations." / "Debug this - I want to understand every step of your process."

    For UPFRONT question timing (prefer all questions asked first):
    - Vary phrases (Randomly choose one from the list): "Ask all questions upfront", "Get your questions out first", "I prefer to clarify everything before you start", "Plan it all out beforehand", "Let's settle all details first", "Ask me everything you need to know now", "I don't like interruptions during work", "Get all clarifications upfront"
    - Examples: "Implement login system. Ask any questions you have before starting - I prefer to get everything clear first." / "Fix the database issue. Plan out your approach and ask all questions upfront."

    For ONGOING question timing (comfortable with questions during work):
    - Vary phrases (Randomly choose one from the list): "Feel free to ask as you go", "Questions during implementation are fine", "I'm okay with back-and-forth", "Ask questions when needed", "Clarify things as we progress", "I don't mind iterative discussion", "Questions along the way are welcome", "We can figure things out as we go"
    - Examples: "Implement the API. Ask questions as you work on it." / "Debug this issue - feel free to clarify things as you go." / "Build the frontend, we can iterate on requirements as needed."

    For SHORT_RESPONSE preference (user gives brief messages):
    - Make user messages more direct and concise
    - Remove polite fluff, unnecessary context, verbose explanations
    - Examples: "Could you please help me implement a user authentication system with proper security?" → "Implement user auth with security." / "I'm having trouble with this database query and would appreciate your assistance" → "Fix this query."

    For LONG_RESPONSE preference (user gives detailed messages):
    - Keep comprehensive user messages, add context/background when natural
    - User provides thorough explanations and context
    - Examples: "Fix bug" → "Fix this authentication bug. It's affecting user login flow in production, happening since yesterday's deployment. Need to understand root cause and ensure it doesn't happen again."

FOR CODING PREFERENCES - Naturally inject the user's technical preferences when relevant:

    When the user's coding preferences are relevant to the request, subtly inject them:
    For example, If they prefer TypeScript: "Use TypeScript for this" or "make sure it's properly typed"
    For example, If they prefer testing: "Include tests for this" or "make sure to add unit tests"

    More Examples:
    Original: "Create a web application"
    With preferences: "Create a web application using React and TypeScript. Make sure to include unit tests and proper documentation."

    Original: "Fix this database query"
    With preferences: "Fix this database query and optimize it for performance. Add proper error handling."

CRITICAL REQUIREMENTS:
- PRESERVE the core technical request and intent completely
- NEVER change what the user is asking for technically
- Only modify tone, style, communication patterns, and naturally inject preferences
- Keep modifications minimal and authentic
- Focus on making the message sound like this type of developer would write it
- Remove any personal identifiable information (PII) from the message
- If the user message demonstrating misalignment with the provided profile, you should modify the message to make it more aligned with the profile (e.g., removing certain requirements that go against the profile)
- Trying your best to inject users coding/interactional preferences into the message (e.g., if the user wants more concise responses, you should add a traits). However, don't change the technical request and intent completely. And you don't need to do this for every message. Only do this if it is natural to add this preference into the message. Also do not add all preferences into one message, only add one or two preferences into one message when it is natural to do so.
- Use different strategies to inject preferences into the message, not always just APPEND_PREFERENCE.

OUTPUT STRUCTURED RESPONSE with:
- alignment_score: How well the message matches the profile (0.0-1.0)
- requires_modification: Whether any changes would improve profile alignment
- modifications: List of specific string operations to apply (REPLACE_SENTENCE, DELETE_CONTENT, APPEND_PREFERENCE, KEEP_UNCHANGED)
  * For REPLACE_SENTENCE: provide exact target_text to find and replacement_text to substitute
  * For DELETE_CONTENT: provide exact target_text to remove
  * For APPEND_PREFERENCE: provide replacement_text to add at the end"""

    async def wash_session_for_profile_async(
        self, session: Dict[str, Any], profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Wash a single session for a specific profile using async batch processing."""

        # First pass: collect all user messages with their context
        user_messages_data = []

        # Build context incrementally
        current_context: List[Dict[str, str]] = []

        for i, event in enumerate(session["convo_events"]):
            if event.get("source") == "user":
                # Store user message with its context
                original_content = event.get("content", "")
                user_messages_data.append((i, original_content, current_context.copy()))

            # Add all messages to context for next messages
            current_context.append(
                {
                    "role": event.get("source", "unknown"),
                    "content": event.get("content", ""),
                }
            )

        # Process all user messages in batch if there are any
        processed_user_messages = {}
        if user_messages_data:
            messages_for_batch: List[
                Tuple[str, Dict[str, Any], Optional[List[Dict[str, str]]]]
            ] = [
                (content, profile, context)
                for _, content, context in user_messages_data
            ]

            desc = f"Processing {len(user_messages_data)} user messages"
            processed_messages = await self.process_user_messages_batch(
                messages_for_batch, desc
            )

            # Map processed messages back to their positions
            for j, (position, _, _) in enumerate(user_messages_data):
                processed_user_messages[position] = processed_messages[j]

        # Second pass: build the washed conversation
        washed_conversation_messages = []

        for i, event in enumerate(session["convo_events"]):
            if event.get("source") == "user":
                # Use processed user message
                processed_content = processed_user_messages.get(
                    i, event.get("content", "")
                )
                washed_event = {"role": "user", "content": processed_content}
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "assistant":
                # Keep assistant messages unchanged
                washed_event = {
                    "role": "assistant",
                    "content": event.get("content", ""),
                }
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "system":
                # Keep system messages unchanged
                washed_event = {"role": "system", "content": event.get("content", "")}
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "tool":
                # Keep tool messages unchanged
                washed_event = {"role": "tool", "content": event.get("content", "")}
                washed_conversation_messages.append(washed_event)

        # Create the washed session in the expected format
        washed_session = {
            "session_id": f"{session['original_session_id']}",
            "start_time": session.get("convo_start"),
            "end_time": session.get("convo_end"),
            "event_count": len(session["convo_events"]),
            "message_count": len(washed_conversation_messages),
            "conversation_messages": washed_conversation_messages,
        }

        return washed_session

    async def generate_washed_dataset_async(
        self,
        sessions_dir: str = "data/stateful_swe/user_raw_sessions",
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        output_dir: str = "data/stateful_swe/washed_sessions",
        sessions_per_profile: int = 20,
        num_profiles_sample: int = 2,
    ) -> None:
        """Generate the complete washed dataset with async processing and progress bars."""

        # Load data
        print("🔄 Loading data...")
        sessions_by_file = self.load_raw_sessions(sessions_dir)
        profiles = self.load_profiles(profiles_path)
        profiles = profiles[6:num_profiles_sample]
        if not sessions_by_file:
            raise ValueError(f"No sessions found in {sessions_dir}")
        if not profiles:
            raise ValueError(f"No profiles found in {profiles_path}")

        total_conversations = sum(
            len(sessions) for sessions in sessions_by_file.values()
        )
        print("\n🧼 Starting dataset generation:")
        print(f"  - {len(profiles)} profiles")
        print(
            f"  - {total_conversations} raw conversations from {len(sessions_by_file)} files"
        )
        print(f"  - {sessions_per_profile} sessions per profile")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each profile with progress bar
        total_generated = 0

        # Profile-level progress bar
        with tqdm(
            total=len(profiles), desc="Processing profiles", unit="profile"
        ) as profile_pbar:
            for i, profile in enumerate(profiles):
                profile_id = profile["profile_id"]
                profile_pbar.set_description(f"Processing {profile_id}")

                # Select one file ID for this profile, then sample sessions from that file
                file_id = random.choice(list(sessions_by_file.keys()))
                file_sessions = sessions_by_file[file_id]

                # Sample sessions from the chosen file
                if len(file_sessions) >= sessions_per_profile:
                    selected_sessions = random.sample(
                        file_sessions, sessions_per_profile
                    )
                else:
                    # Sample with replacement if not enough sessions in this file
                    selected_sessions = [
                        random.choice(file_sessions)
                        for _ in range(sessions_per_profile)
                    ]

                # Wash sessions for this profile with session-level progress
                profile_output = []

                session_desc = f"Sessions for {profile_id}"
                with tqdm(
                    total=len(selected_sessions),
                    desc=session_desc,
                    unit="session",
                    leave=False,
                ) as session_pbar:
                    for j, session in enumerate(selected_sessions):
                        session_pbar.set_description(
                            f"{session_desc} ({j+1}/{len(selected_sessions)})"
                        )

                        # Process session asynchronously
                        washed_session = await self.wash_session_for_profile_async(
                            session, profile
                        )
                        profile_output.append(washed_session)

                        session_pbar.update(1)

                # Save profile sessions as JSONL
                profile_file = output_path / f"{profile_id}.jsonl"
                profile_final = {
                    "metadata": {
                        "session_token_count": count_tokens(profile_output),
                        "original_user_id": file_id,
                    },
                    "user_profile": profile,
                    "user_sessions": profile_output,
                }
                with open(profile_file, "w") as f:
                    f.write(json.dumps(profile_final, indent=2) + "\n")

                total_generated += len(profile_output)
                profile_pbar.set_postfix(
                    {
                        "sessions": len(profile_output),
                        "tokens": count_tokens(profile_output),
                        "file": file_id[:8],
                    }
                )
                profile_pbar.update(1)

        print("\n🎉 Dataset generation complete!")
        print(f"  - Generated {total_generated} total sessions")
        print(f"  - Saved to {output_dir}/")


async def main_async():
    """Async main function for dataset generation."""
    config = WasherConfig(modification_probability=0.4, random_seed=42, batch_size=50)

    washer = SessionWasher(config)
    await washer.generate_washed_dataset_async(
        sessions_per_profile=20, num_profiles_sample=15
    )


def main():
    """Generate the stateful SWE benchmark dataset."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
