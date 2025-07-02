#!/usr/bin/env python3
"""
Enhanced User Behavior Analysis

This script extracts messages where users expressed negative emotions,
analyzes user preferences, and generates best practices for user interaction.
The output includes actionable insights for improving system prompts and user experience.
"""

import asyncio
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from litellm import acompletion

# Load environment variables
load_dotenv()

# Configure LLM settings
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514")

# Constants for magic numbers
MAX_EXAMPLES_PER_BATCH = 15
MAX_USERS_FOR_PREFERENCE_ANALYSIS = 30


@dataclass
class AnalysisData:
    """Container for analysis data to reduce function parameters."""

    negative_messages: Dict[str, List[Dict[str, Any]]]
    user_preferences: Dict[str, Dict[str, Any]]
    frustration_analysis: Dict[str, Any]
    preference_analysis: Dict[str, Any]
    best_practices: Dict[str, Any]


class EnhancedUserBehaviorAnalyzer:
    def __init__(
        self,
        detailed_data_path: str = "data/user_model/user_model_detailed",
        overall_data_path: str = "data/user_model/user_model_overall",
        output_path: str = "data/user_analysis/enhanced_behavior_analysis",
        model: str = DEFAULT_LLM_MODEL,
    ):
        self.detailed_data_path = detailed_data_path
        self.overall_data_path = overall_data_path
        self.output_path = output_path
        self.model = model
        self.batch_size = 50  # Default batch size
        # Default values for sample parameters
        self.sample_users: Optional[int] = None
        self.sample_messages_per_user: int = 10

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Define negative emotions based on database.py
        self.negative_emotions = {"frustrated", "confused", "overwhelmed"}

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for processing."""
        self.batch_size = batch_size

    def set_sampling_parameters(
        self, sample_users: Optional[int] = None, sample_messages_per_user: int = 10
    ) -> None:
        """Set sampling parameters separately to avoid too many constructor arguments."""
        self.sample_users = sample_users
        self.sample_messages_per_user = sample_messages_per_user

    async def call_llm(self, prompt: str) -> str:
        """Simple LLM call using litellm"""
        if not LITELLM_API_KEY:
            return "LLM not configured"

        try:
            completion_args = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }

            if LITELLM_API_KEY:
                completion_args["api_key"] = LITELLM_API_KEY
            if LITELLM_BASE_URL:
                completion_args["api_base"] = LITELLM_BASE_URL

            response = await acompletion(**completion_args)
            content = response.choices[0].message.content
            return str(content).strip() if content else "No response"
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error extracting key phrases"

    def extract_negative_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all messages with negative sentiments."""
        print(f"Loading negative sentiment data from: {self.detailed_data_path}")

        user_dirs = glob.glob(os.path.join(self.detailed_data_path, "*"))
        user_dirs = [d for d in user_dirs if os.path.isdir(d)]

        # Apply user sampling if specified
        if self.sample_users and self.sample_users < len(user_dirs):
            import random

            user_dirs = random.sample(user_dirs, self.sample_users)
            print(
                f"Sampling {len(user_dirs)} users from {len(glob.glob(os.path.join(self.detailed_data_path, '*')))} total users"
            )

        print(f"Found {len(user_dirs)} user directories")

        all_negative_messages = {}

        for user_dir in user_dirs:
            user_id = os.path.basename(user_dir)
            print(f"Processing user: {user_id}")

            user_negative_messages = []

            # Get all JSON files for this user
            json_files = glob.glob(os.path.join(user_dir, "*.json"))
            print(f"  Found {len(json_files)} session files")

            for json_file in json_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        session_data = json.load(f)

                    session_negative_messages = self._extract_from_session(session_data)
                    user_negative_messages.extend(session_negative_messages)

                except Exception as e:
                    print(f"  Error processing {json_file}: {e!s}")
                    continue

            # Apply message sampling per user
            if (
                self.sample_messages_per_user
                and len(user_negative_messages) > self.sample_messages_per_user
            ):
                import random

                user_negative_messages = random.sample(
                    user_negative_messages, self.sample_messages_per_user
                )

            all_negative_messages[user_id] = user_negative_messages
            print(f"  Found {len(user_negative_messages)} negative messages for this user")

        return all_negative_messages

    def _extract_from_session(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract negative messages from a single session."""
        session_id = session_data.get("session_id", "unknown")
        message_analyses = session_data.get("message_analyses", [])
        metadata = session_data.get("metadata", {})

        negative_messages = []

        for message_analysis in message_analyses:
            analysis = message_analysis.get("analysis", {})
            emotions = analysis.get("emotions", [])
            original_message = message_analysis.get("original_message", "")
            timestamp = message_analysis.get("timestamp", "")

            # Check if any emotion is negative
            negative_emotions_found = [
                emotion for emotion in emotions if emotion in self.negative_emotions
            ]

            if negative_emotions_found:
                negative_entry = {
                    "session_id": session_id,
                    "session_title": metadata.get("title", "Untitled"),
                    "message_index": message_analysis.get("message_index", 0),
                    "timestamp": timestamp,
                    "original_message": original_message,
                    "negative_emotions": negative_emotions_found,
                    "intent": analysis.get("intent", "unknown"),
                    "repository": metadata.get("selected_repository", ""),
                    "preferences": analysis.get(
                        "preference", []
                    ),  # Include preferences from negative messages
                }

                negative_messages.append(negative_entry)

        return negative_messages

    def extract_user_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Extract comprehensive user preferences from overall user models."""
        print(f"Loading user preferences from: {self.overall_data_path}")

        json_files = glob.glob(os.path.join(self.overall_data_path, "*.json"))
        print(f"Found {len(json_files)} user files")

        all_user_preferences = {}

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    user_data = json.load(f)

                user_profile = user_data.get("user_profile", {})
                user_id = user_profile.get(
                    "user_id", os.path.basename(json_file).replace(".json", "")
                )

                # Extract comprehensive preference data
                preference_data = {
                    "user_id": user_id,
                    "overall_description": user_profile.get("overall_description", ""),
                    "preference_summary": user_profile.get("preference_summary", []),
                    "intent_distribution": user_profile.get("intent_distribution", {}),
                    "emotion_distribution": user_profile.get("emotion_distribution", {}),
                    "total_sessions": user_profile.get("total_sessions", 0),
                    "session_preferences": [],
                }

                # Extract session-level preferences
                for session in user_data.get("session_summaries", []):
                    session_prefs = {
                        "session_id": session.get("session_id", ""),
                        "key_preferences": session.get("key_preferences", []),
                        "user_modeling_summary": session.get("user_modeling_summary", ""),
                        "intent_distribution": session.get("intent_distribution", {}),
                        "emotion_distribution": session.get("emotion_distribution", {}),
                    }
                    preference_data["session_preferences"].append(session_prefs)

                all_user_preferences[user_id] = preference_data

            except Exception as e:
                print(f"  Error processing {json_file}: {e!s}")
                continue

        print(f"Extracted preferences for {len(all_user_preferences)} users")
        return all_user_preferences

    async def analyze_frustration_patterns(
        self, negative_messages: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze what causes user frustration using LLM with optimized batch processing."""
        print("Analyzing frustration patterns...")

        # Collect all negative messages for pattern analysis
        all_negative_msgs = []

        for _, messages in negative_messages.items():
            for msg in messages:
                all_negative_msgs.append(
                    {
                        "message": msg["original_message"][:300],  # Truncate for efficiency
                        "emotions": msg["negative_emotions"],
                        "intent": msg["intent"],
                        "session_title": msg.get("session_title", "")[:100],  # Truncate
                    }
                )

        if not all_negative_msgs:
            return {"frustration_patterns": [], "root_causes": [], "triggers": []}

        print(f"Analyzing {len(all_negative_msgs)} negative messages in batches...")

        # Use larger, more efficient batches
        batch_size = self.batch_size
        batches = [
            all_negative_msgs[i : i + batch_size]
            for i in range(0, len(all_negative_msgs), batch_size)
        ]

        # Single comprehensive analysis call instead of separate calls
        all_analysis_results = []

        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} messages)")

            # Create more efficient batch context
            batch_summary: List[str] = []
            emotion_counts: Dict[str, int] = {}
            intent_counts: Dict[str, int] = {}

            for msg in batch:
                # Count emotions and intents for pattern detection
                for emotion in msg["emotions"]:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                intent_counts[msg["intent"]] = intent_counts.get(msg["intent"], 0) + 1

                # Add message samples
                if len(batch_summary) < MAX_EXAMPLES_PER_BATCH:  # Limit examples per batch
                    batch_summary.append(
                        f"'{msg['message'][:150]}...' (emotions: {','.join(msg['emotions'])}, intent: {msg['intent']})"
                    )

            # Single comprehensive prompt for this batch
            prompt = f"""
Analyze this batch of {len(batch)} user messages expressing frustration, confusion, or feeling overwhelmed.

EMOTION DISTRIBUTION: {dict(emotion_counts)}
INTENT DISTRIBUTION: {dict(intent_counts)}

SAMPLE MESSAGES:
{chr(10).join(batch_summary)}

Provide analysis in this exact format:

FRUSTRATION_PATTERNS:
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]

ROOT_CAUSES:
- [Cause 1]
- [Cause 2]
- [Cause 3]

TRIGGERS:
- [Trigger 1]
- [Trigger 2]
- [Trigger 3]
"""

            result = await self.call_llm(prompt)
            if result and result != "LLM not configured":
                all_analysis_results.append(result)

        # Consolidate results from all batches
        return self._consolidate_frustration_analysis(all_analysis_results)

    def _consolidate_frustration_analysis(self, analysis_results: List[str]) -> Dict[str, Any]:
        """Consolidate multiple batch analysis results."""
        all_patterns = []
        all_root_causes = []
        all_triggers = []

        for result in analysis_results:
            result_lines = result.split("\n")
            current_section = None

            for line in result_lines:
                stripped_line = line.strip()
                if stripped_line.startswith("FRUSTRATION_PATTERNS:"):
                    current_section = "patterns"
                elif stripped_line.startswith("ROOT_CAUSES:"):
                    current_section = "causes"
                elif stripped_line.startswith("TRIGGERS:"):
                    current_section = "triggers"
                elif stripped_line.startswith("- ") and current_section:
                    item = stripped_line[2:].strip()
                    if current_section == "patterns":
                        all_patterns.append(item)
                    elif current_section == "causes":
                        all_root_causes.append(item)
                    elif current_section == "triggers":
                        all_triggers.append(item)

        return {
            "frustration_patterns": list(set(all_patterns)),
            "root_causes": list(set(all_root_causes)),
            "triggers": list(set(all_triggers)),
        }

    async def analyze_user_preferences_patterns(
        self, user_preferences: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user preference patterns using optimized LLM batch processing."""
        print("Analyzing user preference patterns...")

        # Sample users if too many
        if len(user_preferences) > MAX_USERS_FOR_PREFERENCE_ANALYSIS:
            import random

            sampled_users = dict(
                random.sample(list(user_preferences.items()), MAX_USERS_FOR_PREFERENCE_ANALYSIS)
            )
            print(
                f"Sampling {MAX_USERS_FOR_PREFERENCE_ANALYSIS} users from {len(user_preferences)} for preference analysis"
            )
            user_preferences = sampled_users

        # Create consolidated preference summaries
        all_descriptions = []
        all_preferences = []
        intent_summary: Dict[str, int] = {}
        emotion_summary: Dict[str, int] = {}

        for _, prefs in user_preferences.items():
            # Collect descriptions
            if prefs["overall_description"]:
                all_descriptions.append(prefs["overall_description"][:200])

            # Collect preferences
            all_preferences.extend(prefs["preference_summary"][:3])  # Top 3 per user

            # Aggregate intent and emotion distributions
            for intent, count in prefs["intent_distribution"].items():
                intent_summary[intent] = intent_summary.get(intent, 0) + count
            for emotion, count in prefs["emotion_distribution"].items():
                emotion_summary[emotion] = emotion_summary.get(emotion, 0) + count

        # Single comprehensive analysis call
        top_intents = sorted(intent_summary.items(), key=lambda x: x[1], reverse=True)[:10]
        top_emotions = sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True)[:10]
        unique_preferences = list(set(all_preferences))[:20]  # Top 20 unique preferences

        prompt = f"""
Analyze user preferences and behaviors from {len(user_preferences)} coding assistant users.

TOP USER INTENTS: {[f"{intent}({count})" for intent, count in top_intents]}
TOP EMOTIONS: {[f"{emotion}({count})" for emotion, count in top_emotions]}

COMMON PREFERENCES: {unique_preferences[:15]}

SAMPLE USER DESCRIPTIONS:
{chr(10).join(all_descriptions[:10])}

Provide analysis in this exact format:

COMMUNICATION_PATTERNS:
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]

WORKFLOW_PREFERENCES:
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]

TECHNICAL_PREFERENCES:
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]
"""

        result = await self.call_llm(prompt)
        return self._parse_preference_analysis(result)

    def _parse_preference_analysis(self, result: str) -> Dict[str, Any]:
        """Parse preference analysis result."""
        analysis: Dict[str, List[str]] = {
            "communication_patterns": [],
            "workflow_preferences": [],
            "technical_preferences": [],
        }

        if not result or result == "LLM not configured":
            return analysis

        result_lines = result.split("\n")
        current_section = None

        for line in result_lines:
            stripped_line = line.strip()
            if stripped_line.startswith("COMMUNICATION_PATTERNS:"):
                current_section = "communication_patterns"
            elif stripped_line.startswith("WORKFLOW_PREFERENCES:"):
                current_section = "workflow_preferences"
            elif stripped_line.startswith("TECHNICAL_PREFERENCES:"):
                current_section = "technical_preferences"
            elif stripped_line.startswith("- ") and current_section:
                analysis[current_section].append(stripped_line[2:].strip())

        return analysis

    async def generate_best_practices(
        self, frustration_analysis: Dict[str, Any], preference_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable best practices using a single comprehensive LLM call."""
        print("Generating best practices for user interaction...")

        # Create comprehensive input for single analysis
        frustration_summary = {
            "patterns": frustration_analysis["frustration_patterns"][:8],
            "causes": frustration_analysis["root_causes"][:8],
            "triggers": frustration_analysis["triggers"][:8],
        }

        preference_summary = {
            "communication": preference_analysis["communication_patterns"][:8],
            "workflow": preference_analysis["workflow_preferences"][:8],
            "technical": preference_analysis["technical_preferences"][:8],
        }

        prompt = f"""
Generate specific best practices for coding assistants based on user frustration and preference analysis.

FRUSTRATION ANALYSIS:
- Patterns: {'; '.join(frustration_summary['patterns'])}
- Root Causes: {'; '.join(frustration_summary['causes'])}
- Triggers: {'; '.join(frustration_summary['triggers'])}

PREFERENCE ANALYSIS:
- Communication: {'; '.join(preference_summary['communication'])}
- Workflow: {'; '.join(preference_summary['workflow'])}
- Technical: {'; '.join(preference_summary['technical'])}

Provide actionable best practices in this exact format:

SYSTEM_PROMPT_GUIDELINES:
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

COMMUNICATION_BEST_PRACTICES:
- [Practice 1]
- [Practice 2]
- [Practice 3]

WORKFLOW_OPTIMIZATION:
- [Optimization 1]
- [Optimization 2]
- [Optimization 3]

FRUSTRATION_PREVENTION:
- [Prevention 1]
- [Prevention 2]
- [Prevention 3]

PROACTIVE_ASSISTANCE:
- [Assistance 1]
- [Assistance 2]
- [Assistance 3]
"""

        result = await self.call_llm(prompt)
        return self._parse_best_practices(result)

    def _parse_best_practices(self, result: str) -> Dict[str, List[str]]:
        """Parse best practices result."""
        best_practices: Dict[str, List[str]] = {
            "system_prompt_guidelines": [],
            "communication_best_practices": [],
            "workflow_optimization": [],
            "frustration_prevention": [],
            "proactive_assistance": [],
        }

        if not result or result == "LLM not configured":
            return best_practices

        result_lines = result.split("\n")
        current_section = None

        for line in result_lines:
            stripped_line = line.strip()
            if stripped_line.startswith("SYSTEM_PROMPT_GUIDELINES:"):
                current_section = "system_prompt_guidelines"
            elif stripped_line.startswith("COMMUNICATION_BEST_PRACTICES:"):
                current_section = "communication_best_practices"
            elif stripped_line.startswith("WORKFLOW_OPTIMIZATION:"):
                current_section = "workflow_optimization"
            elif stripped_line.startswith("FRUSTRATION_PREVENTION:"):
                current_section = "frustration_prevention"
            elif stripped_line.startswith("PROACTIVE_ASSISTANCE:"):
                current_section = "proactive_assistance"
            elif stripped_line.startswith("- ") and current_section:
                best_practices[current_section].append(stripped_line[2:].strip())

        return best_practices

    def save_comprehensive_analysis(self, analysis_data: AnalysisData) -> None:
        """Save comprehensive analysis results."""

        # Save negative messages analysis
        negative_file_path = os.path.join(self.output_path, "negative_sentiment_analysis.json")
        with open(negative_file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_users_with_negative_messages": len(
                        [u for u in analysis_data.negative_messages.values() if u]
                    ),
                    "total_negative_messages": sum(
                        len(msgs) for msgs in analysis_data.negative_messages.values()
                    ),
                    "negative_messages_by_user": analysis_data.negative_messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save user preferences analysis
        preferences_file_path = os.path.join(self.output_path, "user_preferences_analysis.json")
        with open(preferences_file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_users_analyzed": len(analysis_data.user_preferences),
                    "user_preferences": analysis_data.user_preferences,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save pattern analysis
        patterns_file_path = os.path.join(self.output_path, "behavior_patterns_analysis.json")
        with open(patterns_file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "frustration_analysis": analysis_data.frustration_analysis,
                    "preference_analysis": analysis_data.preference_analysis,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save best practices
        best_practices_file_path = os.path.join(
            self.output_path, "best_practices_for_system_prompt.json"
        )
        with open(best_practices_file_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data.best_practices, f, indent=2, ensure_ascii=False)

        print(f"Comprehensive analysis saved to: {self.output_path}")

    async def generate_system_prompt_additions(self, best_practices: Dict[str, Any]) -> str:
        """Generate specific additions for system prompts."""
        print("Generating system prompt additions...")

        prompt = f"""
Based on these best practices analysis, create specific text that should be added to coding assistant system prompts to improve user experience:

BEST PRACTICES:
System Prompt Guidelines: {', '.join(best_practices.get('system_prompt_guidelines', []))}
Communication: {', '.join(best_practices.get('communication_best_practices', []))}
Workflow: {', '.join(best_practices.get('workflow_optimization', []))}
Frustration Prevention: {', '.join(best_practices.get('frustration_prevention', []))}
Proactive Assistance: {', '.join(best_practices.get('proactive_assistance', []))}

Generate a coherent system prompt addition that incorporates these insights. The text should be:
1. Directly actionable for a coding assistant
2. Focused on improving user experience
3. Preventing common frustrations
4. Adapting to user preferences
5. Written in clear, directive language

Format it as a system prompt section that can be directly added to existing prompts.
"""

        result = await self.call_llm(prompt)
        return (
            result
            if result != "LLM not configured"
            else "LLM not configured - cannot generate system prompt"
        )

    async def create_comprehensive_summary(
        self, analysis_data: AnalysisData, system_prompt_addition: str
    ) -> None:
        """Create a comprehensive summary report."""
        summary_path = os.path.join(self.output_path, "comprehensive_user_behavior_analysis.md")

        total_users = len(analysis_data.user_preferences)
        users_with_negative = len([u for u in analysis_data.negative_messages.values() if u])
        total_negative_messages = sum(
            len(msgs) for msgs in analysis_data.negative_messages.values()
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Comprehensive User Behavior Analysis\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Users Analyzed**: {total_users}\n")
            f.write(
                f"- **Users with Negative Experiences**: {users_with_negative} ({users_with_negative/total_users*100:.1f}%)\n"
            )
            f.write(f"- **Total Negative Messages**: {total_negative_messages}\n\n")

            f.write("## Frustration Analysis\n\n")
            f.write("### Common Frustration Patterns\n")
            for pattern in analysis_data.frustration_analysis.get("frustration_patterns", [])[:10]:
                f.write(f"- {pattern}\n")

            f.write("\n### Root Causes of Frustration\n")
            for cause in analysis_data.frustration_analysis.get("root_causes", [])[:10]:
                f.write(f"- {cause}\n")

            f.write("\n### Frustration Triggers\n")
            for trigger in analysis_data.frustration_analysis.get("triggers", [])[:10]:
                f.write(f"- {trigger}\n")

            f.write("\n## User Preference Patterns\n\n")
            f.write("### Communication Patterns\n")
            for pattern in analysis_data.preference_analysis.get("communication_patterns", [])[:10]:
                f.write(f"- {pattern}\n")

            f.write("\n### Workflow Preferences\n")
            for pattern in analysis_data.preference_analysis.get("workflow_preferences", [])[:10]:
                f.write(f"- {pattern}\n")

            f.write("\n### Technical Preferences\n")
            for pattern in analysis_data.preference_analysis.get("technical_preferences", [])[:10]:
                f.write(f"- {pattern}\n")

            f.write("\n## Best Practices for Coding Assistants\n\n")
            f.write("### System Prompt Guidelines\n")
            for guideline in analysis_data.best_practices.get("system_prompt_guidelines", []):
                f.write(f"- {guideline}\n")

            f.write("\n### Communication Best Practices\n")
            for practice in analysis_data.best_practices.get("communication_best_practices", []):
                f.write(f"- {practice}\n")

            f.write("\n### Workflow Optimization\n")
            for optimization in analysis_data.best_practices.get("workflow_optimization", []):
                f.write(f"- {optimization}\n")

            f.write("\n### Frustration Prevention\n")
            for prevention in analysis_data.best_practices.get("frustration_prevention", []):
                f.write(f"- {prevention}\n")

            f.write("\n### Proactive Assistance\n")
            for assistance in analysis_data.best_practices.get("proactive_assistance", []):
                f.write(f"- {assistance}\n")

            f.write("\n## Recommended System Prompt Addition\n\n")
            f.write("```\n")
            f.write(system_prompt_addition)
            f.write("\n```\n\n")

            f.write("---\n")
            f.write(
                "*This analysis was generated from user interaction data to improve coding assistant performance and user satisfaction.*\n"
            )

        print(f"Comprehensive summary saved to: {summary_path}")

    async def run_comprehensive_analysis(self) -> None:
        """Run the complete comprehensive analysis."""
        print("Starting comprehensive user behavior analysis...")

        # Extract negative messages
        negative_messages = self.extract_negative_messages()

        # Extract user preferences
        user_preferences = self.extract_user_preferences()

        if not user_preferences:
            print("No user preference data found.")
            return

        # Analyze frustration patterns
        frustration_analysis = await self.analyze_frustration_patterns(negative_messages)

        # Analyze preference patterns
        preference_analysis = await self.analyze_user_preferences_patterns(user_preferences)

        # Generate best practices
        best_practices = await self.generate_best_practices(
            frustration_analysis, preference_analysis
        )

        # Generate system prompt additions
        system_prompt_addition = await self.generate_system_prompt_additions(best_practices)

        # Create analysis data container
        analysis_data = AnalysisData(
            negative_messages=negative_messages,
            user_preferences=user_preferences,
            frustration_analysis=frustration_analysis,
            preference_analysis=preference_analysis,
            best_practices=best_practices,
        )

        # Save all results
        self.save_comprehensive_analysis(analysis_data)

        # Create comprehensive summary
        await self.create_comprehensive_summary(analysis_data, system_prompt_addition)

        print(f"\nComprehensive analysis complete! Results saved to: {self.output_path}")
        print("Generated files:")
        print("  - negative_sentiment_analysis.json")
        print("  - user_preferences_analysis.json")
        print("  - behavior_patterns_analysis.json")
        print("  - best_practices_for_system_prompt.json")
        print("  - comprehensive_user_behavior_analysis.md")

        # Print key insights
        print("\nKEY INSIGHTS:")
        print(f"Total users analyzed: {len(user_preferences):,}")
        users_with_negative = len([u for u in negative_messages.values() if u])
        print(
            f"Users with negative experiences: {users_with_negative} ({users_with_negative/len(user_preferences)*100:.1f}%)"
        )
        print(f"Total negative messages: {sum(len(msgs) for msgs in negative_messages.values()):,}")


async def main():
    """Main function to run the comprehensive analysis."""
    print("Enhanced User Behavior Analyzer with Optimized LLM Processing")
    print("=" * 60)

    # For faster testing, use sampling. Remove these limits for full analysis
    analyzer = EnhancedUserBehaviorAnalyzer()
    analyzer.set_batch_size(50)  # Larger batches for efficiency
    analyzer.set_sampling_parameters(
        sample_users=20,  # Analyze 20 users for faster testing
        sample_messages_per_user=10,  # Max 10 negative messages per user
    )

    print("Configuration:")
    print(f"- Sample Users: {analyzer.sample_users}")
    print(f"- Max Messages per User: {analyzer.sample_messages_per_user}")
    print(f"- LLM Batch Size: {analyzer.batch_size}")
    print(f"- Model: {analyzer.model}")
    print()

    await analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    asyncio.run(main())
