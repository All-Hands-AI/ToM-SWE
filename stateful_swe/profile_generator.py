#!/usr/bin/env python3
"""
Real User-Based Profile Generator for Stateful SWE Benchmark

Creates realistic user profiles with specific interaction preferences and 75 concrete coding preferences
derived from analysis of real power user behavior patterns.
"""

import json
import random
from typing import List, Dict, Any, TypedDict
from pathlib import Path


class ProfileConfig(TypedDict):
    """Type definition for profile configuration dictionary."""

    profile_id: str
    verbosity: str
    question_timing: str
    response_style: str
    coding_preferences: List[str]


class ConcreteUserProfile:
    """Represents a realistic user with specific preferences and coding standards."""

    def __init__(
        self,
        profile_id: str,
        verbosity: str,
        question_timing: str,
        response_style: str,
        coding_preferences: List[str],
    ):
        self.profile_id = profile_id
        self.verbosity = verbosity
        self.question_timing = question_timing
        self.response_style = response_style
        self.coding_preferences = coding_preferences

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "interaction_preferences": {
                "verbosity": self.verbosity,
                "question_timing": self.question_timing,
                "response_style": self.response_style,
            },
            "coding_preferences": self.coding_preferences,
            "user_roleplay_prompt": self.get_roleplay_prompt(),
        }

    def get_roleplay_prompt(self) -> str:
        """Generate roleplay prompt for LLM to simulate this user type."""
        verbosity_desc = {
            "concise": "You prefer brief, to-the-point responses. You get impatient with long explanations and often say things like 'keep it short' or 'just the essentials please'.",
            "verbose": "You appreciate detailed explanations and comprehensive responses. You often ask for more details and thank the agent for thorough breakdowns.",
        }

        timing_desc = {
            "upfront": "You prefer to ask all your clarifying questions at the beginning before any work starts. You like to understand the full scope upfront. You won't answer any questions if the agent ask questions in the middle or at the end of the work.",
            "ongoing": "You're comfortable with questions being asked throughout the process as they arise. You prefer iterative clarification.",
        }

        response_desc = {
            "short_response": "You respond concisely and to the point. Your answers for the SWE agent are usually under 15 words. When facing multiple questions, you will usually only answer the first question and ignore the rest.",
            "verbose_response": "You could provide more detailed answers for the SWE agent. You are willing to answer more than one question from the SWE agent.",
        }

        prefs_desc = "You have specific coding preferences: " + "; ".join(
            self.coding_preferences
        )

        return f"""You are roleplaying as a software developer with these characteristics:

INTERACTION STYLE:
- {verbosity_desc[self.verbosity]}
- {timing_desc[self.question_timing]}
- {response_desc[self.response_style]}

CODING STANDARDS:
- {prefs_desc}

Respond naturally as this type of user would, incorporating these preferences into your messages. Be authentic to this persona while working with the SWE agent."""


class RealUserProfileGenerator:
    """Generates concrete, realistic user profiles for the benchmark."""

    def __init__(self):
        self.verbosity_options = ["concise", "verbose"]
        self.question_timing_options = ["upfront", "ongoing"]

        # 75 concrete coding preferences extracted from real power user data
        self.coding_preference_pool = [
            # Git Workflow & Version Control (12)
            "Use descriptive branch names like 'feature/user-auth' or 'DAISY-1046'",
            "Always use exact same branch name when updating existing work",
            "Push changes without creating PR when specified - maintain explicit control",
            "Separate git push operations from PR/MR creation for better workflow control",
            "Follow PR templates when available rather than ad-hoc descriptions",
            "Clean up merged branches regularly to maintain repository hygiene",
            "Use develop branch as primary development branch instead of main/master",
            "Use rebasing over merging to maintain clean git history",
            "Write descriptive commit messages explaining the 'why' not just 'what'",
            "Handle pushing changes to multiple repositories simultaneously when needed",
            "Be comfortable with force push for updating existing PRs after rebase",
            # Code Quality & Testing (15)
            "Implement comprehensive test coverage: unit, integration, and E2E tests",
            "Create tests before or alongside implementation, not as afterthought",
            "Fix failing tests immediately - prioritize over new feature development",
            "Design tests to be environment-independent and reproducible",
            "Use real implementations over mock/placeholder data in production code",
            "Enforce strict TypeScript compliance and comprehensive type checking",
            "Integrate automated linting (ESLint, Biome, Ruff) in development workflow",
            "Use pre-commit hooks (Husky) for automated quality checks",
            "Ensure all CI/CD pipeline checks pass before considering work complete",
            "Test in containerized/Docker environments rather than local environments only",
            "Use factory patterns (factory_boy) for consistent test data generation",
            "Create specific regression tests for every previously fixed bug",
            # Architecture & Code Organization (18)
            "Centralize configuration management instead of scattered config files",
            "Maintain strict consistency between data models and database storage",
            "Organize API endpoints using consistent RESTful patterns and conventions",
            "Implement standardized error handling patterns across entire codebase",
            "Use environment variables for configuration over hardcoded values",
            "Maintain proper __all__ definitions and clean import organization",
            "In monorepos, maintain proper dependency separation between packages",
            "Design architecture for containerized deployment from initial implementation",
            "Use middleware patterns for authentication over scattered permission checks",
            "Implement proper state management (Redux, Zustand) instead of prop drilling",
            "Always create proper database migration scripts for schema changes",
            "Plan for API versioning from initial implementation, not as afterthought",
            "Implement comprehensive logging with consistent patterns and levels",
            "Ensure proper cleanup of resources, database connections, and temporary data",
            "Implement caching strategies for performance-critical operations",
            "Use async/await patterns consistently for all concurrent operations",
            # Technology Stack Preferences (15)
            "Prefer pnpm over npm/yarn for Node.js projects, Poetry over pip for Python",
            "Use FastAPI over Flask for Python APIs, Next.js for React applications",
            "Choose Biome over ESLint+Prettier for JavaScript/TypeScript linting and formatting",
            "Use PostgreSQL over MySQL with proper ORM patterns (SQLAlchemy, Django ORM)",
            "Implement JWT-based authentication with proper OIDC integration and session management",
            "Build React applications with TypeScript and proper component organization patterns",
            "Use httpx over requests library for Python HTTP clients with proper async support",
            "Create comprehensive markdown documentation with embedded Mermaid diagrams",
            "Set up VSCode with specific extensions and containerized development environments",
            "Use Vite for frontend builds with proper bundling and build optimization",
            "Prefer Pyright over mypy for Python type checking with strict TypeScript config",
            "Use pytest for Python testing with comprehensive async test support",
            "Implement Docker Compose for development, plan for Kubernetes in production",
            "Make explicit choices between GitLab vs GitHub based on specific workflow needs",
            "Implement OpenSSL-based PKI with proper certificate management and security practices",
            # Documentation & Communication (15)
            "Support multilingual documentation (e.g., German/English) when working internationally",
            "Create interactive documentation with collapsible sections and expandable code blocks",
            "Embed Mermaid diagrams directly in markdown for visual system documentation",
            "Write comprehensive API documentation with examples and complete response format specifications",
            "Provide step-by-step debugging and troubleshooting guides for common issues",
            "Create one-line installers and automated setup scripts for easy project onboarding",
            "Document system architecture with clear diagrams and comprehensive explanations",
            "Maintain clear contributing guidelines and development workflow documentation",
            "Keep proper changelog documentation for version tracking and release management",
            "Write minimal but meaningful code comments, prefer self-documenting code structure",
            "Structure README files with clear sections, navigation, and comprehensive project information",
            "Use privacy-conscious examples with properly anonymized sample data",
            "Include cross-references and links between related documentation sections",
            "Maintain version-specific documentation that matches current codebase state",
            "Ensure documentation accessibility for both human developers and AI development agents",
        ]

    def generate_15_profiles(self) -> List[ConcreteUserProfile]:
        """Generate 15 diverse, realistic user profiles based on real power user data."""
        profiles = []

        # Create 15 profiles covering all interaction combinations with diverse real coding preferences
        base_combinations = [
            ("concise", "upfront", "short_response"),
            ("concise", "ongoing", "short_response"),
            ("verbose", "upfront", "verbose_response"),
            ("verbose", "ongoing", "verbose_response"),
            ("verbose", "upfront", "short_response"),
            ("verbose", "ongoing", "short_response"),
            ("concise", "upfront", "verbose_response"),
            ("concise", "ongoing", "verbose_response"),
        ]

        profile_configs: List[ProfileConfig] = []

        for i in range(15):
            # Cycle through interaction combinations
            verbosity, timing, response_style = base_combinations[
                i % len(base_combinations)
            ]

            # Select 4-7 coding preferences from different categories for variety
            num_prefs = random.randint(5, 10)

            # Ensure each profile gets preferences from different categories for realistic diversity
            git_prefs = random.sample(
                self.coding_preference_pool[0:12], random.randint(1, 2)
            )
            quality_prefs = random.sample(
                self.coding_preference_pool[12:27], random.randint(1, 2)
            )
            arch_prefs = random.sample(
                self.coding_preference_pool[27:45], random.randint(1, 2)
            )
            tech_prefs = random.sample(
                self.coding_preference_pool[45:60], random.randint(0, 1)
            )
            doc_prefs = random.sample(
                self.coding_preference_pool[60:75], random.randint(0, 1)
            )

            selected_prefs = (
                git_prefs + quality_prefs + arch_prefs + tech_prefs + doc_prefs
            )

            # Add random additional preferences if needed
            if len(selected_prefs) < num_prefs:
                remaining_pool = [
                    p for p in self.coding_preference_pool if p not in selected_prefs
                ]
                additional = random.sample(
                    remaining_pool,
                    min(num_prefs - len(selected_prefs), len(remaining_pool)),
                )
                selected_prefs.extend(additional)

            # Truncate if we have too many
            selected_prefs = selected_prefs[:num_prefs]

            profile_config: ProfileConfig = {
                "profile_id": f"{verbosity}_{timing}_{response_style}_poweruser_{i+1:02d}",
                "verbosity": verbosity,
                "question_timing": timing,
                "response_style": response_style,
                "coding_preferences": selected_prefs,
            }
            profile_configs.append(profile_config)

        for config in profile_configs:
            profile = ConcreteUserProfile(
                profile_id=config["profile_id"],
                verbosity=config["verbosity"],
                response_style=config["response_style"],
                question_timing=config["question_timing"],
                coding_preferences=config["coding_preferences"],
            )
            profiles.append(profile)

        return profiles

    def save_profiles(
        self, profiles: List[ConcreteUserProfile], output_path: str
    ) -> None:
        """Save profiles to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for profile in profiles:
                profile_data = profile.to_dict()
                f.write(json.dumps(profile_data) + "\n")

        print(f"💾 Saved {len(profiles)} realistic user profiles to {output_path}")


def main():
    """Generate and save realistic user profiles based on real power user data."""
    generator = RealUserProfileGenerator()

    # Generate 15 realistic profiles based on actual user behavior analysis
    profiles = generator.generate_15_profiles()

    # Save to output directory
    output_path = "data/stateful_swe/user_profiles.jsonl"
    generator.save_profiles(profiles, output_path)

    # Print summary
    print("\n📊 Generated realistic user profiles summary:")
    for profile in profiles:
        print(f"  - {profile.profile_id}")
        print(
            f"    Interaction: {profile.verbosity} response, {profile.question_timing} questions, {profile.response_style}"
        )
        print(
            f"    Coding preferences: {len(profile.coding_preferences)} real user-based standards"
        )
        print()


if __name__ == "__main__":
    main()
