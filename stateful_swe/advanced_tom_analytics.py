#!/usr/bin/env python3
"""
Simplified Tom Analytics

Analyzes tom_metrics.json to show consolidated query type analysis.
"""

import json
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import List

from tom_swe.generation.generate import LLMClient, LLMConfig
from tom_swe.generation.dataclass import QueryClassification


def create_llm_client() -> LLMClient:
    """Create and configure LLM client for query classification."""
    config = LLMConfig(
        model="gpt-5",  # Use GPT-5 for better classification accuracy
        fallback_model="gpt-5-nano",
    )
    return LLMClient(config)


async def classify_query_with_llm(
    llm_client: LLMClient, query: str
) -> QueryClassification:
    """Classify a single query using LLM with structured output."""

    classification_prompt = f"""You are an expert at categorizing software development queries. Please classify the following SWE agent consultation query into one of these categories:

**Categories:**
1. **code_understanding**: Understanding and analyzing existing code/systems - queries about reading, reviewing, explaining, or analyzing existing code, documentation, or system behavior
2. **development**: Building new features, system architecture, and deployment setup - queries about implementing, creating, designing, planning, or building new functionality
3. **troubleshooting**: Fixing issues, testing, and resolving environment problems - queries about debugging, fixing bugs, testing, configuration issues, or resolving problems
4. **other**: Queries that don't clearly fit the above categories

**Examples:**
- "I need to understand how this authentication system works" → code_understanding
- "Let's implement a new user registration feature" → development
- "The tests are failing with import errors" → troubleshooting
- "What should we work on next?" → other

**Query to classify:**
"{query}..."

Please provide:
1. The most appropriate category
2. A confidence score (0.0-1.0) for your classification
3. Brief reasoning for your choice

Consider the semantic meaning and context, not just keywords."""

    try:
        result = await llm_client.call_structured_async(
            prompt=classification_prompt, output_type=QueryClassification
        )
        return result
    except Exception as e:
        # Fallback classification if LLM fails
        print(f"LLM classification failed for query, using fallback: {e}")
        return QueryClassification(
            category="other",
            confidence=0.1,
            reasoning="LLM classification failed, using fallback",
        )


async def classify_queries_batch(queries: List[str]) -> List[QueryClassification]:
    """Classify multiple queries using LLM with batch processing."""
    llm_client = create_llm_client()

    print(f"🤖 Classifying {len(queries)} queries with LLM...")

    # Process queries in smaller batches with concurrent processing
    batch_size = 200  # Reduced batch size for faster processing
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        print(
            f"   Processing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}"
        )

        # Create async tasks for concurrent processing
        tasks = [classify_query_with_llm(llm_client, query) for query in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    # Note: We're only processing a subset for testing
    print(f"   ✅ Processed {len(results)} queries with LLM classification")

    return results


def load_consultation_events(file_path: str) -> list:
    """Load and parse consultation events from Tom metrics data."""
    with open(file_path, "r") as f:
        data = json.load(f)

    raw_events = data.get("raw_events", [])
    consultation_events = []

    for event in raw_events:
        if event.get("event") == "tom_consult_agent_interaction":
            consultation_events.append(event)

    # Sort consultation events by timestamp
    consultation_events.sort(key=lambda x: x.get("properties", {}).get("timestamp", ""))

    return consultation_events


async def categorize_queries_with_llm(consultation_events: list) -> None:
    """Analyze and display consolidated query type analysis using LLM classification."""
    print("📊 Consolidated Query Type Analysis (LLM-Powered):")
    print("🎯 3 Major Categories + Other")
    print("-" * 60)

    # Category descriptions for display
    category_descriptions = {
        "code_understanding": "Understanding and analyzing existing code/systems",
        "development": "Building new features, system architecture, and deployment setup",
        "troubleshooting": "Fixing issues, testing, and resolving environment problems",
        "other": "General queries that don't fit the main categories",
    }

    # Extract queries for classification
    queries = []
    events_data = []

    for event in consultation_events:
        original = event.get("properties", {}).get("original", "")
        accepted = event.get("properties", {}).get("accepted", 0)

        queries.append(original)
        events_data.append({"original": original, "accepted": accepted, "event": event})

    # Get LLM classifications (limited subset for testing)
    classifications = await classify_queries_batch(queries)

    # Organize results by category
    categorized = defaultdict(list)
    category_stats = defaultdict(
        lambda: {
            "accepted": 0,
            "partial": 0,
            "rejected": 0,
            "accepted_examples": [],
            "rejected_examples": [],
            "partial_examples": [],
            "confidence_scores": [],
        }
    )

    # Process only the number of classifications we got
    processed_count = min(len(events_data), len(classifications))

    for i in range(processed_count):
        event_data = events_data[i]
        classification = classifications[i]
        original = event_data["original"]
        accepted = event_data["accepted"]
        category = classification.category

        # Store categorized query with LLM confidence
        query_data = {
            "accepted": accepted,
            "original": original[:200] + "..." if len(original) > 200 else original,
            "llm_confidence": classification.confidence,
            "reasoning": classification.reasoning,
        }
        categorized[category].append(query_data)
        category_stats[category]["confidence_scores"].append(classification.confidence)

        # Update category stats and collect examples by type
        if accepted == 1:
            category_stats[category]["accepted"] += 1
            if len(category_stats[category]["accepted_examples"]) < 5:
                category_stats[category]["accepted_examples"].append(
                    original[:150] + "..." if len(original) > 150 else original
                )
        elif accepted == 0.5:
            category_stats[category]["partial"] += 1
            if len(category_stats[category]["partial_examples"]) < 5:
                category_stats[category]["partial_examples"].append(
                    original[:150] + "..." if len(original) > 150 else original
                )
        else:
            category_stats[category]["rejected"] += 1
            if len(category_stats[category]["rejected_examples"]) < 5:
                category_stats[category]["rejected_examples"].append(
                    original[:150] + "..." if len(original) > 150 else original
                )

    # Sort categories by acceptance rate
    sorted_categories = []
    for category, queries in categorized.items():
        if len(queries) > 0:
            accepted = sum(1 for q in queries if q["accepted"] >= 0.5)
            acceptance_rate = accepted / len(queries)
            avg_confidence = sum(category_stats[category]["confidence_scores"]) / len(
                category_stats[category]["confidence_scores"]
            )
            sorted_categories.append(
                (
                    category,
                    len(queries),
                    acceptance_rate,
                    category_stats[category],
                    avg_confidence,
                )
            )

    sorted_categories.sort(key=lambda x: x[2], reverse=True)

    # Display results
    for category, total, acceptance_rate, stats, avg_confidence in sorted_categories:
        print(f"\n🎯 {category.upper().replace('_', ' ')}:")

        # Add category description
        if category in category_descriptions:
            print(f"   📋 {category_descriptions[category]}")

        print(f"   📊 Total queries: {total}")
        print(f"   🤖 Avg LLM confidence: {avg_confidence:.2f}")
        print(
            f"   ✅ Accepted: {stats['accepted']} ({stats['accepted']/total*100:.1f}%)"
        )
        print(f"   ⚠️  Partial: {stats['partial']} ({stats['partial']/total*100:.1f}%)")
        print(
            f"   ❌ Rejected: {stats['rejected']} ({stats['rejected']/total*100:.1f}%)"
        )
        print(f"   📈 Overall success rate: {acceptance_rate:.1%}")

        # Show detailed examples for this category (5 positive, 5 negative)
        print("   📝 SUCCESSFUL EXAMPLES (✅ Accepted):")
        accepted_examples = stats["accepted_examples"]
        if not accepted_examples:
            accepted_examples = stats[
                "partial_examples"
            ]  # Fallback to partial if no accepted

        for i, text in enumerate(accepted_examples[:5], 1):
            # Extract user message after "user's message:" if present
            if "user's message:" in text:
                text = text.split("user's message:")[-1].strip()
            print(f'      {i}. "{text}"')

        if not accepted_examples:
            print("      (No successful examples found)")

        print("   📝 FAILED EXAMPLES (❌ Rejected):")
        rejected_examples = stats["rejected_examples"]
        for i, text in enumerate(rejected_examples[:5], 1):
            # Extract user message after "user's message:" if present
            if "user's message:" in text:
                text = text.split("user's message:")[-1].strip()
            print(f'      {i}. "{text}"')

        if not rejected_examples:
            print("      (No rejected examples found)")


def show_rejection_analysis(consultation_events: list) -> None:
    """Show basic rejection pattern analysis."""
    print("\n❌ REJECTION PATTERN ANALYSIS")
    print("=" * 50)

    # Get basic stats
    rejected = [
        e
        for e in consultation_events
        if e.get("properties", {}).get("accepted", 0) == 0
    ]
    partial = [
        e
        for e in consultation_events
        if e.get("properties", {}).get("accepted", 0) == 0.5
    ]
    accepted = [
        e
        for e in consultation_events
        if e.get("properties", {}).get("accepted", 0) == 1
    ]

    print("📊 Detailed Rejection Overview:")
    total = len(consultation_events)
    print(f"  Total consultations: {total}")
    print(f"  ✅ Fully accepted: {len(accepted)} ({len(accepted)/total*100:.1f}%)")
    print(f"  ⚠️  Partially accepted: {len(partial)} ({len(partial)/total*100:.1f}%)")
    print(f"  ❌ Fully rejected: {len(rejected)} ({len(rejected)/total*100:.1f}%)")
    print(
        f"  📈 Success rate (full + partial): {(len(accepted) + len(partial))/total*100:.1f}%"
    )


async def main():
    """Main entry point."""
    file_path = "/Users/xuhuizhou/Projects/ToM-SWE/data/user_study_v1/tom_metrics.json"

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    print("📊 Loading Tom Metrics Data...")
    consultation_events = load_consultation_events(file_path)
    print(f"📈 Loaded {len(consultation_events)} consultation events")

    # Show LLM-powered consolidated query analysis
    await categorize_queries_with_llm(consultation_events)

    # Show basic rejection analysis
    show_rejection_analysis(consultation_events)

    print("\n💾 Analysis complete!")
    print(f"📅 Source: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
