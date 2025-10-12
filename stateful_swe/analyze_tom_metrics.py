#!/usr/bin/env python3
"""
Tom Metrics Analysis Script

Analyzes the tom_metrics.json file and replicates the statistics format
for Tom agent usage, consultations, and sleeptime analysis.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_tom_metrics(file_path: str) -> None:
    """Analyze Tom metrics and display statistics."""

    print("📊 Loading Tom Metrics Data...")

    # Load the JSON data
    with open(file_path, "r") as f:
        data = json.load(f)

    raw_events = data.get("raw_events", [])
    print(f"📈 Loaded {len(raw_events)} events")

    # Count event types
    event_counts = Counter()
    consultation_events = []
    sleeptime_events = []
    distinct_users = set()
    unique_sessions = set()
    user_sessions = defaultdict(set)

    for event in raw_events:
        event_type = event.get("event", "")
        event_counts[event_type] += 1

        # Collect user IDs
        distinct_id = event.get("distinct_id", "")
        if distinct_id:
            distinct_users.add(distinct_id)

        # Extract session information
        session_id = event.get("properties", {}).get("session_id", "")
        if session_id:
            unique_sessions.add(session_id)
            if distinct_id:
                user_sessions[distinct_id].add(session_id)

        # Collect consultation events
        if event_type == "tom_consult_agent_interaction":
            consultation_events.append(event)

        # Collect sleeptime events
        elif event_type == "tom_sleeptime_triggered":
            sleeptime_events.append(event)

    # Analyze consultations
    consultation_analysis = analyze_consultations(consultation_events)

    # Analyze sleeptime usage
    sleeptime_analysis = analyze_sleeptime(sleeptime_events)

    # Analyze session distribution
    session_analysis = analyze_session_distribution(user_sessions, unique_sessions)

    # Display results in target format
    display_results(
        event_counts,
        consultation_analysis,
        sleeptime_analysis,
        distinct_users,
        session_analysis,
    )


def analyze_consultations(consultation_events: list) -> dict:
    """Analyze consultation acceptance patterns."""

    total = len(consultation_events)
    accepted = 0
    partially_accepted = 0
    rejected = 0

    for event in consultation_events:
        accepted_value = event.get("properties", {}).get("accepted", 0)

        if accepted_value == 1:
            accepted += 1
        elif accepted_value == 0.5:
            partially_accepted += 1
        elif accepted_value == 0:
            rejected += 1

    acceptance_rate = (accepted / total * 100) if total > 0 else 0

    return {
        "total": total,
        "accepted": accepted,
        "partially_accepted": partially_accepted,
        "rejected": rejected,
        "acceptance_rate": acceptance_rate,
    }


def analyze_sleeptime(sleeptime_events: list) -> dict:
    """Analyze sleeptime usage patterns."""

    # Count sleeptime events per user
    user_sleeptime_counts = defaultdict(int)

    for event in sleeptime_events:
        distinct_id = event.get("distinct_id", "")
        if distinct_id:
            user_sleeptime_counts[distinct_id] += 1

    total_triggers = len(sleeptime_events)
    unique_users = len(user_sleeptime_counts)
    avg_calls_per_user = total_triggers / unique_users if unique_users > 0 else 0

    return {
        "total_triggers": total_triggers,
        "unique_users": unique_users,
        "avg_calls_per_user": avg_calls_per_user,
    }


def analyze_session_distribution(user_sessions: dict, unique_sessions: set) -> dict:
    """Analyze session distribution across users."""

    # Calculate sessions per user statistics
    sessions_per_user = [len(sessions) for sessions in user_sessions.values()]

    if not sessions_per_user:
        return {
            "total_unique_sessions": 0,
            "users_with_sessions": 0,
            "avg_sessions_per_user": 0,
            "min_sessions": 0,
            "max_sessions": 0,
            "session_distribution": {},
        }

    # Session count distribution
    session_distribution = Counter(sessions_per_user)

    return {
        "total_unique_sessions": len(unique_sessions),
        "users_with_sessions": len(user_sessions),
        "avg_sessions_per_user": sum(sessions_per_user) / len(sessions_per_user),
        "min_sessions": min(sessions_per_user),
        "max_sessions": max(sessions_per_user),
        "session_distribution": dict(session_distribution),
    }


def display_results(
    event_counts: Counter,
    consultation_analysis: dict,
    sleeptime_analysis: dict,
    distinct_users: set,
    session_analysis: dict,
) -> None:
    """Display results in the target format."""

    print("\n🎯 Tom event types:")
    for event_type in [
        "tom_agent_initialized",
        "tom_consult_agent_interaction",
        "tom_sleeptime_triggered",
    ]:
        count = event_counts.get(event_type, 0)
        print(f"  {event_type}: {count}")

    print("\n💬 CONSULTATION ANALYSIS:")
    c = consultation_analysis
    print(f"  Total consultations: {c['total']}")
    print(f"  ✅ Accepted: {c['accepted']} ({c['accepted']/c['total']*100:.1f}%)")
    print(
        f"  ⚠️  Partially accepted: {c['partially_accepted']} ({c['partially_accepted']/c['total']*100:.1f}%)"
    )
    print(f"  ❌ Rejected: {c['rejected']} ({c['rejected']/c['total']*100:.1f}%)")
    print(f"  📊 Overall acceptance rate: {c['acceptance_rate']:.1f}%")

    print("\n⏰ SLEEPTIME USAGE ANALYSIS:")
    s = sleeptime_analysis
    print(f"  Total sleeptime triggers: {s['total_triggers']}")
    print(f"  Users who used sleeptime: {s['unique_users']}")
    print(f"  Total users: {len(distinct_users)}")

    # Calculate adoption rate
    adoption_rate = (
        (s["unique_users"] / len(distinct_users) * 100)
        if len(distinct_users) > 0
        else 0
    )
    print(f"  📊 Sleeptime adoption rate: {adoption_rate:.1f}%")
    print(f"  📈 Avg sleeptime calls per user: {s['avg_calls_per_user']:.1f}")

    print("\n📊 SESSION ANALYSIS:")
    se = session_analysis
    print(f"  Total unique sessions: {se['total_unique_sessions']}")
    print(f"  Users with sessions: {se['users_with_sessions']}")
    print(f"  Avg sessions per user: {se['avg_sessions_per_user']:.1f}")
    print(f"  Min sessions per user: {se['min_sessions']}")
    print(f"  Max sessions per user: {se['max_sessions']}")

    print("\n📈 SESSION DISTRIBUTION:")
    distribution = se["session_distribution"]
    if distribution:
        # Sort by number of sessions for cleaner display
        sorted_dist = sorted(distribution.items())
        for session_count, user_count in sorted_dist:
            print(
                f"  {user_count} users with {session_count} session{'s' if session_count != 1 else ''}"
            )

    print("\n👥 USER OVERVIEW:")
    print(f"  Total unique users: {len(distinct_users)}")

    # Count users who consulted Tom (have consultation events)
    consultation_users = set()
    with open(
        "/Users/xuhuizhou/Projects/ToM-SWE/data/user_study_v1/tom_metrics.json", "r"
    ) as f:
        data = json.load(f)

    for event in data.get("raw_events", []):
        if event.get("event") == "tom_consult_agent_interaction":
            distinct_id = event.get("distinct_id", "")
            if distinct_id:
                consultation_users.add(distinct_id)

    print(f"  Users who consulted Tom: {len(consultation_users)}")
    print(f"  Users who used sleeptime: {s['unique_users']}")

    print("\n📋 ALL TOM EVENTS:")
    for event_type, count in event_counts.most_common():
        print(f"  {event_type}: {count}")


def main():
    """Main entry point."""
    file_path = "/Users/xuhuizhou/Projects/ToM-SWE/data/user_study_v1/tom_metrics.json"

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    analyze_tom_metrics(file_path)
    print("\n💾 Analysis complete!")
    print(f"📅 Source: {file_path}")


if __name__ == "__main__":
    main()
