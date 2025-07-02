#!/usr/bin/env python3
"""
User Interaction Analysis Script

This script analyzes user interaction data from multiple sources:
- power_user_2025_05.csv: User emails and total sessions
- studio_results_20250604_1645.csv: GUI sessions and repository info
- processed_data/*.json: Detailed message analysis

Creates a comprehensive web table with user metrics and n-gram analysis.
"""

import json
import os
import re
from collections import Counter, defaultdict

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Constants for analysis
MIN_TOKEN_LENGTH = 2
MIN_NGRAM_LENGTH = 3
PROGRESS_UPDATE_INTERVAL = 10
MAX_PERCENTAGE = 100
MIN_SESSION_COUNT = 5
MIN_MESSAGE_LENGTH = 100
MAX_LINE_LENGTH = 100
MIN_REPO_COUNT = 5


def categorize_user_message(content):
    """
    Categorize user messages into two types:
    1. user_typed: Regular user messages
    2. micro_agent_triggered: Messages containing system-generated tags

    Args:
        content (str): The message content

    Returns:
        str: Either 'user_typed' or 'micro_agent_triggered'
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

    # Check if any system tags are present in the content
    for tag in system_tags:
        if tag in content:
            return "micro_agent_triggered"

    return "user_typed"


def load_power_users_data(csv_path):
    """Load power users data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} power users from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading power users CSV: {e}")
        return None


def load_studio_results_data(csv_path):
    """Load studio results data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} studio sessions from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading studio results CSV: {e}")
        return None


def get_empty_stats():
    """Return empty statistics dictionary."""
    return {
        "total_sessions": 0,
        "total_messages": 0,
        "user_messages": 0,
        "agent_messages": 0,
        "user_typed_messages": 0,
        "micro_agent_triggered_messages": 0,
        "gui_sessions": 0,
        "gui_total_messages": 0,
        "gui_user_messages": 0,
        "gui_user_typed_messages": 0,
    }


def process_user_message(event, stats, is_gui_session):
    """Process a user message event and update statistics."""
    stats["user_messages"] += 1
    if is_gui_session:
        stats["gui_user_messages"] += 1

    # Categorize user messages
    category = categorize_user_message(event.get("content", ""))
    if category == "user_typed":
        stats["user_typed_messages"] += 1
        if is_gui_session:
            stats["gui_user_typed_messages"] += 1
    else:
        stats["micro_agent_triggered_messages"] += 1


def process_agent_message(stats):
    """Process an agent message event and update statistics."""
    stats["agent_messages"] += 1


def process_session_events(session_data, stats, is_gui_session):
    """Process all events in a session and update statistics."""
    if "convo_events" not in session_data:
        return

    for event in session_data["convo_events"]:
        source = event.get("source", "unknown")
        stats["total_messages"] += 1
        if is_gui_session:
            stats["gui_total_messages"] += 1

        if source == "user":
            process_user_message(event, stats, is_gui_session)
        elif source == "assistant":
            process_agent_message(stats)


def analyze_user_json_data(user_id, data_dir, gui_conversation_ids=None):
    """
    Analyze detailed session data for a specific user from JSON files.

    Args:
        user_id (str): The user ID
        data_dir (str): Path to processed_data directory
        gui_conversation_ids (set): Set of conversation IDs that are GUI sessions

    Returns:
        dict: User's detailed session statistics
    """
    user_file = os.path.join(data_dir, f"{user_id}.json")

    if not os.path.exists(user_file):
        return get_empty_stats()

    try:
        with open(user_file, encoding="utf-8") as f:
            user_data = json.load(f)

        stats = get_empty_stats()
        stats["total_sessions"] = len(user_data)

        # Initialize GUI conversation IDs if not provided
        if gui_conversation_ids is None:
            gui_conversation_ids = set()

        # Analyze messages in each session
        for session_id, session_data in user_data.items():
            is_gui_session = session_id in gui_conversation_ids
            if is_gui_session:
                stats["gui_sessions"] += 1
            process_session_events(session_data, stats, is_gui_session)

        return stats

    except Exception as e:
        print(f"Error processing {user_file}: {e}")
        return get_empty_stats()


def preprocess_text_for_ngrams(text):
    """
    Preprocess text for n-gram analysis.

    Args:
        text (str): Input text

    Returns:
        list: List of preprocessed tokens
    """
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        # Fallback to simple split if nltk fails
        print(f"Word tokenization failed: {e}")
        tokens = text.split()

    # Remove stopwords and short tokens
    try:
        stop_words = set(stopwords.words("english"))
    except Exception as e:
        print(f"Stopwords loading failed: {e}")
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

    filtered_tokens = [
        token
        for token in tokens
        if len(token) > MIN_TOKEN_LENGTH
        and token.lower() not in stop_words
        and not token.isdigit()
        and re.match(r"^[a-zA-Z-]+$", token)
    ]

    return filtered_tokens


def extract_ngrams(tokens, n=3):
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens (list): List of tokens
        n (int): N-gram size (default: 3)

    Returns:
        list: List of n-gram tuples
    """
    if len(tokens) < n:
        return []

    return list(ngrams(tokens, n))


def analyze_user_ngrams(user_id, data_dir, gui_conversation_ids, n=3):
    """
    Analyze n-grams for a specific user's GUI sessions.

    Args:
        user_id (str): The user ID
        data_dir (str): Path to processed_data directory
        gui_conversation_ids (set): Set of GUI conversation IDs for this user
        n (int): N-gram size (default: 3)

    Returns:
        dict: N-gram analysis results
    """
    user_file = os.path.join(data_dir, f"{user_id}.json")

    if not os.path.exists(user_file) or not gui_conversation_ids:
        return {
            "total_user_typed_messages": 0,
            "total_tokens": 0,
            "top_ngrams": [],
            "ngram_counts": {},
        }

    try:
        with open(user_file, encoding="utf-8") as f:
            user_data = json.load(f)

        all_tokens = []
        user_typed_count = 0

        # Collect all user typed messages from GUI sessions
        for session_id, session_data in user_data.items():
            if session_id in gui_conversation_ids and "convo_events" in session_data:
                for event in session_data["convo_events"]:
                    if event.get("source") == "user":
                        content = event.get("content", "")
                        if categorize_user_message(content) == "user_typed":
                            user_typed_count += 1
                            tokens = preprocess_text_for_ngrams(content)
                            all_tokens.extend(tokens)

        # Extract n-grams
        if len(all_tokens) >= n:
            ngram_list = extract_ngrams(all_tokens, n)
            ngram_counter = Counter(ngram_list)
            top_ngrams = ngram_counter.most_common(10)  # Top 10 n-grams
        else:
            ngram_counter = Counter()
            top_ngrams = []

        return {
            "total_user_typed_messages": user_typed_count,
            "total_tokens": len(all_tokens),
            "top_ngrams": [(" ".join(ngram), count) for ngram, count in top_ngrams],
            "ngram_counts": dict(ngram_counter),
        }

    except Exception as e:
        print(f"Error processing n-grams for {user_file}: {e}")
        return {
            "total_user_typed_messages": 0,
            "total_tokens": 0,
            "top_ngrams": [],
            "ngram_counts": {},
        }


def create_comprehensive_analysis(power_users_df, studio_df, processed_data_dir, n=3):
    """
    Create comprehensive analysis combining all data sources.

    Args:
        power_users_df: DataFrame with power user data
        studio_df: DataFrame with studio results
        processed_data_dir: Path to processed data directory
        n (int): N-gram size for analysis (default: 3)

    Returns:
        tuple: (Complete user analysis DataFrame, Overall n-gram analysis)
    """

    # Initialize results list
    results = []
    overall_ngram_counter = Counter()
    total_overall_tokens = 0
    total_overall_messages = 0

    print("Processing user data with n-gram analysis...")

    # Group studio results by user_id to get GUI session info
    gui_sessions_by_user = defaultdict(list)
    gui_conversation_ids_by_user = defaultdict(set)

    for _, row in studio_df.iterrows():
        if row["trigger"] == "gui":
            gui_sessions_by_user[row["user_id"]].append(row)
            gui_conversation_ids_by_user[row["user_id"]].add(row["conversation_id"])

    # Process each power user
    for idx, user_row in enumerate(power_users_df.iterrows()):
        if idx % PROGRESS_UPDATE_INTERVAL == 0:
            print(f"Processed {idx}/{len(power_users_df)} users...")

        user_id = user_row["user_id"]
        user_email = user_row["user_email"]
        total_sessions = user_row["conversation_count"]

        # Get GUI sessions for this user
        gui_sessions = gui_sessions_by_user.get(user_id, [])
        gui_conversation_ids = gui_conversation_ids_by_user.get(user_id, set())

        # Get most common repository
        repos = [
            session["selected_repository"]
            for session in gui_sessions
            if session["selected_repository"]
        ]
        # filter nan
        repos = [repo for repo in repos if pd.notna(repo)]
        most_common_repo = Counter(repos).most_common(1)[0][0] if repos else "No repository"

        # Get detailed session analysis from JSON data with GUI conversation IDs
        json_stats = analyze_user_json_data(user_id, processed_data_dir, gui_conversation_ids)

        # Get n-gram analysis for this user's GUI sessions
        ngram_stats = analyze_user_ngrams(user_id, processed_data_dir, gui_conversation_ids, n)

        # Update overall n-gram statistics
        for ngram_tuple, count in ngram_stats["ngram_counts"].items():
            overall_ngram_counter[ngram_tuple] += count
        total_overall_tokens += ngram_stats["total_tokens"]
        total_overall_messages += ngram_stats["total_user_typed_messages"]

        # Calculate metrics using exact data from matched GUI sessions
        avg_msg_per_session = json_stats["total_messages"] / max(json_stats["total_sessions"], 1)
        avg_user_msg_per_session = json_stats["user_messages"] / max(
            json_stats["total_sessions"], 1
        )

        # Calculate exact GUI session metrics
        if json_stats["gui_sessions"] > 0:
            avg_msg_per_gui_session = json_stats["gui_total_messages"] / json_stats["gui_sessions"]
            avg_user_msg_per_gui_session = (
                json_stats["gui_user_messages"] / json_stats["gui_sessions"]
            )
            avg_user_typed_per_gui_session = (
                json_stats["gui_user_typed_messages"] / json_stats["gui_sessions"]
            )
        else:
            avg_msg_per_gui_session = 0
            avg_user_msg_per_gui_session = 0
            avg_user_typed_per_gui_session = 0

        # Format top n-grams for display
        top_ngrams_str = "; ".join(
            [f"{ngram} ({count})" for ngram, count in ngram_stats["top_ngrams"][:5]]
        )

        results.append(
            {
                "user_email": user_email,
                "total_sessions": total_sessions,
                "gui_sessions": json_stats["gui_sessions"],
                "most_common_repo": most_common_repo,
                "avg_msg_per_session": round(avg_msg_per_session, 2),
                "avg_user_msg_per_session": round(avg_user_msg_per_session, 2),
                "avg_msg_per_gui_session": round(avg_msg_per_gui_session, 2),
                "avg_user_msg_per_gui_session": round(avg_user_msg_per_gui_session, 2),
                "avg_user_typed_per_gui_session": round(avg_user_typed_per_gui_session, 2),
                # N-gram analysis
                "gui_user_typed_tokens": ngram_stats["total_tokens"],
                "top_ngrams": top_ngrams_str,
                # Additional useful metrics
                "total_messages_analyzed": json_stats["total_messages"],
                "user_typed_messages": json_stats["user_typed_messages"],
                "micro_agent_triggered_messages": json_stats["micro_agent_triggered_messages"],
            }
        )

    print(f"Processed {len(power_users_df)} users successfully!")

    # Create overall n-gram analysis
    overall_top_ngrams = overall_ngram_counter.most_common(20)
    overall_ngram_analysis = {
        "total_tokens": total_overall_tokens,
        "total_messages": total_overall_messages,
        "total_unique_ngrams": len(overall_ngram_counter),
        "top_ngrams": [(" ".join(ngram), count) for ngram, count in overall_top_ngrams],
        "n": n,
    }

    # Convert to DataFrame and sort by total sessions (descending)
    df = pd.DataFrame(results)
    df = df.sort_values("total_sessions", ascending=False)

    return df, overall_ngram_analysis


def create_html_table(
    df, overall_ngram_analysis, output_path="data/user_analysis/user_metrics_table.html"
):
    """
    Create an interactive HTML table with the user metrics.

    Args:
        df (pd.DataFrame): User metrics data
        output_path (str): Path to save the HTML file
    """

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User Interaction Metrics Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .stats-summary {{
                display: flex;
                justify-content: space-around;
                padding: 20px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                color: #6c757d;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .table-container {{
                padding: 20px;
                overflow-x: auto;
            }}
            .search-container {{
                margin-bottom: 20px;
                text-align: center;
            }}
            .search-input {{
                padding: 10px 15px;
                font-size: 16px;
                border: 2px solid #e9ecef;
                border-radius: 25px;
                width: 300px;
                outline: none;
                transition: border-color 0.3s;
            }}
            .search-input:focus {{
                border-color: #667eea;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background-color: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 12px;
                text-align: left;
                font-weight: 600;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                cursor: pointer;
                user-select: none;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            th:hover {{
                background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #e9ecef;
                font-size: 0.9em;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            tr:nth-child(even) {{
                background-color: #fcfcfc;
            }}
            .email-col {{
                font-weight: 600;
                color: #495057;
                max-width: 200px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .number-col {{
                text-align: right;
                font-weight: 500;
                color: #495057;
            }}
            .repo-col {{
                max-width: 250px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                color: #6c757d;
            }}
            .highlight {{
                background-color: #fff3cd !important;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #6c757d;
                font-size: 0.9em;
                border-top: 1px solid #e9ecef;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>User Interaction Metrics Analysis</h1>
                <p>Comprehensive analysis of user sessions, GUI interactions, messaging patterns, and n-gram analysis</p>
            </div>

            <div class="stats-summary">
                <div class="stat-item">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Total Users</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{df['total_sessions'].sum():,}</div>
                    <div class="stat-label">Total Sessions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{df['gui_sessions'].sum():,}</div>
                    <div class="stat-label">GUI Sessions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{overall_ngram_analysis['total_tokens']:,}</div>
                    <div class="stat-label">Total Tokens Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{overall_ngram_analysis['total_unique_ngrams']:,}</div>
                    <div class="stat-label">Unique {overall_ngram_analysis['n']}-grams</div>
                </div>
            </div>

            <div class="ngram-summary" style="padding: 20px; background-color: #f8f9fa; border-bottom: 1px solid #e9ecef;">
                <h3 style="margin-top: 0; color: #495057;">Top {overall_ngram_analysis['n']}-grams Across All GUI Sessions</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
    """

    # Add top n-grams in a grid layout
    for _, (ngram, count) in enumerate(overall_ngram_analysis["top_ngrams"][:10]):
        html_content += f"""
                    <div style="background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #667eea;">
                        <strong>"{ngram}"</strong> <span style="color: #6c757d;">({count:,} occurrences)</span>
                    </div>
        """

    html_content += """
                </div>
            </div>

            <div class="table-container">
                <div class="search-container">
                    <input type="text" id="searchInput" class="search-input" placeholder="Search by email or repository..." onkeyup="filterTable()">
                </div>

                <table id="userTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">User Email ↕</th>
                            <th onclick="sortTable(1)">Total Sessions ↕</th>
                            <th onclick="sortTable(2)">GUI Sessions ↕</th>
                            <th onclick="sortTable(3)">Most Common Repo ↕</th>
                            <th onclick="sortTable(4)">Avg Msg/Session ↕</th>
                            <th onclick="sortTable(5)">Avg User Msg/Session ↕</th>
                            <th onclick="sortTable(6)">Avg Msg/GUI Session ↕</th>
                            <th onclick="sortTable(7)">Avg User Typed/GUI Session ↕</th>
                            <th onclick="sortTable(8)">GUI Tokens ↕</th>
                            <th onclick="sortTable(9)">Top 3-grams ↕</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    # Add table rows
    for _, row in df.iterrows():
        html_content += f"""
                        <tr>
                            <td class="email-col">{row['user_email']}</td>
                            <td class="number-col">{row['total_sessions']}</td>
                            <td class="number-col">{row['gui_sessions']}</td>
                            <td class="repo-col" title="{row['most_common_repo']}">{row['most_common_repo']}</td>
                            <td class="number-col">{row['avg_msg_per_session']}</td>
                            <td class="number-col">{row['avg_user_msg_per_session']}</td>
                            <td class="number-col">{row['avg_msg_per_gui_session']}</td>
                            <td class="number-col">{row['avg_user_typed_per_gui_session']}</td>
                            <td class="number-col">{row['gui_user_typed_tokens']}</td>
                            <td class="repo-col" title="{row['top_ngrams']}">{row['top_ngrams'][:MAX_LINE_LENGTH]}{'...' if len(row['top_ngrams']) > MAX_LINE_LENGTH else ''}</td>
                        </tr>
        """

    # Close HTML content
    html_content += f"""
                    </tbody>
                </table>
            </div>

            <div class="footer">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data includes {len(df)} power users from May 2025</p>
            </div>
        </div>

        <script>
            function filterTable() {{
                var input, filter, table, tr, td, i, txtValue;
                input = document.getElementById("searchInput");
                filter = input.value.toUpperCase();
                table = document.getElementById("userTable");
                tr = table.getElementsByTagName("tr");

                for (i = 1; i < tr.length; i++) {{
                    tr[i].style.display = "none";
                    td = tr[i].getElementsByTagName("td");
                    for (var j = 0; j < td.length; j++) {{
                        if (td[j]) {{
                            txtValue = td[j].textContent || td[j].innerText;
                            if (txtValue.toUpperCase().indexOf(filter) > -1) {{
                                tr[i].style.display = "";
                                break;
                            }}
                        }}
                    }}
                }}
            }}

            function sortTable(n) {{
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.getElementById("userTable");
                switching = true;
                dir = "asc";

                while (switching) {{
                    switching = false;
                    rows = table.rows;

                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];

                        var xValue = isNaN(x.innerHTML) ? x.innerHTML.toLowerCase() : parseFloat(x.innerHTML);
                        var yValue = isNaN(y.innerHTML) ? y.innerHTML.toLowerCase() : parseFloat(y.innerHTML);

                        if (dir == "asc") {{
                            if (xValue > yValue) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xValue < yValue) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}

                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """

    # Save HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML table saved to: {output_path}")


def get_empty_ngram_stats():
    """Return empty n-gram statistics dictionary."""
    return {
        "total_user_typed_messages": 0,
        "total_tokens": 0,
        "top_ngrams": [],
        "ngram_counts": {},
    }


def process_user_messages_for_ngrams(user_data, gui_conversation_ids, n):
    """Process user messages to extract n-grams."""
    all_tokens = []
    total_messages = 0

    for session_id, session_data in user_data.items():
        if session_id not in gui_conversation_ids:
            continue

        if "convo_events" not in session_data:
            continue

        for event in session_data["convo_events"]:
            if event.get("source") != "user":
                continue

            content = event.get("content", "")
            if not content:
                continue

            category = categorize_user_message(content)
            if category != "user_typed":
                continue

            total_messages += 1
            tokens = preprocess_text_for_ngrams(content)
            all_tokens.extend(tokens)

    return all_tokens, total_messages


def analyze_ngrams_by_repository(power_users_df, studio_df, processed_data_dir, n=3):
    """
    Analyze n-grams for each repository in the dataset.

    Args:
        power_users_df (pd.DataFrame): DataFrame containing power user data
        studio_df (pd.DataFrame): DataFrame containing studio session data
        processed_data_dir (str): Path to processed data directory
        n (int): N-gram size (default: 3)

    Returns:
        dict: Repository-level n-gram analysis results
    """
    # Get GUI conversation IDs for each user
    gui_conversation_ids = {}
    for _, row in studio_df.iterrows():
        user_id = row["user_id"]
        conv_id = row["conversation_id"]
        if user_id not in gui_conversation_ids:
            gui_conversation_ids[user_id] = set()
        gui_conversation_ids[user_id].add(conv_id)

    # Group users by repository
    repo_users = {}
    for _, row in power_users_df.iterrows():
        repo = row["repository"]
        user_id = row["user_id"]
        if repo not in repo_users:
            repo_users[repo] = set()
        repo_users[repo].add(user_id)

    # Analyze n-grams for each repository
    repo_analysis = {}
    for repo, users in repo_users.items():
        repo_analysis[repo] = get_empty_ngram_stats()
        all_tokens = []
        total_messages = 0

        for user_id in users:
            user_file = os.path.join(processed_data_dir, f"{user_id}.json")
            if not os.path.exists(user_file):
                continue

            try:
                with open(user_file, encoding="utf-8") as f:
                    user_data = json.load(f)

                user_tokens, user_messages = process_user_messages_for_ngrams(
                    user_data, gui_conversation_ids.get(user_id, set()), n
                )
                all_tokens.extend(user_tokens)
                total_messages += user_messages

            except Exception as e:
                print(f"Error processing {user_file}: {e}")
                continue

        # Calculate n-grams for the repository
        if all_tokens:
            ngrams_list = extract_ngrams(all_tokens, n)
            ngram_counts = Counter(ngrams_list)
            repo_analysis[repo].update(
                {
                    "total_user_typed_messages": total_messages,
                    "total_tokens": len(all_tokens),
                    "top_ngrams": ngram_counts.most_common(20),
                    "ngram_counts": dict(ngram_counts),
                }
            )

    return repo_analysis


def main():
    """Main function to run the comprehensive analysis."""

    # File paths
    power_users_csv = "./data/power_user_2025_05.csv"
    studio_results_csv = "./data/studio_results_20250604_1645.csv"
    processed_data_dir = "./data/processed_data"

    print("Starting comprehensive user interaction analysis...")

    # Load data
    power_users_df = load_power_users_data(power_users_csv)
    studio_df = load_studio_results_data(studio_results_csv)

    if power_users_df is None or studio_df is None:
        print("Failed to load required data files!")
        return

    # Create comprehensive analysis with n-gram analysis (default n=3)
    results_df, overall_ngram_analysis = create_comprehensive_analysis(
        power_users_df, studio_df, processed_data_dir, n=3
    )

    # Create HTML table
    create_html_table(results_df, overall_ngram_analysis)

    # Also save CSV for further analysis
    csv_output_path = "data/user_analysis/user_metrics_data.csv"
    results_df.to_csv(csv_output_path, index=False)
    print(f"CSV data saved to: {csv_output_path}")

    # Save n-gram analysis separately
    ngram_output_path = "data/user_analysis/ngram_analysis.json"
    os.makedirs(os.path.dirname(ngram_output_path), exist_ok=True)
    with open(ngram_output_path, "w", encoding="utf-8") as f:
        json.dump(overall_ngram_analysis, f, indent=2, ensure_ascii=False)
    print(f"N-gram analysis saved to: {ngram_output_path}")

    # Analyze n-grams by repository
    repo_ngram_analysis = analyze_ngrams_by_repository(
        power_users_df, studio_df, processed_data_dir, n=3
    )

    # Save repository n-gram analysis
    repo_ngram_output_path = "data/user_analysis/repository_ngram_analysis.json"
    with open(repo_ngram_output_path, "w", encoding="utf-8") as f:
        json.dump(repo_ngram_analysis, f, indent=2, ensure_ascii=False)
    print(f"Repository n-gram analysis saved to: {repo_ngram_output_path}")

    # Print summary
    print("\nAnalysis completed!")
    print(f"Total users analyzed: {len(results_df)}")
    print(f"Users with GUI sessions: {len(results_df[results_df['gui_sessions'] > 0])}")
    print(
        f"Most active user: {results_df.iloc[0]['user_email']} ({results_df.iloc[0]['total_sessions']} sessions)"
    )
    print(f"Most GUI sessions: {results_df['gui_sessions'].max()} sessions")
    print("\nN-gram Analysis Summary:")
    print(f"Total tokens analyzed: {overall_ngram_analysis['total_tokens']:,}")
    print(
        f"Total unique {overall_ngram_analysis['n']}-grams: {overall_ngram_analysis['total_unique_ngrams']:,}"
    )
    print(f"Top 5 {overall_ngram_analysis['n']}-grams:")
    for i, (ngram, count) in enumerate(overall_ngram_analysis["top_ngrams"][:5], 1):
        print(f"  {i}. '{ngram}' ({count:,} occurrences)")

    print("\nRepository Analysis Summary:")
    print(f"Analyzed {len(repo_ngram_analysis)} repositories with sufficient data")

    # Show top repositories by activity
    if repo_ngram_analysis:
        sorted_repos = sorted(
            repo_ngram_analysis.items(), key=lambda x: x[1]["total_tokens"], reverse=True
        )
        print("Top 3 most active repositories by token count:")
        for i, (repo, data) in enumerate(sorted_repos[:3], 1):
            print(
                f"  {i}. {repo}: {data['total_tokens']:,} tokens, {data['total_sessions']} sessions"
            )
            if data["top_ngrams"]:
                print(
                    f"     Top 3-gram: '{data['top_ngrams'][0][0]}' ({data['top_ngrams'][0][1]} times)"
                )
            print()


if __name__ == "__main__":
    main()
