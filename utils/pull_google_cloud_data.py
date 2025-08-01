import csv
import os
import subprocess


def pull_google_cloud_data():
    # Create data directory if it doesn't exist
    data_dir = "./data/sessions"
    os.makedirs(data_dir, exist_ok=True)

    # Construct the gsutil command
    source = "gs://prod-openhands-sessions/sessions"

    # First, list all available sessions
    try:
        result = subprocess.run(
            ["gsutil", "ls", source], capture_output=True, text=True, check=True
        )
        available_sessions = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error listing available sessions: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return

    # Get list of already downloaded sessions
    existing_sessions = set(os.listdir(data_dir)) if os.path.exists(data_dir) else set()

    # Pull each session that doesn't exist locally
    for session_path in available_sessions:
        session_id = session_path.strip("/").split("/")[-1]
        if session_id not in existing_sessions:
            try:
                # Create a subdirectory for each session
                session_dir = os.path.join(data_dir, session_id)
                os.makedirs(session_dir, exist_ok=True)

                # Use -r flag for recursive copy and ensure destination is a directory
                subprocess.run(
                    ["gsutil", "-m", "cp", "-r", session_path, session_dir], check=True
                )
                print(f"Successfully pulled session {session_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error pulling session {session_id}: {e}")
            except Exception as e:
                print(f"Unexpected error for session {session_id}: {e}")
        else:
            print(f"Skipping session {session_id} as it already exists")


def precise_pull(csv_file, max_users=None):
    """
    Pull specific sessions from Google Cloud Storage based on user IDs in a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing user IDs
        max_users (int, optional): Maximum number of users to pull. If None, pulls all users.
    """
    # Create data directory if it doesn't exist
    data_dir = "./data/sessions"
    os.makedirs(data_dir, exist_ok=True)

    # Get list of already downloaded users
    existing_users = set(os.listdir(data_dir)) if os.path.exists(data_dir) else set()

    # Read user IDs from CSV file
    session_ids = set()
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if max_users is not None and count >= max_users:
                break
            session_ids.add(row["user_id"])
            count += 1

    # Filter out users that already exist locally
    new_session_ids = session_ids - existing_users
    skipped_count = len(session_ids) - len(new_session_ids)

    print(f"Found {len(session_ids)} user IDs in CSV")
    if skipped_count > 0:
        print(f"Skipping {skipped_count} users that already exist locally")
    print(
        f"Will pull {len(new_session_ids)} new users"
        + (f" (limited to {max_users})" if max_users else "")
    )

    # Base source path
    base_source = "gs://prod-openhands-sessions/users"

    # Pull each specific session that doesn't exist locally
    for session_id in new_session_ids:
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{base_source}/{session_id}/", data_dir],
                check=True,
            )
            print(f"Successfully pulled session {session_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling session {session_id}: {e}")
        except Exception as e:
            print(f"Unexpected error for session {session_id}: {e}")


if __name__ == "__main__":
    # pull_google_cloud_data()
    precise_pull("./data/power_user_2025_05.csv")
