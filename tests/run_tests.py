#!/usr/bin/env python3
"""
Simple test runner for tom_module tests.
"""

import os
import sys

import pytest


def main() -> int:
    """Run all tests with pytest."""

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print("Running tom_module tests...")
    print("=" * 50)

    # Run pytest with verbose output
    test_args = [
        "tests/",
        "-v",  # Verbose output
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
    ]

    # If specific test file is provided as argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_args = [f"tests/{test_file}", "-v", "-s", "--tb=short"]
        print(f"Running specific test file: {test_file}")

    # Run the tests
    exit_code = pytest.main(test_args)

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
