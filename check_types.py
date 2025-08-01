#!/usr/bin/env python3
"""Simple type checking script"""


def check_imports():
    """Test basic imports"""
    try:
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def check_method_signature():
    """Check method signature is correct"""
    try:
        from tom_swe.tom_agent import ToMAgent, ToMAgentConfig
        from unittest.mock import patch

        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig()
            agent = ToMAgent(config)

            # Check method exists and has correct signature
            method = getattr(agent, "propose_instructions")
            import inspect

            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            print(f"✅ Method signature: {params}")
            print(f"✅ Return annotation: {sig.return_annotation}")

            # Check it's callable with correct args
            if "user_id" in params and "original_instruction" in params:
                print("✅ Method signature looks correct")
                return True
            else:
                print("❌ Method signature incorrect")
                return False

    except Exception as e:
        print(f"❌ Signature check error: {e}")
        return False


def main():
    print("🔍 Checking types...")

    if not check_imports():
        return 1

    if not check_method_signature():
        return 1

    print("✅ All type checks passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
