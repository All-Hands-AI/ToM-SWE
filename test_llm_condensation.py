#!/usr/bin/env python3
"""
Test script for LLM condensation in RAG module
"""

import sys

from tom_swe.rag_module import VectorDB

sys.path.append(".")


def test_llm_condensation():
    """Test LLM condensation functionality"""

    # Create a VectorDB instance to access condensation methods
    vector_db = VectorDB(name="test_db")

    # Test long agent message
    long_agent_msg = """
I understand you're having trouble with the authentication system. Let me help you debug this step by step.

First, let's look at the error you're encountering. Based on the stack trace you provided, it seems like the issue is in the `validate_token` function in `auth/jwt_handler.py` at line 45. The error message indicates that the JWT token is being decoded incorrectly.

Here's what I think is happening:
1. The token is being passed to the validation function
2. The function is trying to decode it using the wrong secret key
3. This is causing a InvalidTokenError to be raised

Let me walk you through the debugging process:

Step 1: Check your JWT_SECRET_KEY in the environment variables
Step 2: Verify that the token format is correct (should start with "Bearer ")
Step 3: Make sure the token hasn't expired
Step 4: Check if the algorithm specified in the decode function matches what was used to encode

Here's the code you should look at in `auth/jwt_handler.py`:

```python
def validate_token(token: str) -> dict:
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]

        # Decode the token
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")
```

Try implementing this fix and let me know if you're still encountering issues. If the problem persists, we might need to look at how the token is being generated in the first place.
    """

    # Test condensation
    print("Testing LLM condensation...")
    print(f"Original length: {len(long_agent_msg)} characters")
    print(f"Original tokens: {vector_db._count_tokens(long_agent_msg)}")

    # Test condensing to 200 tokens
    condensed = vector_db._condense_if_needed(long_agent_msg, 200, "prev_agent_msg")

    print(f"\nCondensed length: {len(condensed)} characters")
    print(f"Condensed tokens: {vector_db._count_tokens(condensed)}")
    print(f"\nCondensed content:\n{condensed}")


if __name__ == "__main__":
    test_llm_condensation()
