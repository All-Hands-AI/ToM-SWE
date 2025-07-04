"""
Example script demonstrating how to use the RAG agent with organic document structure.

This script shows how to:
1. Load users as documents with sessions as chunks (organic structure)
2. Perform queries using basic and contextual retrieval
3. Retrieve relevant sessions from user's conversation history
"""

import asyncio
import json
from pathlib import Path
from typing import List

from tom_swe.rag_module import Document, RAGAgent, VectorDB
from tom_swe.utils import split_text_for_embedding

# Constants
CONTENT_PREVIEW_LENGTH = 150

# Target user for this example
TARGET_USER_ID = "20d03f52-abb6-4414-b024-67cc89d53e12"


def load_user_session_documents(
    user_id: str, data_path: str = "data/processed_data"
) -> List[Document]:
    """Load one user as one Document with sessions as chunks."""

    user_file = Path(data_path) / f"{user_id}.json"

    if not user_file.exists():
        print(f"User file not found: {user_file}")
        return []

    try:
        with open(user_file, encoding="utf-8") as f:
            user_data = json.load(f)

        # Create chunks from all sessions
        chunks = []
        session_ids = []
        current_position = 0

        for session_id, session_data in user_data.items():
            # Use the entire session data as chunk content
            session_content = json.dumps(session_data, indent=2)
            # Split if too large for embedding
            content_chunks = split_text_for_embedding(session_content)
            for i, chunk_content in enumerate(content_chunks):
                chunk_id = (
                    f"{user_id}_{session_id}"
                    if len(content_chunks) == 1
                    else f"{user_id}_{session_id}_part_{i}"
                )
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": chunk_content,
                        "start_index": current_position,
                        "end_index": current_position + len(chunk_content),
                        "session_id": session_id,
                    }
                )
                current_position += len(chunk_content) + 2  # +2 for \n\n separator

            session_ids.append(session_id)

        # Combine all session content for the full document content
        full_content = "\n\n".join(chunk["content"] for chunk in chunks)

        # Create single Document for the user
        doc = Document(
            doc_id=user_id,
            content=full_content,
            chunks=chunks,
            metadata={
                "user_id": user_id,
                "session_ids": session_ids,
                "session_count": len(session_ids),
                "source": "user_all_sessions",
            },
        )

        print(f"Loaded user {user_id} as 1 document with {len(chunks)} session chunks")
        return [doc]

    except Exception as e:
        print(f"Error loading user data: {e}")
        return []


async def demonstrate_simple_user_rag() -> None:
    """Simple RAG demonstration - one user as one document with session chunks."""

    print("\n" + "=" * 60)
    print("SIMPLE USER RAG DEMONSTRATION")
    print(f"Target User: {TARGET_USER_ID}")
    print("=" * 60)

    # Load user document with session chunks
    documents = load_user_session_documents(TARGET_USER_ID)

    if not documents:
        print(f"No documents found for user {TARGET_USER_ID}")
        return

    user_doc = documents[0]  # Should be exactly one document per user
    print(
        f"Loaded user {TARGET_USER_ID}: {(user_doc.metadata or {}).get('session_count', 0)} sessions as chunks"
    )

    # Create basic vector database
    vector_db = VectorDB(name=f"user_{TARGET_USER_ID}_simple_db")
    vector_db.load_data(documents)

    # Create RAG agent
    rag_agent = RAGAgent(vector_db)

    # Example queries
    queries = [
        "What are the user's main development preferences?",
        "How does the user typically handle debugging issues?",
        "What kind of testing approaches does the user prefer?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = rag_agent.retrieve(query, k=3)

        print("Retrieved sessions:")
        for i, result in enumerate(results, 1):
            session_id = result.metadata.get("session_id", "unknown")
            similarity = result.similarity
            content_preview = (
                result.content[:CONTENT_PREVIEW_LENGTH] + "..."
                if len(result.content) > CONTENT_PREVIEW_LENGTH
                else result.content
            )
            print(f"  {i}. Session: {session_id} (Similarity: {similarity:.3f})")
            print(f"     Preview: {content_preview}")


async def main() -> None:
    """Main demonstration function - organic approach."""

    print("User-Specific RAG Agent Demonstration")
    print(
        f"This script demonstrates RAG with organic document structure for user: {TARGET_USER_ID}"
    )

    print("\n1. Running simple RAG demonstration...")
    await demonstrate_simple_user_rag()


if __name__ == "__main__":
    asyncio.run(main())
