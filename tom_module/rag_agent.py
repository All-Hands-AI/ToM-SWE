"""
RAG Agent with Contextual Embeddings

This module implements a Retrieval Augmented Generation (RAG) agent based on
Anthropic's contextual embeddings approach. It enhances traditional RAG by adding
relevant context to each chunk before embedding, improving retrieval accuracy.

The implementation includes:
1. VectorDB: Basic vector database for storing embeddings
2. ContextualVectorDB: Enhanced vector database with contextual embeddings
3. RAGAgent: Main agent class that orchestrates retrieval and generation
"""

import asyncio
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# LiteLLM imports
import litellm
import numpy as np
from dotenv import load_dotenv
from litellm import acompletion, embedding
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure litellm for better error handling
litellm.set_verbose = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LiteLLM configuration
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")


@dataclass
class Document:
    """Represents a document with its chunks and metadata."""

    doc_id: str
    content: str
    chunks: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata and similarity score."""

    content: str
    metadata: Dict[str, Any]
    similarity: float
    context: Optional[str] = None


class VectorDB:
    """
    Basic vector database for storing and retrieving embeddings.
    Uses in-memory storage with optional persistence to disk.
    """

    def __init__(self, name: str, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector database.

        Args:
            name: Name of the database (used for file persistence)
            embedding_model: Name of the embedding model to use (defaults to OpenAI's best model)
        """
        self.name = name
        self.embedding_model = embedding_model
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.query_cache: Dict[str, List[float]] = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using litellm.

        Args:
            texts: List of text strings to embed (should already be truncated)

        Returns:
            List of embedding vectors
        """
        # Use smaller batch size to avoid hitting the 300,000 tokens per request limit
        # With texts at ~7800 tokens each, batch size of 20 gives us ~156K tokens per batch
        # This provides larger safety margin for chunks that exceed expected size
        batch_size = 20
        embeddings: List[List[float]] = []

        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = embedding(model=self.embedding_model, input=batch)
                # Extract embeddings from litellm response
                # Handle both dict and object formats for compatibility
                if hasattr(response, "data"):
                    # If response.data is available, use it
                    batch_embeddings = []
                    for data in response.data:
                        if hasattr(data, "embedding"):
                            batch_embeddings.append(data.embedding)
                        elif isinstance(data, dict) and "embedding" in data:
                            batch_embeddings.append(data["embedding"])
                        else:
                            logger.error(f"Unexpected data format: {type(data)}, {data}")
                            raise ValueError(
                                f"Cannot extract embedding from response data: {type(data)}"
                            )
                else:
                    # Fallback for different response formats
                    logger.error(f"Unexpected response format: {type(response)}, {response}")
                    raise ValueError(f"Cannot extract embeddings from response: {type(response)}")

                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

        return embeddings

    def load_data(self, documents: List[Document]) -> None:
        """
        Load documents into the vector database.

        Args:
            documents: List of Document objects to load
        """

        if self.embeddings and self.metadata:
            logger.info("Vector database is already loaded. Skipping data loading.")
            return

        if os.path.exists(self.db_path):
            logger.info("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []

        total_chunks = sum(len(doc.chunks) for doc in documents)

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for doc in documents:
                for chunk in doc.chunks:
                    content = chunk["content"]

                    # Content should already be properly chunked at preprocessing stage

                    texts_to_embed.append(content)
                    metadata.append(
                        {
                            "doc_id": doc.doc_id,
                            "chunk_id": chunk.get("chunk_id", f"{doc.doc_id}_{len(metadata)}"),
                            "content": content,
                            "doc_metadata": doc.metadata or {},
                            "chunk_metadata": chunk,  # Store all chunk metadata including token counts, etc.
                        }
                    )
                    pbar.update(1)

        # Generate embeddings
        self.embeddings = self._get_embeddings(texts_to_embed)
        self.metadata = metadata

        # Save to disk
        self.save_db()
        logger.info(
            f"Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}"
        )

    def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self._get_embeddings([query])[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            result = RetrievalResult(
                content=self.metadata[idx]["content"],
                metadata=self.metadata[idx],
                similarity=float(similarities[idx]),
            )
            results.append(result)

        return results

    def save_db(self) -> None:
        """Save the database to disk."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": self.query_cache,
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self) -> None:
        """Load the database from disk."""
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found.")

        with open(self.db_path, "rb") as file:
            data = pickle.load(file)

        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = data.get("query_cache", {})


class ContextualVectorDB(VectorDB):
    """
    Enhanced vector database that adds contextual information to chunks before embedding.
    This improves retrieval accuracy by providing more context for each chunk.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "text-embedding-3-large",
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the contextual vector database.

        Args:
            name: Name of the database
            embedding_model: Name of the embedding model to use
            llm_model: LLM model to use for context generation (defaults to DEFAULT_LLM_MODEL)
        """
        super().__init__(name, embedding_model)

        # Set LLM model for context generation
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        # Default user model path, can be overridden
        self.user_model_path = "data/user_model"

    def _load_session_context_string(self, user_id: str, session_id: str) -> str:
        """
        Load session context as a simple string from user_model_detailed data.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Context string for the session
        """
        # Try both .json and .jsonl extensions
        detailed_file_json = (
            Path(self.user_model_path) / "user_model_detailed" / user_id / f"{session_id}.json"
        )
        detailed_file_jsonl = (
            Path(self.user_model_path) / "user_model_detailed" / user_id / f"{session_id}.jsonl"
        )

        detailed_file = None
        if detailed_file_json.exists():
            detailed_file = detailed_file_json
        elif detailed_file_jsonl.exists():
            detailed_file = detailed_file_jsonl

        if not detailed_file:
            logger.warning(f"Detailed session file not found for {user_id}/{session_id}")
            return ""

        try:
            with open(detailed_file, encoding="utf-8") as f:
                content = f.read().strip()

            # Simple context string with session info
            context = f"Session {session_id} analysis: {content}"
            return context

        except Exception as e:
            logger.error(f"Error loading session context from {detailed_file}: {e}")
            return ""

    def load_data(self, documents: List[Document]) -> None:
        """
        Load documents with contextual embeddings using existing user model data.

        Args:
            documents: List of Document objects to load
        """
        if self.embeddings and self.metadata:
            logger.info("Contextual vector database is already loaded. Skipping data loading.")
            return

        if os.path.exists(self.db_path):
            logger.info("Loading contextual vector database from disk.")
            self.load_db()
            return

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []

        total_chunks = sum(len(doc.chunks) for doc in documents)

        with tqdm(total=total_chunks, desc="Processing chunks with simple context") as pbar:
            for doc in documents:
                # Extract user_id and session_id from document metadata
                doc_metadata = doc.metadata or {}
                user_id = doc_metadata.get("user_id", "")
                session_id = doc.doc_id  # Assuming doc_id is the session_id

                # Load simple session context string
                session_context = ""
                if user_id and session_id:
                    session_context = self._load_session_context_string(user_id, session_id)

                for chunk in doc.chunks:
                    chunk_content = chunk["content"]

                    # Create simple contextualized content
                    if session_context:
                        contextualized_content = (
                            f"Context: {session_context}\n\nContent: {chunk_content}"
                        )
                    else:
                        contextualized_content = chunk_content

                    # Content should already be properly chunked, even with added context
                    final_content = contextualized_content

                    texts_to_embed.append(final_content)
                    metadata.append(
                        {
                            "doc_id": doc.doc_id,
                            "chunk_id": chunk.get("chunk_id", f"{doc.doc_id}_{len(metadata)}"),
                            "content": final_content,
                            "original_content": chunk_content,  # Store original chunk content without context
                            "context": session_context,
                            "doc_metadata": doc.metadata or {},
                            "chunk_metadata": chunk,  # Store all chunk metadata
                        }
                    )
                    pbar.update(1)

        # Generate embeddings
        self.embeddings = self._get_embeddings(texts_to_embed)
        self.metadata = metadata

        # Save to disk
        self.save_db()
        logger.info(
            f"Contextual vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}"
        )

    def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """
        Search for similar documents using contextual embeddings.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of RetrievalResult objects with context information
        """
        results = super().search(query, k)

        # Add context information to results
        for result in results:
            result.context = result.metadata.get("context", "")

        return results


class RAGAgent:
    """
    Main RAG agent that orchestrates retrieval and generation using contextual embeddings.
    """

    def __init__(
        self, vector_db: Union[VectorDB, ContextualVectorDB], llm_model: Optional[str] = None
    ):
        """
        Initialize the RAG agent.

        Args:
            vector_db: Vector database for retrieval
            llm_model: LLM model to use for generation (defaults to DEFAULT_LLM_MODEL)
        """
        self.vector_db = vector_db
        self.llm_model = llm_model or DEFAULT_LLM_MODEL

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        return self.vector_db.search(query, k)

    async def generate_response(
        self, query: str, retrieved_docs: List[RetrievalResult], max_tokens: int = 1024
    ) -> str:
        """
        Generate a response using retrieved documents.

        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            max_tokens: Maximum tokens for response

        Returns:
            Generated response
        """
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Use top 5 documents
            context_parts.append(f"Document {i+1}:\n{doc.metadata['original_content']}")

        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

        try:
            # Prepare the completion call
            completion_args = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }

            # Add API key and base URL if available
            if LITELLM_API_KEY:
                completion_args["api_key"] = LITELLM_API_KEY
            if LITELLM_BASE_URL:
                completion_args["api_base"] = LITELLM_BASE_URL

            # Use native async completion
            response = await acompletion(**completion_args)
            content: str = response.choices[0].message.content
            return content.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    async def query(self, question: str, k: int = 10, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.

        Args:
            question: User question
            k: Number of documents to retrieve
            max_tokens: Maximum tokens for response

        Returns:
            Dictionary containing retrieved documents and generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, k)

        # Generate response
        response = await self.generate_response(question, retrieved_docs, max_tokens)

        return {
            "question": question,
            "retrieved_documents": [
                {
                    "content": doc.metadata["original_content"],
                    "similarity": doc.similarity,
                    "context": getattr(doc, "context", ""),
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ],
            "response": response,
        }


def load_processed_data(data_path: str) -> List[Document]:
    """
    Load processed data from the specified path - organic approach.
    Each user JSON file becomes one document with sessions as chunks.

    Args:
        data_path: Path to the processed data directory

    Returns:
        List of Document objects (one per user)
    """
    documents: List[Document] = []
    data_dir = Path(data_path)

    if not data_dir.exists():
        logger.warning(f"Data directory {data_path} does not exist. Creating empty document list.")
        return documents

    # Look for JSON files in the directory (each should be a user file)
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                user_data = json.load(f)

            # Assume this is user data with sessions
            if isinstance(user_data, dict):
                chunks = []
                session_ids = []
                current_position = 0

                for session_id, session_data in user_data.items():
                    # Convert session to string content
                    session_content = json.dumps(session_data, indent=2)

                    chunks.append(
                        {
                            "chunk_id": f"{json_file.stem}_{session_id}",
                            "content": session_content,
                            "start_index": current_position,
                            "end_index": current_position + len(session_content),
                            "session_id": session_id,
                        }
                    )
                    session_ids.append(session_id)
                    current_position += len(session_content) + 2  # +2 for \n\n separator

                # Combine all session content
                full_content = "\n\n".join(chunk["content"] for chunk in chunks)

                # Create Document object for the user
                doc = Document(
                    doc_id=json_file.stem,  # User ID from filename
                    content=full_content,
                    chunks=chunks,
                    metadata={
                        "user_id": json_file.stem,
                        "session_ids": session_ids,
                        "session_count": len(session_ids),
                        "source_file": str(json_file),
                    },
                )
                documents.append(doc)
            else:
                # Fallback: treat entire file as single chunk
                content = json.dumps(user_data, indent=2)
                chunks = [
                    {
                        "chunk_id": f"{json_file.stem}_chunk_0",
                        "content": content,
                        "start_index": 0,
                        "end_index": len(content),
                    }
                ]

                doc = Document(
                    doc_id=json_file.stem,
                    content=content,
                    chunks=chunks,
                    metadata={"source_file": str(json_file)},
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(documents)} user documents from {data_path}")
    return documents


def _parse_data_item(item: Dict[str, Any], doc_id: str) -> Optional[Document]:
    """
    Parse a data item into a Document object - simplified version.
    Creates a single chunk from the entire item.

    Args:
        item: Data item dictionary
        doc_id: Document ID

    Returns:
        Document object or None if parsing fails
    """
    try:
        # Convert entire item to string content
        content = json.dumps(item, indent=2)

        # Create single chunk for entire content
        chunks = [
            {
                "chunk_id": f"{doc_id}_chunk_0",
                "content": content,
                "start_index": 0,
                "end_index": len(content),
            }
        ]

        return Document(
            doc_id=doc_id, content=content, chunks=chunks, metadata=item.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Error parsing data item {doc_id}: {e}")
        return None


def load_user_model_data(user_model_path: str) -> Optional[Dict[str, Any]]:
    """
    Load user model data for contextual information.

    Args:
        user_model_path: Path to the user model file

    Returns:
        User model data dictionary or None if loading fails
    """
    user_model_file = Path(user_model_path)

    if not user_model_file.exists():
        logger.warning(f"User model file {user_model_path} does not exist.")
        return None

    try:
        with open(user_model_file, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded user model data from {user_model_path}")
        return dict(data)
    except Exception as e:
        logger.error(f"Error loading user model data: {e}")
        return None


async def create_rag_agent(
    data_path: str = "data/processed_data",
    user_model_path: str = "data/user_model",
    use_contextual: bool = True,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> RAGAgent:
    """
    Create and initialize a RAG agent.

    Args:
        data_path: Path to processed data
        user_model_path: Path to user model base directory (contains user_model_detailed subdirectory)
        use_contextual: Whether to use contextual embeddings
        llm_model: LLM model to use for generation

    Returns:
        Initialized RAGAgent
    """
    # Load data
    documents = load_processed_data(data_path)

    # Create vector database
    vector_db: Union[VectorDB, ContextualVectorDB]
    if use_contextual:
        vector_db = ContextualVectorDB(name="contextual_rag_db", llm_model=llm_model)
        # Pass the user_model_path to the ContextualVectorDB for loading detailed data
        vector_db.user_model_path = user_model_path
        vector_db.load_data(documents)
    else:
        vector_db = VectorDB(name="basic_rag_db")
        vector_db.load_data(documents)

    # Create RAG agent
    rag_agent = RAGAgent(vector_db, llm_model)

    return rag_agent


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Example of how to use the RAG agent

        # Create RAG agent (OPENAI_API_KEY should be set in environment)
        agent = await create_rag_agent(
            data_path="data/processed_data",
            user_model_path="user_model/user_model_detailed.json",
            use_contextual=True,
        )

        # Example query
        question = "What are the main features of the system?"
        result = await agent.query(question)

        print(f"Question: {result['question']}")
        print(f"Response: {result['response']}")
        print(f"Retrieved {len(result['retrieved_documents'])} documents")

    asyncio.run(main())
