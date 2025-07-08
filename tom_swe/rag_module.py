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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# LiteLLM imports
import litellm
import numpy as np
from dotenv import load_dotenv
from litellm import acompletion, embedding
from tqdm import tqdm

# Token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

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
class ChunkingConfig:
    """Configuration for the chunking strategy."""

    target_chunk_tokens: int = 5000  # Target tokens per chunk (optimal for context)
    max_chunk_tokens: int = 6000  # Hard limit per chunk (safe for embedding with context)
    user_message_priority: float = 0.6  # Proportion of tokens for user message
    context_priority: float = 0.4  # Proportion for surrounding context
    min_user_message_tokens: int = 100  # Minimum tokens to preserve user intent
    overlap_tokens: int = 200  # Overlap between chunks for continuity

    # System tag patterns to extract
    system_tag_patterns: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.system_tag_patterns is None:
            self.system_tag_patterns = [
                r"<REPOSITORY_INFO>.*?</REPOSITORY_INFO>",
                r"<RUNTIME_INFORMATION>.*?</RUNTIME_INFORMATION>",
                r"<EXTRA_INFO>.*?</EXTRA_INFO>",
                r"<ENVIRONMENT>.*?</ENVIRONMENT>",
                r"<CONTEXT>.*?</CONTEXT>",
            ]


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

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken if available, otherwise estimate.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if tiktoken is None:
            # Fallback to character-based estimation
            return len(text) // 4

        try:
            # Use cl100k_base encoding (GPT-4, text-embedding-3-large)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback if tiktoken fails
            return len(text) // 4

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using litellm.

        Args:
            texts: List of text strings to embed (should already be properly chunked)

        Returns:
            List of embedding vectors
        """
        # Check text sizes with precise token counting
        total_tokens = sum(self._count_tokens(text) for text in texts)
        logger.info("🔍 EMBEDDING DEBUG:")
        logger.info(f"  - Total texts: {len(texts)}")
        logger.info(f"  - Total tokens: {total_tokens:,}")

        if len(texts) > 0:
            max_tokens = max(self._count_tokens(text) for text in texts)
            avg_tokens = total_tokens // len(texts)
            logger.info(f"  - Max text tokens: {max_tokens:,}")
            logger.info(f"  - Avg text tokens: {avg_tokens:,}")

        # Use very conservative batch size to stay well under 300K token limit
        # With up to 8K token chunks, batch of 30 = ~240K tokens (safe margin)
        batch_size = 30
        embeddings: List[List[float]] = []

        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Verify batch token count
                batch_tokens = sum(self._count_tokens(text) for text in batch)
                logger.debug(
                    f"Batch {i//batch_size + 1}: {len(batch)} texts, {batch_tokens:,} tokens"
                )

                if batch_tokens > 300000:
                    logger.error(f"❌ BATCH TOO LARGE: {batch_tokens:,} tokens > 300K limit!")
                    raise ValueError(
                        f"Batch {i//batch_size + 1} has {batch_tokens:,} tokens, exceeds 300K limit"
                    )

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

    def load_data(
        self, documents: List[Document], chunking_config: Optional[ChunkingConfig] = None
    ) -> None:
        """
        Load documents into the vector database with optimized chunking.

        Args:
            documents: List of Document objects to load
            chunking_config: Configuration for chunking strategy
        """

        if self.embeddings and self.metadata:
            logger.info("Vector database is already loaded. Skipping data loading.")
            return

        if os.path.exists(self.db_path):
            logger.info("Loading vector database from disk.")
            self.load_db()
            return

        # Use default chunking config if not provided
        config = chunking_config or ChunkingConfig()

        # Process documents with new chunking strategy
        processed_chunks = self._create_optimized_chunks(documents, config)

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for chunk_info in processed_chunks:
            texts_to_embed.append(chunk_info["content"])
            metadata.append(chunk_info["metadata"])

        logger.info(f"Generated {len(texts_to_embed)} optimized chunks")

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
        search_start_time = time.time()
        logger.info(f"🔍 Starting vector search for query (k={k})...")

        # Query embedding generation
        embedding_start_time = time.time()
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
            logger.info("⏱️  Query embedding: cached (0.00s)")
        else:
            query_embedding = self._get_embeddings([query])[0]
            self.query_cache[query] = query_embedding
            embedding_end_time = time.time()
            logger.info(
                f"⏱️  Query embedding generation: {embedding_end_time - embedding_start_time:.2f}s"
            )

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        # Calculate similarities
        similarity_start_time = time.time()
        logger.info(f"🔢 Computing similarities with {len(self.embeddings)} stored embeddings...")
        similarities = np.dot(self.embeddings, query_embedding)
        similarity_end_time = time.time()
        logger.info(
            f"⏱️  Matrix multiplication (np.dot): {similarity_end_time - similarity_start_time:.2f}s"
        )

        # Sort and get top results
        sort_start_time = time.time()
        top_indices = np.argsort(similarities)[::-1][:k]
        sort_end_time = time.time()
        logger.info(f"⏱️  Sorting and top-k selection: {sort_end_time - sort_start_time:.2f}s")

        # Build results
        results_start_time = time.time()
        results = []
        for idx in top_indices:
            result = RetrievalResult(
                content=self.metadata[idx]["content"],
                metadata=self.metadata[idx],
                similarity=float(similarities[idx]),
            )
            results.append(result)
        results_end_time = time.time()
        logger.info(f"⏱️  Results construction: {results_end_time - results_start_time:.2f}s")

        total_search_time = time.time() - search_start_time
        logger.info(f"⏱️  Total vector search time: {total_search_time:.2f}s")
        return results

    def _create_optimized_chunks(
        self, documents: List[Document], config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """
        Create optimized chunks using user message-based chunking strategy.

        Args:
            documents: List of Document objects
            config: Chunking configuration

        Returns:
            List of chunk information dictionaries
        """
        all_chunks = []

        for doc in documents:
            # Parse the document content to extract session data
            try:
                if isinstance(doc.content, str):
                    session_data = json.loads(doc.content)
                else:
                    session_data = doc.content

                # Process each session separately
                for session_id, session_content in session_data.items():
                    if isinstance(session_content, dict) and "convo_events" in session_content:
                        chunks = self._chunk_session_by_user_messages(
                            session_content, config, doc.doc_id, session_id
                        )
                        all_chunks.extend(chunks)
                    else:
                        # Fallback: treat as single chunk if structure is unexpected
                        content = json.dumps(session_content, indent=2)
                        if self._count_tokens(content) <= config.max_chunk_tokens:
                            all_chunks.append(
                                {
                                    "content": content,
                                    "metadata": {
                                        "doc_id": doc.doc_id,
                                        "session_id": session_id,
                                        "chunk_id": f"{doc.doc_id}_{session_id}_fallback",
                                        "chunk_type": "fallback",
                                        "token_count": self._count_tokens(content),
                                        "doc_metadata": doc.metadata or {},
                                    },
                                }
                            )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse document {doc.doc_id} for chunking: {e}")
                # Fallback to original chunking for malformed data
                for chunk in doc.chunks:
                    content = chunk["content"]
                    if self._count_tokens(content) <= config.max_chunk_tokens:
                        all_chunks.append(
                            {
                                "content": content,
                                "metadata": {
                                    "doc_id": doc.doc_id,
                                    "chunk_id": chunk.get(
                                        "chunk_id", f"{doc.doc_id}_{len(all_chunks)}"
                                    ),
                                    "content": content,
                                    "chunk_type": "legacy",
                                    "token_count": self._count_tokens(content),
                                    "doc_metadata": doc.metadata or {},
                                    "chunk_metadata": chunk,
                                },
                            }
                        )

        logger.info(f"Created {len(all_chunks)} optimized chunks from {len(documents)} documents")
        return all_chunks

    def _chunk_session_by_user_messages(
        self, session: Dict[str, Any], config: ChunkingConfig, doc_id: str, session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk a single session by user messages with smart token allocation.

        Args:
            session: Session data dictionary
            config: Chunking configuration
            doc_id: Document ID
            session_id: Session ID

        Returns:
            List of chunk dictionaries
        """
        events = session.get("convo_events", [])
        chunks = []

        # Find user message events
        user_message_indices = []
        for i, event in enumerate(events):
            if event.get("source") == "user":
                user_message_indices.append(i)

        if not user_message_indices:
            # No user messages, create single chunk if small enough
            content = json.dumps(session, indent=2)
            token_count = self._count_tokens(content)
            if token_count <= config.max_chunk_tokens:
                chunks.append(
                    {
                        "content": content,
                        "metadata": {
                            "doc_id": doc_id,
                            "session_id": session_id,
                            "chunk_id": f"{doc_id}_{session_id}_no_user_msg",
                            "chunk_type": "no_user_messages",
                            "token_count": token_count,
                            "user_message_count": 0,
                        },
                    }
                )
            return chunks

        # Process each user message with surrounding context
        for i, user_idx in enumerate(user_message_indices):
            chunk_events = self._build_token_limited_context(events, user_idx, config)

            # Create chunk content
            chunk_session = {
                "convo_start": session.get("convo_start"),
                "convo_end": session.get("convo_end"),
                "convo_events": chunk_events,
            }

            content = json.dumps(chunk_session, indent=2)

            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "chunk_id": f"{doc_id}_{session_id}_user_msg_{i}",
                        "chunk_type": "user_message_centered",
                        "token_count": self._count_tokens(content),
                        "user_message_index": user_idx,
                        "user_message_count": 1,
                        "context_events": len(chunk_events) - 1,
                    },
                }
            )

        return chunks

    def _build_token_limited_context(
        self, events: List[Dict[str, Any]], user_msg_idx: int, config: ChunkingConfig
    ) -> List[Dict[str, Any]]:
        """
        Build context around a user message within token limits.

        Args:
            events: All conversation events
            user_msg_idx: Index of the user message to center on
            config: Chunking configuration

        Returns:
            List of events for the chunk
        """
        user_event = events[user_msg_idx].copy()  # Make a copy to avoid modifying original

        # Extract system tags from user message to get pure intent
        user_content = user_event.get("content", "")
        pure_user_content = self._extract_pure_user_intent(user_content, config)

        # Create a cleaned user event with pure content for better chunking
        cleaned_user_event = user_event.copy()
        cleaned_user_event["content"] = pure_user_content
        cleaned_user_event["original_content"] = user_content  # Store original for reference

        # Calculate actual tokens for the cleaned event
        user_event_tokens = self._count_tokens(json.dumps(cleaned_user_event, indent=2))

        # If even the cleaned user message exceeds max tokens, truncate it
        if user_event_tokens > config.max_chunk_tokens:
            logger.warning(
                f"User message too large ({user_event_tokens} tokens), truncating to {config.max_chunk_tokens}"
            )
            # Truncate the pure user content to fit within limits
            words = pure_user_content.split()
            truncated_content = ""
            for word in words:
                test_content = f"{truncated_content} {word}".strip()
                test_event = cleaned_user_event.copy()
                test_event["content"] = test_content
                if self._count_tokens(json.dumps(test_event, indent=2)) > config.max_chunk_tokens:
                    break
                truncated_content = test_content

            cleaned_user_event["content"] = truncated_content
            cleaned_user_event["truncated"] = True
            user_event_tokens = self._count_tokens(json.dumps(cleaned_user_event, indent=2))

        # Start with the cleaned user message
        chunk_events = [cleaned_user_event]
        current_tokens = user_event_tokens

        # Add surrounding context within token limit
        added_before = 0
        added_after = 0

        # Alternate between before and after context
        for offset in range(1, len(events)):
            # Hard limit: never exceed max_chunk_tokens
            if current_tokens >= config.max_chunk_tokens:
                break

            # Soft limit: prefer to stay under target_chunk_tokens
            if current_tokens >= config.target_chunk_tokens:
                # Only add small events if we're over target but under max
                max_additional_tokens = config.max_chunk_tokens - current_tokens
                if max_additional_tokens < 100:  # Not enough room for meaningful context
                    break

            # Try adding event before user message
            before_idx = user_msg_idx - offset
            if before_idx >= 0 and added_before < 10:  # Limit context window
                event = events[before_idx]
                event_tokens = self._count_tokens(json.dumps(event, indent=2))
                if current_tokens + event_tokens <= config.max_chunk_tokens:
                    chunk_events.insert(0, event)
                    current_tokens += event_tokens
                    added_before += 1
                    continue

            # Try adding event after user message
            after_idx = user_msg_idx + offset
            if after_idx < len(events) and added_after < 10:  # Limit context window
                event = events[after_idx]
                event_tokens = self._count_tokens(json.dumps(event, indent=2))
                if current_tokens + event_tokens <= config.max_chunk_tokens:
                    chunk_events.append(event)
                    current_tokens += event_tokens
                    added_after += 1

        return chunk_events

    def _extract_pure_user_intent(self, content: str, config: ChunkingConfig) -> str:
        """
        Extract pure user intent by removing system-injected tags.

        Args:
            content: Original user message content
            config: Chunking configuration with system tag patterns

        Returns:
            Pure user intent without system tags
        """
        pure_content = content

        if config.system_tag_patterns:
            for pattern in config.system_tag_patterns:
                pure_content = re.sub(pattern, "", pure_content, flags=re.DOTALL | re.IGNORECASE)

        return pure_content.strip()

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

    def load_data(
        self, documents: List[Document], chunking_config: Optional[ChunkingConfig] = None
    ) -> None:
        """
        Load documents with contextual embeddings using optimized chunking.

        Args:
            documents: List of Document objects to load
            chunking_config: Configuration for chunking strategy
        """
        if self.embeddings and self.metadata:
            logger.info("Contextual vector database is already loaded. Skipping data loading.")
            return

        if os.path.exists(self.db_path):
            logger.info("Loading contextual vector database from disk.")
            self.load_db()
            return

        # Use default chunking config if not provided
        config = chunking_config or ChunkingConfig()

        # Process documents with new chunking strategy
        processed_chunks = self._create_optimized_chunks(documents, config)

        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []

        with tqdm(
            total=len(processed_chunks), desc="Processing optimized chunks with context"
        ) as pbar:
            for chunk_info in processed_chunks:
                chunk_content = chunk_info["content"]
                chunk_metadata = chunk_info["metadata"]

                # Extract user_id and session_id from chunk metadata
                user_id = chunk_metadata.get("doc_id", "")
                session_id = chunk_metadata.get("session_id", "")

                # Load simple session context string
                session_context = ""
                if user_id and session_id:
                    session_context = self._load_session_context_string(user_id, session_id)

                # Create contextualized content
                if session_context:
                    contextualized_content = (
                        f"Context: {session_context}\n\nContent: {chunk_content}"
                    )
                else:
                    contextualized_content = chunk_content

                # Verify final content doesn't exceed embedding model limits (8192 tokens)
                final_token_count = self._count_tokens(contextualized_content)
                embedding_token_limit = 8000  # Conservative limit for embedding model

                if final_token_count > embedding_token_limit:
                    # Truncate context if needed
                    logger.warning(
                        f"Chunk {chunk_metadata.get('chunk_id')} with context has {final_token_count} tokens, using original content"
                    )
                    final_content = chunk_content
                    final_token_count = self._count_tokens(final_content)

                    # If even original content is too large, truncate it
                    if final_token_count > embedding_token_limit:
                        logger.warning(
                            f"Original chunk {chunk_metadata.get('chunk_id')} too large ({final_token_count} tokens), skipping"
                        )
                        continue
                else:
                    final_content = contextualized_content

                texts_to_embed.append(final_content)

                # Update metadata with contextual information
                enhanced_metadata = {
                    **chunk_metadata,
                    "content": final_content,
                    "original_content": chunk_content,
                    "context": session_context,
                    "final_token_count": self._count_tokens(final_content),
                }
                metadata.append(enhanced_metadata)
                pbar.update(1)

        logger.info(f"Generated {len(texts_to_embed)} contextual chunks")

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
        generation_start_time = time.time()
        logger.info("🤖 Starting RAG response generation...")

        # Prepare context from retrieved documents
        context_prep_start = time.time()
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Use top 5 documents
            context_parts.append(f"Document {i+1}:\n{doc.metadata['original_content']}")

        context = "\n\n".join(context_parts)
        context_prep_end = time.time()

        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

        # Log prompt statistics
        prompt_length = len(prompt)
        context_length = len(context)
        logger.info(f"📊 Context preparation: {context_prep_end - context_prep_start:.2f}s")
        logger.info(f"📊 Prompt length: {prompt_length:,} chars (~{prompt_length//4:,} tokens)")
        logger.info(f"📊 Context length: {context_length:,} chars (~{context_length//4:,} tokens)")
        logger.info(f"📊 Retrieved docs used: {len(retrieved_docs[:5])}")

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
            llm_start_time = time.time()
            logger.info("🤖 Calling LLM for response generation...")
            response = await acompletion(**completion_args)
            llm_end_time = time.time()

            content: str = response.choices[0].message.content

            generation_end_time = time.time()
            logger.info(f"⏱️  LLM response generation: {llm_end_time - llm_start_time:.2f}s")
            logger.info(
                f"⏱️  Total generate_response time: {generation_end_time - generation_start_time:.2f}s"
            )

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
    Load processed data from the specified path.
    Each user JSON file becomes one document. Chunking will be handled during embedding.

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

            # Store the raw user data - chunking will be handled during load_data
            if isinstance(user_data, dict):
                # Store user data as-is for optimized chunking
                doc = Document(
                    doc_id=json_file.stem,  # User ID from filename
                    content=json.dumps(user_data),  # Convert dict to string
                    chunks=[],  # Will be populated during load_data
                    metadata={
                        "user_id": json_file.stem,
                        "session_count": len(user_data),
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
    chunking_config: Optional[ChunkingConfig] = None,
) -> RAGAgent:
    """
    Create and initialize a RAG agent with optimized chunking.

    Args:
        data_path: Path to processed data
        user_model_path: Path to user model base directory (contains user_model_detailed subdirectory)
        use_contextual: Whether to use contextual embeddings
        llm_model: LLM model to use for generation
        chunking_config: Configuration for chunking strategy (uses default if not provided)

    Returns:
        Initialized RAGAgent
    """
    # Load data
    documents = load_processed_data(data_path)

    # Use default chunking config if not provided
    config = chunking_config or ChunkingConfig()
    logger.info(
        f"Using chunking config: target={config.target_chunk_tokens}, max={config.max_chunk_tokens}"
    )

    # Create vector database
    vector_db: Union[VectorDB, ContextualVectorDB]
    if use_contextual:
        vector_db = ContextualVectorDB(name="contextual_rag_db", llm_model=llm_model)
        # Pass the user_model_path to the ContextualVectorDB for loading detailed data
        vector_db.user_model_path = user_model_path
        vector_db.load_data(documents, config)
    else:
        vector_db = VectorDB(name="basic_rag_db")
        vector_db.load_data(documents, config)

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
