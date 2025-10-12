"""
RAG Agent for User Behavior Analysis

This module implements a Retrieval Augmented Generation (RAG) agent for
retrieving and analyzing user behavior patterns from processed data.

The implementation includes:
1. VectorDB: Basic vector database for storing embeddings
2. RAGAgent: Main agent class that orchestrates retrieval and generation
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
from typing import Any, Dict, List, Optional

# LiteLLM imports
import litellm
import numpy as np
from dotenv import load_dotenv
from litellm import completion, embedding
from tqdm import tqdm
from tom_swe.prompts.manager import render_prompt

# Token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

# Load environment variables
load_dotenv()

# Configure litellm for better error handling
litellm.set_verbose = False

# Get logger that properly integrates with parent applications like OpenHands
try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)

# LiteLLM configuration
DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514"
)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")


@dataclass
class ChunkingConfig:
    """Configuration for the chunking strategy."""

    target_chunk_tokens: int = (
        2500  # Target tokens per chunk (optimal for user messages)
    )
    max_chunk_tokens: int = (
        3000  # Hard limit per chunk (efficient for user messages + context)
    )
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
                    logger.error(
                        f"❌ BATCH TOO LARGE: {batch_tokens:,} tokens > 300K limit!"
                    )
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
                            logger.error(
                                f"Unexpected data format: {type(data)}, {data}"
                            )
                            raise ValueError(
                                f"Cannot extract embedding from response data: {type(data)}"
                            )
                else:
                    # Fallback for different response formats
                    logger.error(
                        f"Unexpected response format: {type(response)}, {response}"
                    )
                    raise ValueError(
                        f"Cannot extract embeddings from response: {type(response)}"
                    )

                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

        return embeddings

    def load_data(
        self,
        documents: List[Document],
        chunking_config: Optional[ChunkingConfig] = None,
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
            content = chunk_info["content"]
            texts_to_embed.append(content)
            # Content is already stored in metadata during chunk creation
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
        logger.info(
            f"🔢 Computing similarities with {len(self.embeddings)} stored embeddings..."
        )
        similarities = np.dot(self.embeddings, query_embedding)
        similarity_end_time = time.time()
        logger.info(
            f"⏱️  Matrix multiplication (np.dot): {similarity_end_time - similarity_start_time:.2f}s"
        )

        # Sort and get top results
        sort_start_time = time.time()
        top_indices = np.argsort(similarities)[::-1][:k]
        sort_end_time = time.time()
        logger.info(
            f"⏱️  Sorting and top-k selection: {sort_end_time - sort_start_time:.2f}s"
        )

        # Build results
        results_start_time = time.time()
        results = []
        for idx in top_indices:
            chunk_metadata = self.metadata[idx]
            result = RetrievalResult(
                content=chunk_metadata["content"],  # Content is now stored in metadata
                metadata=chunk_metadata,
                similarity=float(similarities[idx]),
            )
            results.append(result)
        results_end_time = time.time()
        logger.info(
            f"⏱️  Results construction: {results_end_time - results_start_time:.2f}s"
        )

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
                    if (
                        isinstance(session_content, dict)
                        and "convo_events" in session_content
                    ):
                        chunks = self._chunk_session_by_user_messages(
                            session_content, config, doc.doc_id, session_id
                        )
                        all_chunks.extend(chunks)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Could not parse document {doc.doc_id} for chunking: {e}"
                )
                continue

        logger.info(
            f"Created {len(all_chunks)} optimized chunks from {len(documents)} documents"
        )
        return all_chunks

    def _chunk_session_by_user_messages(
        self,
        session: Dict[str, Any],
        config: ChunkingConfig,
        doc_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced chunking: one chunk per user message with surrounding context.

        Args:
            session: Session data dictionary
            config: Chunking configuration
            doc_id: Document ID
            session_id: Session ID

        Returns:
            List of chunk dictionaries with enriched context metadata
        """
        events = session.get("convo_events", [])
        chunks = []

        # Find user message events
        for i, event in enumerate(events):
            if event.get("source") == "user":
                user_content = event.get("content", "")

                # Skip empty messages
                if not user_content.strip():
                    continue

                # Basic cleaning - remove very long system tags
                cleaned_content = self._clean_user_message(user_content)

                # Skip if empty after cleaning
                if not cleaned_content.strip():
                    logger.debug(f"Skipping empty message after cleaning at index {i}")
                    continue

                # Skip if still too long after cleaning
                if self._count_tokens(cleaned_content) > config.max_chunk_tokens:
                    logger.debug(
                        f"Skipping oversized message ({self._count_tokens(cleaned_content)} tokens) at index {i}"
                    )
                    continue

                # Extract surrounding context
                surrounding_context = self._extract_surrounding_context(
                    events, i, config
                )

                # Calculate total tokens (user message + context)
                context_tokens = self._count_tokens(surrounding_context)
                user_msg_tokens = self._count_tokens(cleaned_content)
                total_tokens = user_msg_tokens + context_tokens

                # If total exceeds 2500 tokens, condense the user message
                final_user_content = cleaned_content
                if total_tokens > config.target_chunk_tokens:
                    # Calculate how much we need to reduce the user message
                    # final_user_content = self._condense_if_needed(cleaned_content, config.target_chunk_tokens // 5, "current_user_msg")
                    # logger.debug(f"Condensed user message from {user_msg_tokens} to {self._count_tokens(final_user_content)} tokens")
                    continue  # we give up on this chunk

                # Load detailed session metadata from user_model_detailed folder
                detailed_metadata = self._load_session_metadata(doc_id, session_id)

                # Create chunk ID
                chunk_id = f"{doc_id}_{session_id}_msg_{i}"

                # Create condensed "all_one_need" field
                all_one_need = self._create_all_one_need_string(
                    final_user_content, detailed_metadata, surrounding_context, chunk_id
                )

                chunks.append(
                    {
                        "content": final_user_content,
                        "metadata": {
                            "doc_id": doc_id,
                            "session_id": session_id,
                            "chunk_id": chunk_id,
                            "chunk_type": "user_message",
                            "message_index": i,
                            "token_count": self._count_tokens(final_user_content),
                            # Rich session info from detailed metadata
                            "session_title": detailed_metadata.get("title", ""),
                            "repository_context": detailed_metadata.get(
                                "selected_repository", ""
                            ),
                            "selected_branch": detailed_metadata.get(
                                "selected_branch", ""
                            ),
                            "trigger": detailed_metadata.get("trigger", ""),
                            # Enhanced context
                            "surrounding_context": surrounding_context,
                            # Store content in metadata for retrieval
                            "content": final_user_content,
                            # Condensed everything in one field
                            "all_one_need": all_one_need,
                        },
                    }
                )

        return chunks

    def _clean_user_message(self, content: str) -> str:
        """
        Simple cleaning of user message content.

        Args:
            content: Raw user message content

        Returns:
            Cleaned content with system tags removed
        """
        # Remove common system tags and instructions
        system_patterns = [
            r"<REPOSITORY_INSTRUCTIONS>.*?</REPOSITORY_INSTRUCTIONS>",
            r"<REPOSITORY_INFO>.*?</REPOSITORY_INFO>",
            r"<RUNTIME_INFORMATION>.*?</RUNTIME_INFORMATION>",
            r"<EXTRA_INFO>.*?</EXTRA_INFO>",
            r"<ENVIRONMENT>.*?</ENVIRONMENT>",
            r"<CONTEXT>.*?</CONTEXT>",
            r"<system-reminder>.*?</system-reminder>",
            # Also remove common instruction patterns
            r"# OpenHands Glossary.*?(?=\n\n|\Z)",
            r"This repository contains the code for OpenHands.*?(?=\n\n|\Z)",
        ]

        cleaned = content

        for pattern in system_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace and empty lines
        cleaned = re.sub(
            r"\n\s*\n\s*\n", "\n\n", cleaned
        )  # Reduce multiple empty lines
        cleaned = cleaned.strip()

        return cleaned

    def _load_session_metadata(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Load detailed session metadata from user_model_detailed folder.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Session metadata dictionary
        """
        metadata_path = Path(
            f"data/user_model/user_model_detailed/{user_id}/{session_id}.json"
        )

        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    session_data = json.load(f)
                metadata = session_data.get("metadata", {})
                return dict(metadata) if metadata else {}
            except Exception as e:
                logger.warning(f"Failed to load session metadata for {session_id}: {e}")

        return {}

    def _create_all_one_need_string(
        self,
        user_content: str,
        metadata: Dict[str, Any],
        surrounding_context: str,
        chunk_id: str,
    ) -> str:
        """
        Create a comprehensive markdown-formatted string with all important information.

        Args:
            user_content: The user's message content
            metadata: Session metadata
            surrounding_context: Surrounding conversation context
            chunk_id: The chunk identifier

        Returns:
            Markdown-formatted string with all relevant info
        """
        lines = []

        # Header with chunk identification
        lines.append("## User Message")
        lines.append(user_content)

        # Surrounding context if available
        if surrounding_context:
            lines.append("")
            lines.append("### Context")
            lines.append(surrounding_context)

        # Metadata
        lines.append("")
        lines.append("### Metadata")
        lines.append(f"**Chunk ID**: `{chunk_id}`")
        if metadata.get("title"):
            lines.append(f"**Session**: {metadata['title']}")
        if metadata.get("selected_repository"):
            lines.append(f"**Repository**: {metadata['selected_repository']}")
        if metadata.get("selected_branch"):
            lines.append(f"**Branch**: {metadata['selected_branch']}")
        if metadata.get("trigger"):
            lines.append(f"**Trigger**: {metadata['trigger']}")

        return "\n".join(lines)

    def _extract_surrounding_context(
        self, events: List[Dict[str, Any]], user_msg_index: int, config: ChunkingConfig
    ) -> str:
        """
        Extract surrounding context as a condensed string.

        Args:
            events: List of conversation events
            user_msg_index: Index of the current user message
            config: Chunking configuration for token limits

        Returns:
            String containing surrounding context
        """
        context_parts = []
        # Use 1/5 of target chunk tokens for each context message
        context_token_limit = (
            config.target_chunk_tokens // 10
        )  # Smaller limit since we're combining

        # Find previous agent message
        for i in range(user_msg_index - 1, -1, -1):
            if events[i].get("source") == "assistant":
                content = events[i].get("content", "")
                if content.strip():
                    condensed = self._condense_if_needed(
                        content, context_token_limit, "prev_agent_msg"
                    )
                    context_parts.append(f"PREV_AGENT: {condensed}")
                break

        # Find previous user message
        for i in range(user_msg_index - 1, -1, -1):
            if events[i].get("source") == "user":
                content = events[i].get("content", "")
                cleaned_content = self._clean_user_message(content)
                if cleaned_content.strip():
                    condensed = self._condense_if_needed(
                        cleaned_content, context_token_limit, "prev_user_msg"
                    )
                    context_parts.append(f"PREV_USER: {condensed}")
                    break

        # Find next user message
        for i in range(user_msg_index + 1, len(events)):
            if events[i].get("source") == "user":
                content = events[i].get("content", "")
                cleaned_content = self._clean_user_message(content)
                if cleaned_content.strip():
                    condensed = self._condense_if_needed(
                        cleaned_content, context_token_limit, "next_user_msg"
                    )
                    context_parts.append(f"NEXT_USER: {condensed}")
                    break

        # Find next agent message
        for i in range(user_msg_index + 1, len(events)):
            if events[i].get("source") == "assistant":
                content = events[i].get("content", "")
                if content.strip():
                    condensed = self._condense_if_needed(
                        content, context_token_limit, "next_agent_msg"
                    )
                    context_parts.append(f"NEXT_AGENT: {condensed}")
                break

        return " | ".join(context_parts) if context_parts else ""

    def _condense_if_needed(
        self, content: str, max_tokens: int, content_type: str = "message"
    ) -> str:
        """
        LLM condensation to preserve main points while staying under token limit.

        Args:
            content: Content to potentially condense
            max_tokens: Maximum allowed tokens
            content_type: Type of content (for context-aware condensation)

        Returns:
            Condensed content if needed, original content otherwise
        """
        current_tokens = self._count_tokens(content)

        if current_tokens <= max_tokens:
            return content

        logger.info(
            f"LLM condensing {content_type}: {current_tokens} -> {max_tokens} tokens"
        )

        # # Use LLM to condense while preserving main points
        # condensed = self._llm_condense_content(content, max_tokens, content_type)

        # if condensed:
        #     logger.debug(f"LLM condensation successful: {current_tokens} -> {self._count_tokens(condensed)} tokens")
        #     return condensed
        # else:
        #     # Fallback to simple truncation if LLM fails
        #     logger.warning("LLM condensation failed, using truncation fallback")
        return self._simple_truncate(content, max_tokens)

    def _simple_truncate(self, content: str, max_tokens: int) -> str:
        """Simple truncation fallback."""
        chars_per_token = 4  # Rough approximation
        max_chars = max_tokens * chars_per_token

        if len(content) <= max_chars:
            return content

        return content[: max_chars - 3] + "..."

    def _llm_condense_content(
        self, content: str, max_tokens: int, content_type: str
    ) -> Optional[str]:
        """
        Use LLM to intelligently condense content while preserving context relevance.

        Args:
            content: Content to condense
            max_tokens: Target token count
            content_type: Type of content for context-aware prompting

        Returns:
            Condensed content or None if condensation fails
        """
        try:
            prompt = render_prompt(
                "message_condensation", max_tokens=max_tokens, content=content
            )

            # Use efficient model for condensation
            response = completion(
                model="gpt-4o-mini",  # Fast and cost-effective
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens + 50,  # Buffer for response formatting
            )

            condensed_content = response.choices[0].message.content
            if condensed_content and condensed_content.strip():
                return str(condensed_content.strip())
            else:
                return None

        except Exception as e:
            logger.warning(f"LLM condensation failed for {content_type}: {e}")
            return None

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


class RAGAgent:
    """
    Main RAG agent that orchestrates retrieval and generation using contextual embeddings.
    """

    def __init__(self, vector_db: VectorDB, llm_model: Optional[str] = None):
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
        logger.warning(
            f"Data directory {data_path} does not exist. Creating empty document list."
        )
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
    llm_model: str = DEFAULT_LLM_MODEL,
    chunking_config: Optional[ChunkingConfig] = None,
) -> RAGAgent:
    """
    Create and initialize a RAG agent with optimized chunking.

    Args:
        data_path: Path to processed data
        user_model_path: Path to user model base directory (contains user_model_detailed subdirectory)
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

    # Create vector database - simplified to always use basic VectorDB
    vector_db = VectorDB(name="rag_db")
    vector_db.load_data(documents, config)

    # Create RAG agent
    rag_agent = RAGAgent(vector_db, llm_model)

    return rag_agent


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Example of how to use the simplified RAG agent

        # Create RAG agent (LITELLM_API_KEY should be set in environment)
        agent = await create_rag_agent(
            data_path="data/processed_data",
            user_model_path="data/user_model",
        )

        # Example retrieval (no generation, just retrieval)
        query = "Hi Can you work on this website for me?"
        results = agent.retrieve(query, k=5)

        print(f"Query: {query}")
        print(f"Retrieved {len(results)} documents:")
        for i, result in enumerate(results[:3]):  # Show top 3
            print(f"{i+1}. Score: {result.similarity:.3f}")
            print(f"   Content: {result.content}")
            # Show all_one_need field
            all_one_need = result.metadata.get("all_one_need", "")
            if all_one_need:
                print(f"   All One Need: {all_one_need}")
            print()

    asyncio.run(main())
