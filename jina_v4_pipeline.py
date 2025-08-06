#!/usr/bin/env python3
"""
Jina Embeddings v4 Advanced RAG Pipeline

This module implements a comprehensive RAG pipeline using Jina Embeddings v4
with support for multimodal (text + image) processing, multilingual content,
and advanced retrieval capabilities.

Features:
- Multimodal embedding generation (text + images)
- Task-specific adapters (retrieval, similarity, code)
- Multi-vector and single-vector embeddings
- Multilingual support (29+ languages)
- Advanced chunking strategies
- Vector storage with FAISS/ChromaDB
- Hybrid search capabilities
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from transformers import AutoModel, AutoTokenizer
from PIL import Image
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from langdetect import detect
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for Jina v4 embeddings"""
    model_name: str = "jinaai/jina-embeddings-v4"
    embedding_dim: int = 2048  # Can be truncated to 128
    max_length: int = 32768
    task_type: str = "retrieval"  # retrieval, similarity, code
    output_type: str = "single"  # single, multi
    batch_size: int = 8
    device: str = "auto"
    trust_remote_code: bool = True
    
@dataclass 
class DocumentChunk:
    """Represents a processed document chunk"""
    id: str
    content: str
    chunk_type: str  # text, image, table, code
    source_file: str
    page_number: int
    language: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    multi_vector_embedding: Optional[np.ndarray] = None

class JinaV4Embedder:
    """Advanced Jina v4 embedder with multimodal capabilities"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
        return device
    
    def _load_model(self):
        """Load Jina v4 model and tokenizer"""
        try:
            logger.info(f"Loading Jina v4 model: {self.config.model_name}")
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            logger.info("Jina v4 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Jina v4 model: {e}")
            raise
    
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode text using Jina v4 with task-specific optimization
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters (task, truncate_dim, etc.)
        
        Returns:
            numpy array of embeddings
        """
        try:
            # Detect language for each text (multilingual optimization)
            languages = []
            for text in texts:
                try:
                    lang = detect(text)
                    languages.append(lang)
                except:
                    languages.append("en")  # default to English
            
            # Encode with task-specific adapter
            embeddings = self.model.encode_text(
                texts=texts,
                task=self.config.task_type,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                **kwargs
            )
            
            # Truncate dimensions if specified
            truncate_dim = kwargs.get('truncate_dim')
            if truncate_dim and truncate_dim < self.config.embedding_dim:
                embeddings = embeddings[:, :truncate_dim]
            
            logger.info(f"Encoded {len(texts)} texts with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_image(self, images: List[Union[str, Image.Image]], **kwargs) -> np.ndarray:
        """
        Encode images using Jina v4 multimodal capabilities
        
        Args:
            images: List of image paths or PIL Images
            **kwargs: Additional encoding parameters
        
        Returns:
            numpy array of image embeddings
        """
        try:
            # Process images
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    # Load from path
                    processed_images.append(Image.open(img).convert("RGB"))
                else:
                    processed_images.append(img)
            
            # Encode images with multimodal model
            embeddings = self.model.encode_image(
                images=processed_images,
                task=self.config.task_type,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                **kwargs
            )
            
            logger.info(f"Encoded {len(images)} images with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode images: {e}")
            raise
    
    def encode_multimodal(self, 
                         texts: List[str], 
                         images: List[Union[str, Image.Image]], 
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode both text and images for multimodal retrieval
        
        Args:
            texts: List of text strings
            images: List of images
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        text_embeddings = self.encode_text(texts, **kwargs)
        image_embeddings = self.encode_image(images, **kwargs)
        
        return text_embeddings, image_embeddings
    
    def get_multi_vector_embeddings(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Generate multi-vector embeddings for fine-grained retrieval
        
        Args:
            texts: Input texts
            **kwargs: Additional parameters
        
        Returns:
            List of multi-vector embeddings per text
        """
        try:
            # Enable multi-vector output
            embeddings = self.model.encode_text(
                texts=texts,
                task=self.config.task_type,
                output_type="multi",
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                **kwargs
            )
            
            logger.info(f"Generated multi-vector embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate multi-vector embeddings: {e}")
            raise

class VectorStore:
    """Advanced vector storage and retrieval system"""
    
    def __init__(self, embedding_dim: int = 2048, store_type: str = "faiss"):
        self.embedding_dim = embedding_dim
        self.store_type = store_type
        self.documents: List[DocumentChunk] = []
        self.document_index = {}  # id -> document mapping
        
        if store_type == "faiss":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            self.index = faiss.IndexIVFFlat(self.index, embedding_dim, 100)  # Faster search
        elif store_type == "chromadb":
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.create_collection("jina_v4_embeddings")
        
        logger.info(f"Initialized {store_type} vector store with dimension {embedding_dim}")
    
    def add_documents(self, documents: List[DocumentChunk]):
        """Add documents with embeddings to the vector store"""
        try:
            embeddings = []
            doc_ids = []
            metadatas = []
            
            for doc in documents:
                if doc.embedding is not None:
                    self.documents.append(doc)
                    self.document_index[doc.id] = doc
                    
                    embeddings.append(doc.embedding)
                    doc_ids.append(doc.id)
                    
                    # Prepare metadata
                    metadata = {
                        "content": doc.content[:500],  # Truncate for storage
                        "chunk_type": doc.chunk_type,
                        "source_file": doc.source_file,
                        "page_number": doc.page_number,
                        "language": doc.language,
                        **doc.metadata
                    }
                    metadatas.append(metadata)
            
            if not embeddings:
                logger.warning("No documents with embeddings to add")
                return
            
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            
            if self.store_type == "faiss":
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                if not self.index.is_trained:
                    self.index.train(embeddings_array)
                
                self.index.add(embeddings_array)
                
            elif self.store_type == "chromadb":
                self.collection.add(
                    embeddings=embeddings_array.tolist(),
                    documents=[doc.content for doc in documents if doc.embedding is not None],
                    metadatas=metadatas,
                    ids=doc_ids
                )
            
            logger.info(f"Added {len(embeddings)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10, 
               filters: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if self.store_type == "faiss":
                # Normalize query for cosine similarity
                query_normalized = query_embedding.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(query_normalized)
                
                scores, indices = self.index.search(query_normalized, k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.documents):
                        doc = self.documents[idx]
                        
                        # Apply filters if specified
                        if filters and not self._match_filters(doc, filters):
                            continue
                            
                        results.append((doc, float(score)))
                
                return results[:k]
                
            elif self.store_type == "chromadb":
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=filters
                )
                
                formatted_results = []
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    score = results['distances'][0][i]
                    doc = self.document_index.get(doc_id)
                    
                    if doc:
                        formatted_results.append((doc, 1.0 - score))  # Convert distance to similarity
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def _match_filters(self, doc: DocumentChunk, filters: Dict) -> bool:
        """Check if document matches filters"""
        for key, value in filters.items():
            if hasattr(doc, key):
                if getattr(doc, key) != value:
                    return False
            elif key in doc.metadata:
                if doc.metadata[key] != value:
                    return False
        return True
    
    def save_index(self, filepath: str):
        """Save vector index to disk"""
        try:
            if self.store_type == "faiss":
                faiss.write_index(self.index, f"{filepath}.faiss")
                
                # Save document metadata
                doc_metadata = [asdict(doc) for doc in self.documents]
                with open(f"{filepath}_metadata.json", "w") as f:
                    json.dump({
                        "documents": doc_metadata,
                        "config": {
                            "embedding_dim": self.embedding_dim,
                            "store_type": self.store_type,
                            "num_documents": len(self.documents)
                        }
                    }, f, indent=2)
            
            elif self.store_type == "chromadb":
                # ChromaDB handles persistence automatically
                pass
                
            logger.info(f"Saved vector index to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            raise
    
    def load_index(self, filepath: str):
        """Load vector index from disk"""
        try:
            if self.store_type == "faiss":
                self.index = faiss.read_index(f"{filepath}.faiss")
                
                # Load document metadata
                with open(f"{filepath}_metadata.json", "r") as f:
                    data = json.load(f)
                
                self.documents = []
                self.document_index = {}
                
                for doc_data in data["documents"]:
                    # Convert embedding back to numpy array
                    if doc_data.get("embedding"):
                        doc_data["embedding"] = np.array(doc_data["embedding"])
                    
                    doc = DocumentChunk(**doc_data)
                    self.documents.append(doc)
                    self.document_index[doc.id] = doc
                
            logger.info(f"Loaded vector index from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            raise

class JinaV4Pipeline:
    """Complete Jina v4 RAG Pipeline"""
    
    def __init__(self, 
                 config: EmbeddingConfig,
                 vector_store_type: str = "faiss",
                 downloads_dir: str = "../downloads"):
        
        self.config = config
        self.downloads_dir = Path(downloads_dir)
        self.embedder = JinaV4Embedder(config)
        self.vector_store = VectorStore(config.embedding_dim, vector_store_type)
        self.processed_documents = []
        
        logger.info("Jina v4 RAG Pipeline initialized")
    
    def process_documents(self, document_paths: Optional[List[str]] = None) -> List[DocumentChunk]:
        """
        Process documents from downloads folder
        
        Args:
            document_paths: Specific paths to process, or None for all
            
        Returns:
            List of processed document chunks
        """
        if document_paths is None:
            document_paths = list(self.downloads_dir.glob("*.pdf"))
        
        all_chunks = []
        
        for doc_path in tqdm(document_paths, desc="Processing documents"):
            try:
                chunks = self._process_single_document(doc_path)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
                continue
        
        logger.info(f"Processed {len(document_paths)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _process_single_document(self, doc_path: Path) -> List[DocumentChunk]:
        """Process a single document into chunks"""
        # This will be implemented in the PDF processing utilities
        # For now, return a placeholder
        return []
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build vector index from document chunks"""
        try:
            # Generate embeddings for all chunks
            text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text"]
            image_chunks = [chunk for chunk in chunks if chunk.chunk_type == "image"]
            
            if text_chunks:
                texts = [chunk.content for chunk in text_chunks]
                text_embeddings = self.embedder.encode_text(texts)
                
                for chunk, embedding in zip(text_chunks, text_embeddings):
                    chunk.embedding = embedding
            
            if image_chunks:
                # Process image chunks (implement when PDF processor is ready)
                pass
            
            # Add all chunks to vector store
            self.vector_store.add_documents(chunks)
            self.processed_documents = chunks
            
            logger.info(f"Built vector index with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def query(self, 
              query_text: str, 
              k: int = 10,
              task_type: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Query the RAG system
        
        Args:
            query_text: Query string
            k: Number of results
            task_type: Override task type for query
            
        Returns:
            List of (document, relevance_score) tuples
        """
        try:
            # Use task-specific encoding for query
            original_task = self.config.task_type
            if task_type:
                self.config.task_type = task_type
            
            query_embedding = self.embedder.encode_text([query_text])[0]
            
            # Restore original task
            if task_type:
                self.config.task_type = original_task
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return []
    
    def save_pipeline(self, save_dir: str):
        """Save the complete pipeline"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector index
        self.vector_store.save_index(str(save_path / "vector_index"))
        
        # Save pipeline config
        config_data = {
            "embedding_config": asdict(self.config),
            "pipeline_metadata": {
                "created_at": datetime.now().isoformat(),
                "num_documents": len(self.processed_documents),
                "downloads_dir": str(self.downloads_dir)
            }
        }
        
        with open(save_path / "pipeline_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Pipeline saved to {save_dir}")
    
    def load_pipeline(self, save_dir: str):
        """Load a saved pipeline"""
        save_path = Path(save_dir)
        
        # Load vector index
        self.vector_store.load_index(str(save_path / "vector_index"))
        
        logger.info(f"Pipeline loaded from {save_dir}")


def main():
    """Example usage of the Jina v4 Pipeline"""
    # Configuration for advanced features
    config = EmbeddingConfig(
        task_type="retrieval",
        embedding_dim=2048,
        batch_size=4,
        output_type="single"
    )
    
    # Initialize pipeline
    pipeline = JinaV4Pipeline(config)
    
    # Process documents
    chunks = pipeline.process_documents()
    
    # Build index
    pipeline.build_index(chunks)
    
    # Example queries
    queries = [
        "pericyte regulation in vascular function",
        "blood-brain barrier permeability mechanisms", 
        "renal ischemia reperfusion injury"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = pipeline.query(query, k=5)
        
        for doc, score in results:
            print(f"Score: {score:.3f} | {doc.source_file} | {doc.content[:100]}...")
    
    # Save pipeline
    pipeline.save_pipeline("./saved_pipeline")


if __name__ == "__main__":
    main()