# Jina Embeddings v4 Advanced RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) pipeline built with Jina Embeddings v4, featuring multimodal document processing, advanced chunking strategies, and intelligent retrieval capabilities.

## Features

### 🚀 Advanced Jina v4 Capabilities
- **Multimodal Embeddings**: Process both text and images using Jina v4's universal embedding model
- **Task-Specific Adapters**: Optimized for retrieval, similarity, and code understanding tasks
- **Multi-Vector Support**: Fine-grained embeddings for enhanced retrieval precision
- **Multilingual Processing**: Support for 29+ languages with automatic language detection
- **High-Performance**: 3.8B parameter model with 32,768 token context length

### 📄 Comprehensive PDF Processing
- **Advanced Text Extraction**: Layout-aware text extraction with formatting preservation
- **Image Extraction**: Automatic extraction and OCR of images and figures
- **Table Detection**: Intelligent table parsing and structured data extraction
- **Smart Chunking**: Context-aware text chunking with semantic boundaries
- **Multimodal Content**: Unified processing of text, images, and structured data

### 🔍 Intelligent Retrieval
- **Vector Storage**: FAISS and ChromaDB support for scalable vector search
- **Hybrid Search**: Combine semantic similarity with metadata filtering
- **Query Optimization**: Task-specific query embedding with caching
- **Result Explanation**: Detailed analysis of retrieval results and relevance

### 🖥️ Multiple Interfaces
- **Interactive CLI**: Rich terminal interface with real-time results
- **REST API**: FastAPI-based web service for integration
- **Batch Processing**: Bulk document processing and indexing
- **Query Analytics**: Comprehensive query performance analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Tesseract OCR (for image text extraction)

### Setup
```bash
# Clone and navigate to the jina directory
cd jina

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install Tesseract (macOS)
brew install tesseract

# Install Tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Optional: Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

## Quick Start

### 1. Process Documents and Build Index
```python
from jina_v4_pipeline import JinaV4Pipeline, EmbeddingConfig
from pdf_processor import process_downloads_folder, ProcessingConfig

# Configure embedding model
config = EmbeddingConfig(
    task_type="retrieval",
    embedding_dim=2048,
    batch_size=4
)

# Initialize pipeline
pipeline = JinaV4Pipeline(config, downloads_dir="../downloads")

# Process PDFs and build index
processing_config = ProcessingConfig(
    extract_images=True,
    extract_tables=True,
    use_ocr=True,
    max_chunk_size=1200
)

chunks = process_downloads_folder(config=processing_config)
pipeline.build_index(chunks)

# Save for future use
pipeline.save_pipeline("./saved_pipeline")
```

### 2. Query the System
```python
# Load existing pipeline
pipeline.load_pipeline("./saved_pipeline")

# Execute queries
results = pipeline.query("pericyte regulation in blood vessels", k=10)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Source: {doc.source_file} (page {doc.page_number})")
    print(f"Content: {doc.content[:200]}...")
    print("-" * 80)
```

### 3. Interactive CLI
```bash
python rag_interface.py interactive --downloads-dir ../downloads
```

### 4. Web API Server
```bash
python rag_interface.py api --port 8000
```

Then query via HTTP:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "blood brain barrier permeability",
    "k": 5,
    "task_type": "retrieval"
  }'
```

## Configuration

### Embedding Configuration
```python
from jina_v4_pipeline import EmbeddingConfig

config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    embedding_dim=2048,  # Can truncate to 128 for efficiency
    max_length=32768,    # Maximum context length
    task_type="retrieval",  # retrieval, similarity, code
    output_type="single",   # single or multi-vector
    batch_size=8,
    device="auto"  # auto, cuda, cpu, mps
)
```

### PDF Processing Configuration
```python
from pdf_processor import ProcessingConfig

config = ProcessingConfig(
    extract_images=True,        # Extract and OCR images
    extract_tables=True,        # Parse tables
    use_ocr=True,              # OCR for scanned content
    min_image_size=(100, 100),  # Minimum image dimensions
    max_chunk_size=1500,        # Maximum chunk size in characters
    overlap_size=200,           # Overlap between chunks
    preserve_formatting=True,   # Maintain document structure
    language_detection=True,    # Auto-detect language
    quality_threshold=0.7       # OCR quality threshold
)
```

## Advanced Usage

### Multimodal Queries
```python
# Query with both text and image context
text_embeddings, image_embeddings = pipeline.embedder.encode_multimodal(
    texts=["vascular structure analysis"],
    images=["path/to/diagram.png"]
)
```

### Multi-Vector Embeddings
```python
# Generate fine-grained embeddings
multi_vector_embeddings = pipeline.embedder.get_multi_vector_embeddings(
    texts=["Complex document with multiple concepts..."]
)
```

### Custom Task Types
```python
# Use different task-specific adapters
similarity_results = pipeline.query(
    "cell membrane structure", 
    task_type="similarity"
)

code_results = pipeline.query(
    "python data processing", 
    task_type="code"
)
```

### Advanced Filtering
```python
# Filter by document properties
results = pipeline.query(
    "renal function",
    k=10,
    filters={
        'chunk_type': 'text',
        'language': 'en',
        'page_number': [1, 2, 3]  # First three pages only
    }
)
```

## API Reference

### Core Classes

#### `JinaV4Pipeline`
Main pipeline orchestrator
- `__init__(config, vector_store_type, downloads_dir)`
- `process_documents(document_paths)` 
- `build_index(chunks)`
- `query(query_text, k, task_type)`
- `save_pipeline(save_dir)`
- `load_pipeline(save_dir)`

#### `JinaV4Embedder`
Advanced embedding generation
- `encode_text(texts, **kwargs)`
- `encode_image(images, **kwargs)`
- `encode_multimodal(texts, images, **kwargs)`
- `get_multi_vector_embeddings(texts, **kwargs)`

#### `PDFProcessor`
Comprehensive PDF processing
- `process_pdf(pdf_path, output_dir)`
- `_extract_text_from_page(page, page_num, pdf_path)`
- `_extract_images_from_page(page, page_num, pdf_path, output_dir)`
- `_extract_tables_from_page(pdf_path, page_num)`

#### `RAGInterface`
User interface and API
- `query(query_text, k, task_type, filters)`
- `explain_results(query_result)`
- `get_statistics()`

## Performance Optimization

### GPU Acceleration
```python
config = EmbeddingConfig(device="cuda")  # Use GPU
```

### Batch Processing
```python
config = EmbeddingConfig(batch_size=16)  # Larger batches for GPU
```

### Dimension Reduction
```python
# Reduce embedding dimensions for speed
embeddings = embedder.encode_text(texts, truncate_dim=512)
```

### Query Caching
```python
interface = RAGInterface()
result = interface.query("query", use_cache=True)  # Enable caching
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Jina v4 RAG Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ PDF Files   │───▶│ PDF Processor│───▶│ Document     │   │
│  │ (Downloads) │    │              │    │ Chunks       │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│                            │                     │          │
│                            ▼                     ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Images/OCR  │    │ Tables       │    │ Text Chunks  │   │
│  │ Extraction  │    │ Extraction   │    │ w/ Overlap   │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│                            │                     │          │
│                            ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Jina v4 Embedder (3.8B Parameters)          │ │
│  │  • Multimodal (Text + Image)                           │ │
│  │  • Task-Specific Adapters                              │ │
│  │  • Multi-Vector Support                                │ │
│  │  • 29+ Languages                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Vector Store (FAISS/ChromaDB)             │ │
│  │  • Cosine Similarity Search                            │ │
│  │  • Metadata Filtering                                  │ │
│  │  • Scalable Indexing                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                RAG Interface                            │ │
│  │  • Interactive CLI                                     │ │
│  │  • REST API                                           │ │
│  │  • Query Analytics                                    │ │
│  │  • Result Explanation                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   config = EmbeddingConfig(batch_size=2, device="cpu")
   ```

2. **Tesseract Not Found**
   ```bash
   # Linux/Mac
   which tesseract
   export PATH=$PATH:/usr/local/bin
   
   # Windows
   pip install pytesseract
   # Add Tesseract to PATH
   ```

3. **Model Download Issues**
   ```bash
   # Pre-download model
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-embeddings-v4')"
   ```

4. **Memory Issues with Large PDFs**
   ```python
   config = ProcessingConfig(max_chunk_size=800)  # Smaller chunks
   ```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project follows Jina AI's licensing terms. Commercial use requires Jina AI API access.

## Citation

```bibtex
@article{jina2024embeddings,
  title={Jina Embeddings v4: Universal Embeddings for Multimodal Retrieval},
  author={Jina AI},
  year={2024},
  url={https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval/}
}
```

## Support

- Documentation: [Jina AI Docs](https://jina.ai/models/jina-embeddings-v4/)
- Issues: GitHub Issues
- Community: [Jina Discord](https://discord.gg/jina)