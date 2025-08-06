#!/usr/bin/env python3
"""
Advanced RAG Query Interface for Jina v4 Pipeline

This module provides a comprehensive interface for querying the RAG system with:
- Interactive CLI interface
- Web API endpoints
- Advanced query processing
- Multimodal query support
- Result explanation and citation
- Query optimization and caching
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from jina_v4_pipeline import JinaV4Pipeline, EmbeddingConfig, DocumentChunk
from pdf_processor import PDFProcessor, ProcessingConfig, process_downloads_folder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for beautiful CLI output
console = Console()

@dataclass
class QueryResult:
    """Structured query result"""
    query: str
    results: List[Tuple[DocumentChunk, float]]
    query_time: float
    total_results: int
    query_embedding_time: float
    search_time: float
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    """API request model for queries"""
    query: str
    k: int = 10
    task_type: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True

class QueryResponse(BaseModel):
    """API response model"""
    query: str
    results: List[Dict[str, Any]]
    query_time: float
    total_results: int
    metadata: Dict[str, Any]

class RAGInterface:
    """Advanced RAG Interface with multiple interaction modes"""
    
    def __init__(self, 
                 pipeline_dir: str = "./saved_pipeline",
                 downloads_dir: str = "../downloads"):
        
        self.pipeline_dir = Path(pipeline_dir)
        self.downloads_dir = Path(downloads_dir)
        self.pipeline = None
        self.query_cache = {}  # Simple query cache
        self.query_history = []
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize or load the RAG pipeline"""
        try:
            if self.pipeline_dir.exists():
                # Load existing pipeline
                console.print(f"[green]Loading existing pipeline from {self.pipeline_dir}[/green]")
                config = EmbeddingConfig()
                self.pipeline = JinaV4Pipeline(config, downloads_dir=str(self.downloads_dir))
                self.pipeline.load_pipeline(str(self.pipeline_dir))
                console.print("[green]Pipeline loaded successfully![/green]")
                
                # Check if pipeline is empty and needs to be rebuilt
                if len(self.pipeline.vector_store.documents) == 0:
                    console.print("[yellow]Pipeline is empty. Processing documents...[/yellow]")
                    self._process_documents_and_rebuild()
            else:
                # Create new pipeline
                console.print("[yellow]No existing pipeline found. Creating new pipeline...[/yellow]")
                self._create_new_pipeline()
                
        except Exception as e:
            console.print(f"[red]Failed to initialize pipeline: {e}[/red]")
            raise
    
    def _create_new_pipeline(self):
        """Create a new pipeline from scratch"""
        try:
            # Configuration for advanced features
            config = EmbeddingConfig(
                task_type="retrieval",
                embedding_dim=2048,
                batch_size=4,
                output_type="single"
            )
            
            # Initialize pipeline
            self.pipeline = JinaV4Pipeline(config, downloads_dir=str(self.downloads_dir))
            
            # Process documents and build index
            self._process_documents_and_rebuild()
            
        except Exception as e:
            console.print(f"[red]Failed to create pipeline: {e}[/red]")
            raise
    
    def _process_documents_and_rebuild(self):
        """Process documents and rebuild the pipeline index"""
        try:
            # Process documents
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                task = progress.add_task("Processing documents...", total=None)
                
                # Process all PDFs in downloads folder
                processing_config = ProcessingConfig(
                    extract_images=True,
                    extract_tables=True,
                    use_ocr=True,
                    max_chunk_size=1200,
                    overlap_size=150
                )
                
                chunks = process_downloads_folder(
                    str(self.downloads_dir),
                    config=processing_config
                )
                
                progress.update(task, description=f"Building index from {len(chunks)} chunks...")
                
                # Build index
                self.pipeline.build_index(chunks)
                
                # Save pipeline
                progress.update(task, description="Saving pipeline...")
                self.pipeline.save_pipeline(str(self.pipeline_dir))
            
            console.print(f"[green]Pipeline processed {len(chunks)} document chunks successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to rebuild pipeline: {e}[/red]")
            raise
    
    def query(self, 
              query_text: str, 
              k: int = 10,
              task_type: Optional[str] = None,
              filters: Optional[Dict] = None,
              use_cache: bool = True) -> QueryResult:
        """
        Execute a query against the RAG system
        
        Args:
            query_text: The query string
            k: Number of results to return
            task_type: Override task type for query
            filters: Optional filters for results
            use_cache: Whether to use query cache
            
        Returns:
            QueryResult object with results and metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query_text}:{k}:{task_type}:{json.dumps(filters, sort_keys=True)}"
        if use_cache and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result.metadata['from_cache'] = True
            return cached_result
        
        try:
            # Measure embedding time
            embed_start = time.time()
            
            # Execute query
            results = self.pipeline.query(query_text, k=k, task_type=task_type)
            
            embed_time = time.time() - embed_start
            search_time = time.time() - start_time - embed_time
            total_time = time.time() - start_time
            
            # Apply additional filters if specified
            if filters:
                filtered_results = []
                for doc, score in results:
                    match = True
                    for key, value in filters.items():
                        if hasattr(doc, key):
                            if getattr(doc, key) != value:
                                match = False
                                break
                        elif key in doc.metadata:
                            if doc.metadata[key] != value:
                                match = False
                                break
                    if match:
                        filtered_results.append((doc, score))
                results = filtered_results[:k]
            
            # Create result object
            query_result = QueryResult(
                query=query_text,
                results=results,
                query_time=total_time,
                total_results=len(results),
                query_embedding_time=embed_time,
                search_time=search_time,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'task_type': task_type or self.pipeline.config.task_type,
                    'filters': filters,
                    'from_cache': False
                }
            )
            
            # Cache result
            if use_cache:
                self.query_cache[cache_key] = query_result
            
            # Add to history
            self.query_history.append(query_result)
            
            return query_result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def explain_results(self, query_result: QueryResult) -> Dict[str, Any]:
        """Generate explanations for query results"""
        explanations = {
            'query_analysis': self._analyze_query(query_result.query),
            'result_analysis': [],
            'retrieval_strategy': {
                'task_type': query_result.metadata.get('task_type', 'retrieval'),
                'embedding_model': 'jina-embeddings-v4',
                'search_method': 'cosine_similarity'
            },
            'performance': {
                'total_time_ms': query_result.query_time * 1000,
                'embedding_time_ms': query_result.query_embedding_time * 1000,
                'search_time_ms': query_result.search_time * 1000,
                'results_found': query_result.total_results
            }
        }
        
        # Analyze each result
        for i, (doc, score) in enumerate(query_result.results[:5]):  # Analyze top 5
            analysis = {
                'rank': i + 1,
                'relevance_score': score,
                'document_type': doc.chunk_type,
                'source': doc.source_file,
                'page': doc.page_number,
                'language': doc.language,
                'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                'relevance_factors': self._analyze_relevance(query_result.query, doc, score)
            }
            explanations['result_analysis'].append(analysis)
        
        return explanations
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics"""
        words = query.lower().split()
        
        analysis = {
            'word_count': len(words),
            'character_count': len(query),
            'query_type': 'unknown',
            'key_terms': words[:5],  # First 5 words as key terms
            'complexity': 'simple' if len(words) <= 5 else 'medium' if len(words) <= 10 else 'complex'
        }
        
        # Determine query type based on keywords
        if any(word in words for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            analysis['query_type'] = 'question'
        elif any(word in words for word in ['find', 'search', 'locate', 'show']):
            analysis['query_type'] = 'search'
        elif any(word in words for word in ['define', 'explain', 'describe']):
            analysis['query_type'] = 'definition'
        else:
            analysis['query_type'] = 'keyword'
        
        return analysis
    
    def _analyze_relevance(self, query: str, doc: DocumentChunk, score: float) -> List[str]:
        """Analyze why a document is relevant"""
        factors = []
        
        query_words = set(query.lower().split())
        doc_words = set(doc.content.lower().split())
        
        # Check for word overlap
        overlap = query_words.intersection(doc_words)
        if overlap:
            factors.append(f"Keyword overlap: {', '.join(list(overlap)[:3])}")
        
        # High relevance score
        if score > 0.8:
            factors.append("High semantic similarity")
        elif score > 0.6:
            factors.append("Moderate semantic similarity")
        else:
            factors.append("Lower semantic similarity")
        
        # Document type relevance
        if doc.chunk_type == "text":
            factors.append("Text content match")
        elif doc.chunk_type == "image":
            factors.append("Image/visual content")
        elif doc.chunk_type == "table":
            factors.append("Structured data match")
        
        return factors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline and query statistics"""
        stats = {
            'pipeline_info': {
                'total_documents': len(self.pipeline.processed_documents) if self.pipeline else 0,
                'vector_store_type': self.pipeline.vector_store.store_type if self.pipeline else None,
                'embedding_dimension': self.pipeline.config.embedding_dim if self.pipeline else None
            },
            'query_stats': {
                'total_queries': len(self.query_history),
                'average_query_time': sum(q.query_time for q in self.query_history) / len(self.query_history) if self.query_history else 0,
                'cache_hits': len([q for q in self.query_history if q.metadata.get('from_cache', False)]),
                'cache_size': len(self.query_cache)
            },
            'document_stats': self._get_document_stats() if self.pipeline else {}
        }
        
        return stats
    
    def _get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not self.pipeline or not self.pipeline.processed_documents:
            return {}
        
        docs = self.pipeline.processed_documents
        
        # Count by type
        type_counts = {}
        language_counts = {}
        source_counts = {}
        
        for doc in docs:
            type_counts[doc.chunk_type] = type_counts.get(doc.chunk_type, 0) + 1
            language_counts[doc.language] = language_counts.get(doc.language, 0) + 1  
            source_counts[doc.source_file] = source_counts.get(doc.source_file, 0) + 1
        
        return {
            'total_chunks': len(docs),
            'chunk_types': type_counts,
            'languages': language_counts,
            'source_files': len(source_counts),
            'chunks_per_file': source_counts
        }

# CLI Interface
@click.group()
def cli():
    """Jina v4 RAG Pipeline - Advanced Retrieval System"""
    pass

@cli.command()
@click.option('--downloads-dir', default='../downloads', help='Downloads directory')
@click.option('--pipeline-dir', default='./saved_pipeline', help='Pipeline save directory')
def interactive(downloads_dir, pipeline_dir):
    """Start interactive RAG query session"""
    console.print(Panel.fit(
        "[bold blue]Jina v4 RAG Pipeline[/bold blue]\n"
        "Advanced Multimodal Retrieval System",
        border_style="blue"
    ))
    
    # Initialize interface
    interface = RAGInterface(pipeline_dir, downloads_dir)
    
    # Show statistics
    stats = interface.get_statistics()
    
    stats_table = Table(title="Pipeline Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Documents", str(stats['pipeline_info']['total_documents']))
    stats_table.add_row("Embedding Dimension", str(stats['pipeline_info']['embedding_dimension']))
    stats_table.add_row("Vector Store", stats['pipeline_info']['vector_store_type'])
    
    console.print(stats_table)
    console.print()
    
    # Interactive query loop
    while True:
        try:
            query = console.input("[bold cyan]Enter your query (or 'quit' to exit): [/bold cyan]")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            # Execute query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("Searching...", total=None)
                result = interface.query(query, k=10)
            
            # Display results
            console.print(f"\n[bold green]Found {result.total_results} results in {result.query_time:.3f}s[/bold green]")
            
            if result.results:
                results_table = Table(title=f"Results for: {query}")
                results_table.add_column("Rank", width=6)
                results_table.add_column("Score", width=8)
                results_table.add_column("Source", width=25)
                results_table.add_column("Type", width=8)
                results_table.add_column("Content Preview", width=50)
                
                for i, (doc, score) in enumerate(result.results[:10], 1):
                    content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                    results_table.add_row(
                        str(i),
                        f"{score:.3f}",
                        f"{doc.source_file} (p.{doc.page_number})",
                        doc.chunk_type,
                        content_preview
                    )
                
                console.print(results_table)
                
                # Ask if user wants explanation
                explain = console.input("\n[dim]Show detailed explanation? (y/N): [/dim]")
                if explain.lower().startswith('y'):
                    explanation = interface.explain_results(result)
                    
                    # Display explanation
                    console.print("\n[bold]Query Analysis:[/bold]")
                    query_analysis = explanation['query_analysis']
                    console.print(f"Type: {query_analysis['query_type']}")
                    console.print(f"Complexity: {query_analysis['complexity']}")
                    console.print(f"Key terms: {', '.join(query_analysis['key_terms'])}")
                    
                    console.print("\n[bold]Top Results Analysis:[/bold]")
                    for analysis in explanation['result_analysis'][:3]:
                        console.print(f"\nRank {analysis['rank']} (Score: {analysis['relevance_score']:.3f}):")
                        console.print(f"  Source: {analysis['source']} (page {analysis['page']})")
                        console.print(f"  Relevance factors: {', '.join(analysis['relevance_factors'])}")
            else:
                console.print("[yellow]No results found. Try different keywords.[/yellow]")
            
            console.print("\n" + "-" * 80 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

@cli.command()
@click.option('--port', default=8000, help='API server port')
@click.option('--host', default='127.0.0.1', help='API server host')
@click.option('--downloads-dir', default='../downloads', help='Downloads directory')
@click.option('--pipeline-dir', default='./saved_pipeline', help='Pipeline save directory')
def api(port, host, downloads_dir, pipeline_dir):
    """Start FastAPI web server"""
    
    # Initialize interface
    interface = RAGInterface(pipeline_dir, downloads_dir)
    
    # Create FastAPI app
    app = FastAPI(
        title="Jina v4 RAG API",
        description="Advanced multimodal retrieval system using Jina Embeddings v4",
        version="1.0.0"
    )
    
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(request: QueryRequest):
        """Query the RAG system"""
        try:
            result = interface.query(
                query_text=request.query,
                k=request.k,
                task_type=request.task_type,
                filters=request.filters
            )
            
            # Format results for API
            formatted_results = []
            for doc, score in result.results:
                result_data = {
                    'id': doc.id,
                    'content': doc.content,
                    'score': score,
                    'source_file': doc.source_file,
                    'page_number': doc.page_number,
                    'chunk_type': doc.chunk_type,
                    'language': doc.language
                }
                
                if request.include_metadata:
                    result_data['metadata'] = doc.metadata
                
                formatted_results.append(result_data)
            
            return QueryResponse(
                query=result.query,
                results=formatted_results,
                query_time=result.query_time,
                total_results=result.total_results,
                metadata=result.metadata
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats")
    async def stats_endpoint():
        """Get pipeline statistics"""
        return interface.get_statistics()
    
    @app.post("/explain")
    async def explain_endpoint(request: QueryRequest):
        """Get explanation for query results"""
        try:
            result = interface.query(
                query_text=request.query,
                k=request.k,
                task_type=request.task_type,
                filters=request.filters
            )
            
            explanation = interface.explain_results(result)
            return explanation
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    console.print(f"[green]Starting API server on http://{host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli()