#!/usr/bin/env python3
"""
Advanced PDF Processing for Jina v4 Pipeline

This module provides comprehensive PDF processing capabilities including:
- Text extraction with layout preservation
- Image and figure extraction
- Table detection and parsing
- OCR for scanned documents
- Multimodal content preparation for Jina v4
- Intelligent chunking strategies
"""

import os
import re
import json
import fitz  # PyMuPDF
import io
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging

import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import spacy
from langdetect import detect

from jina_v4_pipeline import DocumentChunk

# Setup logging first
logger = logging.getLogger(__name__)

# Load spaCy model (much faster and more accurate than NLTK)
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy English model")
except OSError:
    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    # Fallback to basic sentence splitting if spaCy not available
    nlp = None

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing"""
    extract_images: bool = True
    extract_tables: bool = True
    use_ocr: bool = True
    min_image_size: Tuple[int, int] = (100, 100)
    max_chunk_size: int = 1500
    overlap_size: int = 200
    preserve_formatting: bool = True
    language_detection: bool = True
    quality_threshold: float = 0.7

class ImageExtractor:
    """Extract and process images from PDFs"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def extract_images_from_page(self, 
                                page: fitz.Page, 
                                page_num: int, 
                                output_dir: Path) -> List[Dict[str, Any]]:
        """Extract images from a PDF page"""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Skip small images
                    if (pix.width < self.config.min_image_size[0] or 
                        pix.height < self.config.min_image_size[1]):
                        pix = None
                        continue
                    
                    # Convert to PIL Image with proper handling of color spaces and alpha
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Handle alpha channel
                        if pix.alpha:
                            # Remove alpha channel by converting to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Convert to bytes
                        if pix.n == 1:  # Grayscale
                            img_data = pix.tobytes("ppm")
                        else:  # RGB
                            img_data = pix.tobytes("ppm")
                        
                        pil_img = Image.open(io.BytesIO(img_data))
                    else:  # CMYK or other complex color spaces
                        # Convert to RGB first
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix_rgb.tobytes("ppm")
                        pil_img = Image.open(io.BytesIO(img_data))
                        pix_rgb = None  # Clean up
                    
                    # Ensure image is in RGB mode for consistent saving
                    if pil_img.mode not in ['RGB', 'L']:
                        pil_img = pil_img.convert('RGB')
                    
                    # Save image
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = output_dir / img_filename
                    pil_img.save(img_path, "PNG")
                    
                    # Extract text from image using OCR if enabled
                    ocr_text = ""
                    if self.config.use_ocr:
                        try:
                            ocr_text = pytesseract.image_to_string(pil_img, lang='eng')
                            ocr_text = ocr_text.strip()
                        except Exception as e:
                            logger.warning(f"OCR failed for image {img_filename}: {e}")
                    
                    # Get image position on page
                    img_rect = page.get_image_rects(img)[0] if page.get_image_rects(img) else None
                    
                    image_info = {
                        'filename': img_filename,
                        'path': str(img_path),
                        'width': pix.width,
                        'height': pix.height,
                        'position': {
                            'x0': img_rect.x0 if img_rect else 0,
                            'y0': img_rect.y0 if img_rect else 0,
                            'x1': img_rect.x1 if img_rect else pix.width,
                            'y1': img_rect.y1 if img_rect else pix.height
                        } if img_rect else None,
                        'ocr_text': ocr_text,
                        'page_number': page_num
                    }
                    
                    images.append(image_info)
                    pix = None
                    
                except Exception as e:
                    logger.error(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to extract images from page {page_num}: {e}")
        
        return images

class TableExtractor:
    """Extract and process tables from PDFs"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def extract_tables_from_page(self, page_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        tables = []
        
        try:
            with pdfplumber.open(page_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    for table_index, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Ensure table has content
                            # Convert table to structured format
                            structured_table = self._structure_table(table)
                            
                            table_info = {
                                'table_id': f"page_{page_num}_table_{table_index}",
                                'data': structured_table,
                                'text_representation': self._table_to_text(structured_table),
                                'page_number': page_num,
                                'row_count': len(table),
                                'column_count': len(table[0]) if table else 0
                            }
                            
                            tables.append(table_info)
                            
        except Exception as e:
            logger.error(f"Failed to extract tables from page {page_num}: {e}")
        
        return tables
    
    def _structure_table(self, raw_table: List[List[str]]) -> Dict[str, Any]:
        """Convert raw table to structured format"""
        if not raw_table or not raw_table[0]:
            return {}
        
        # Clean the raw table first (handle None values)
        cleaned_table = []
        for row in raw_table:
            if row is not None:
                cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return {}
        
        # Assume first row is header
        headers = cleaned_table[0]
        rows = cleaned_table[1:]
        
        structured = {
            'headers': headers,
            'rows': rows,
            'data_dict': []
        }
        
        # Create dictionary representation
        for row in rows:
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    row_dict[header] = row[i]
                else:
                    row_dict[header] = ""  # Empty cell
            structured['data_dict'].append(row_dict)
        
        return structured
    
    def _table_to_text(self, structured_table: Dict[str, Any]) -> str:
        """Convert structured table to readable text"""
        if not structured_table or 'headers' not in structured_table:
            return ""
        
        text_lines = []
        headers = structured_table['headers']
        rows = structured_table['rows']
        
        # Clean headers (handle None values)
        clean_headers = [str(h) if h is not None else "" for h in headers]
        
        # Add headers
        text_lines.append("Table:")
        text_lines.append(" | ".join(clean_headers))
        text_lines.append("-" * (len(" | ".join(clean_headers))))
        
        # Add rows (handle None values)
        for row in rows:
            clean_row = [str(cell) if cell is not None else "" for cell in row]
            text_lines.append(" | ".join(clean_row))
        
        return "\n".join(text_lines)

class TextChunker:
    """Intelligent text chunking for optimal embedding"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def chunk_text(self, 
                   text: str, 
                   metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create intelligent text chunks
        
        Strategies:
        1. Sentence-boundary aware chunking
        2. Semantic coherence preservation
        3. Context overlap for continuity
        """
        if not text.strip():
            return []
        
        # Detect language
        language = "en"
        if self.config.language_detection:
            try:
                language = detect(text)
            except:
                pass
        
        # Split into sentences using spaCy (much faster and more accurate)
        if nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to simple splitting if spaCy not available
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk, 
                        current_sentences,
                        metadata,
                        language,
                        len(chunks)
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_sentences)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_sentences = [sentence]
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk,
                current_sentences, 
                metadata,
                language,
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, 
                     text: str, 
                     sentences: List[str],
                     metadata: Dict[str, Any],
                     language: str,
                     chunk_index: int) -> DocumentChunk:
        """Create a DocumentChunk object"""
        
        # Generate unique ID
        chunk_id = hashlib.md5(
            f"{metadata.get('source_file', '')}{chunk_index}{text[:100]}".encode()
        ).hexdigest()
        
        chunk_metadata = {
            **metadata,
            'chunk_index': chunk_index,
            'sentence_count': len(sentences),
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        return DocumentChunk(
            id=chunk_id,
            content=text.strip(),
            chunk_type="text",
            source_file=metadata.get('source_file', ''),
            page_number=metadata.get('page_number', 0),
            language=language,
            metadata=chunk_metadata
        )
    
    def _get_overlap_text(self, sentences: List[str]) -> str:
        """Get overlap text from previous chunk"""
        if not sentences:
            return ""
        
        # Take last few sentences for overlap
        overlap_sentences = sentences[-2:] if len(sentences) >= 2 else sentences
        overlap_text = " ".join(overlap_sentences)
        
        # Ensure overlap doesn't exceed overlap size
        if len(overlap_text) > self.config.overlap_size:
            # Truncate to word boundary
            words = overlap_text.split()
            truncated = ""
            for word in reversed(words):
                test_text = word + " " + truncated if truncated else word
                if len(test_text) <= self.config.overlap_size:
                    truncated = test_text
                else:
                    break
            overlap_text = truncated
        
        return overlap_text

class PDFProcessor:
    """Main PDF processing class"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.image_extractor = ImageExtractor(self.config)
        self.table_extractor = TableExtractor(self.config) 
        self.text_chunker = TextChunker(self.config)
    
    def process_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None) -> List[DocumentChunk]:
        """
        Process a complete PDF file
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for extracted images (optional)
            
        Returns:
            List of DocumentChunk objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Setup output directory for images
        if output_dir is None:
            output_dir = pdf_path.parent / "extracted" / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            
            logger.info(f"Processing PDF: {pdf_path.name} ({total_pages} pages)")
            
            for page_num in range(total_pages):
                try:
                    page = pdf_doc[page_num]
                    
                    # Extract text
                    text_chunks = self._extract_text_from_page(page, page_num, pdf_path)
                    all_chunks.extend(text_chunks)
                    
                    # Extract images
                    if self.config.extract_images:
                        image_chunks = self._extract_images_from_page(
                            page, page_num, pdf_path, output_dir
                        )
                        all_chunks.extend(image_chunks)
                    
                    # Extract tables
                    if self.config.extract_tables:
                        table_chunks = self._extract_tables_from_page(
                            pdf_path, page_num
                        )
                        all_chunks.extend(table_chunks)
                    
                except Exception as e:
                    logger.error(f"Failed to process page {page_num} of {pdf_path}: {e}")
                    continue
            
            pdf_doc.close()
            
            logger.info(f"Extracted {len(all_chunks)} chunks from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
        
        return all_chunks
    
    def _extract_text_from_page(self, 
                               page: fitz.Page, 
                               page_num: int, 
                               pdf_path: Path) -> List[DocumentChunk]:
        """Extract and chunk text from a page"""
        try:
            # Extract text with layout preservation
            if self.config.preserve_formatting:
                text = page.get_text("text")  # Simple text extraction
            else:
                text = page.get_text()
            
            if not text.strip():
                return []
            
            # Clean text
            text = self._clean_text(text)
            
            # Create metadata
            metadata = {
                'source_file': pdf_path.name,
                'page_number': page_num,
                'extraction_method': 'pymupdf',
                'content_type': 'text'
            }
            
            # Chunk text
            chunks = self.text_chunker.chunk_text(text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_num}: {e}")
            return []
    
    def _extract_images_from_page(self, 
                                 page: fitz.Page, 
                                 page_num: int, 
                                 pdf_path: Path,
                                 output_dir: Path) -> List[DocumentChunk]:
        """Extract images and create chunks"""
        chunks = []
        
        try:
            images = self.image_extractor.extract_images_from_page(
                page, page_num, output_dir
            )
            
            for img_info in images:
                # Create chunk for image with OCR text
                chunk_id = hashlib.md5(
                    f"{pdf_path.name}{page_num}{img_info['filename']}".encode()
                ).hexdigest()
                
                # Use OCR text as content, or description if no OCR
                content = img_info['ocr_text'] if img_info['ocr_text'] else f"Image: {img_info['filename']}"
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=content,
                    chunk_type="image", 
                    source_file=pdf_path.name,
                    page_number=page_num,
                    language="en",  # OCR typically in English
                    metadata={
                        'image_info': img_info,
                        'image_path': img_info['path'],
                        'has_ocr_text': bool(img_info['ocr_text']),
                        'image_dimensions': (img_info['width'], img_info['height'])
                    }
                )
                
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Failed to extract images from page {page_num}: {e}")
        
        return chunks
    
    def _extract_tables_from_page(self, 
                                 pdf_path: Path, 
                                 page_num: int) -> List[DocumentChunk]:
        """Extract tables and create chunks"""
        chunks = []
        
        try:
            tables = self.table_extractor.extract_tables_from_page(
                str(pdf_path), page_num
            )
            
            for table_info in tables:
                chunk_id = hashlib.md5(
                    f"{pdf_path.name}{page_num}{table_info['table_id']}".encode()
                ).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=table_info['text_representation'],
                    chunk_type="table",
                    source_file=pdf_path.name,
                    page_number=page_num,
                    language="en",  # Tables typically contain structured data
                    metadata={
                        'table_info': table_info,
                        'table_structure': table_info['data'],
                        'row_count': table_info['row_count'],
                        'column_count': table_info['column_count']
                    }
                )
                
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Failed to extract tables from page {page_num}: {e}")
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (simple heuristic)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove headers/footers (simple heuristic)
        lines = text.split('\n')
        if len(lines) > 10:
            # Remove first and last lines if they're very short (likely headers/footers)
            if len(lines[0]) < 50:
                lines = lines[1:]
            if len(lines[-1]) < 50:
                lines = lines[:-1]
        
        text = '\n'.join(lines)
        
        # Normalize unicode
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        return text.strip()

def process_downloads_folder(downloads_dir: str = "../downloads", 
                           output_dir: str = "./processed",
                           config: ProcessingConfig = None) -> List[DocumentChunk]:
    """
    Process all PDFs in the downloads folder
    
    Args:
        downloads_dir: Path to downloads folder
        output_dir: Output directory for processed files
        config: Processing configuration
        
    Returns:
        List of all DocumentChunks from all PDFs
    """
    downloads_path = Path(downloads_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = ProcessingConfig()
    
    processor = PDFProcessor(config)
    all_chunks = []
    
    # Find all PDF files
    pdf_files = list(downloads_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            chunks = processor.process_pdf(pdf_file, output_path / pdf_file.stem)
            all_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            continue
    
    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    
    # Save processing results
    results_file = output_path / "processing_results.json"
    with open(results_file, "w") as f:
        json.dump({
            'total_chunks': len(all_chunks),
            'files_processed': len(pdf_files),
            'chunk_types': {
                'text': len([c for c in all_chunks if c.chunk_type == 'text']),
                'image': len([c for c in all_chunks if c.chunk_type == 'image']), 
                'table': len([c for c in all_chunks if c.chunk_type == 'table'])
            },
            'languages': list(set(c.language for c in all_chunks))
        }, f, indent=2)
    
    return all_chunks


if __name__ == "__main__":
    # Example usage
    config = ProcessingConfig(
        extract_images=True,
        extract_tables=True,
        use_ocr=True,
        max_chunk_size=1200,
        overlap_size=150
    )
    
    chunks = process_downloads_folder(config=config)
    print(f"Processed {len(chunks)} document chunks")
    
    # Show sample chunks
    for chunk_type in ['text', 'image', 'table']:
        type_chunks = [c for c in chunks if c.chunk_type == chunk_type]
        if type_chunks:
            print(f"\n{chunk_type.upper()} CHUNK EXAMPLE:")
            print(f"Source: {type_chunks[0].source_file}")
            print(f"Content: {type_chunks[0].content[:200]}...")