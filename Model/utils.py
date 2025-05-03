import requests
import fitz  # PyMuPDF
import os
import json
import re

# Function to download PDF from Google Drive
def download_pdf_from_drive(url, output_path):
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return True
    
    try:
        # Extract the file ID from the Google Drive URL
        file_id = url.split('/d/')[1].split('/')[0]
        
        # First, get the file metadata to get the original filename
        metadata_url = f"https://drive.google.com/file/d/{file_id}/view"
        session = requests.Session()
        response = session.get(metadata_url)
        
        # Extract the title from the page content
        title_start = response.text.find('<title>') + 7
        title_end = response.text.find('</title>')
        if title_start > 7 and title_end > title_start:
            original_filename = response.text[title_start:title_end].strip()
            # Clean the filename and ensure it's a PDF
            original_filename = original_filename.replace(' - Google Drive', '').strip()
            if not original_filename.lower().endswith('.pdf'):
                original_filename += '.pdf'
        else:
            original_filename = f"document_{file_id}.pdf"
        
        # Create the direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Get the initial response
        response = session.get(download_url, stream=True)
        
        # Handle large file warning
        if 'confirm=' in response.url:
            confirm_token = response.url.split('confirm=')[1].split('&')[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
            response = session.get(download_url, stream=True)
        
        # Save the file with original filename
        final_path = os.path.join(os.path.dirname(output_path), original_filename)
        with open(final_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Successfully downloaded: {original_filename}")
        return True
    except Exception as e:
        print(f"Error downloading {output_path}: {str(e)}")
        return False

def clean_text(text):
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep legal citations and important punctuation
    text = re.sub(r'[^\w\s.,;:()\[\]{}§\-\'"]', ' ', text)
    # Remove page numbers and headers/footers
    text = re.sub(r'Page\s+\d+', '', text)
    text = re.sub(r'THE INDIAN PENAL CODE', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CONSTITUTION OF INDIA', '', text, flags=re.IGNORECASE)
    # Remove line numbers if present
    text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk = ""
    current_section = None

    # Document Type Detection
    is_ipc = False
    is_constitution = False
    first_page_text = doc[0].get_text("text").lower()
    if "indian penal code" in first_page_text or "ipc" in first_page_text:
        is_ipc = True
        print(f"Identified {pdf_path} as Indian Penal Code document")
    elif "constitution of india" in first_page_text or "fundamental rights" in first_page_text:
        is_constitution = True
        print(f"Identified {pdf_path} as Constitution document")

    section_patterns = [
        r'^\s*Section\s+(\d+[A-Za-z]*)[\.\-:\s]',
        r'^\s*(\d+[A-Za-z]*)[\.\-:\s]',
        r'^\s*Article\s+(\d+[A-Za-z]*)[\.\-:\s]',
        r'^\s*Art\.?\s*(\d+[A-Za-z]*)[\.\-:\s]'
    ]

    for page in doc:
        lines = page.get_text("text").split('\n')

        for line in lines:
            line = clean_text(line)
            section_match = None
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_match = match.group(1)
                    break

            if section_match:
                # Save previous chunk
                if current_chunk and current_section:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': current_section,
                        'source': os.path.basename(pdf_path),
                        'citations': extract_citations(current_chunk),
                        'is_ipc': is_ipc,
                        'is_constitution': is_constitution
                    })
                # Start new chunk
                if is_constitution:
                    current_section = f"Article {section_match}"
                else:
                    current_section = f"Section {section_match}"
                current_chunk = line
            else:
                current_chunk += ' ' + line

    # Save last chunk
    if current_chunk and current_section:
        chunks.append({
            'text': current_chunk.strip(),
            'section': current_section,
            'source': os.path.basename(pdf_path),
            'citations': extract_citations(current_chunk),
            'is_ipc': is_ipc,
            'is_constitution': is_constitution
        })

    return chunks

def extract_citations(text):
    # Extract legal citations
    citations = re.findall(r'(?:Article|Section|Rule|Regulation|Act)\s+\d+[A-Za-z]*(?:\s*\([^)]+\))?', text)
    return ", ".join(citations) if citations else "No specific citations"
