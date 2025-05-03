import os
import shutil
from Model.utils import extract_text_from_pdf
from Model.faiss_index import initialize_index, index_document

DOCUMENTS_PATH = './documents/'
INDEX_STATE_PATH = './index_state/'

def clear_index_state():
    """Clear the existing FAISS index and document metadata."""
    if os.path.exists(INDEX_STATE_PATH):
        shutil.rmtree(INDEX_STATE_PATH)
        print("Old index_state/ folder deleted.")
    else:
        print("No existing index_state/ found. Proceeding fresh.")

def rebuild_index():
    """Rebuild the FAISS index from all PDFs in documents/"""
    clear_index_state()
    initialize_index()

    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in documents/ folder.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_PATH, pdf_file)
        try:
            print(f"\nProcessing {pdf_file}...")
            chunks = extract_text_from_pdf(pdf_path)
            if chunks:
                print(f"Extracted {len(chunks)} chunks. Indexing...")
                index_document(chunks)
            else:
                print(f"Warning: No chunks extracted from {pdf_file}. Skipping.")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    print("\n✅ Reindexing completed successfully!")

if __name__ == "__main__":
    rebuild_index()
