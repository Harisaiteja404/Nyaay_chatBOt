import os
from Model.utils import download_pdf_from_drive, extract_text_from_pdf
from Model.faiss_index import index_document, retrieve_documents, initialize_index
from Model.phi2_model import generate_answer

# Path to store downloaded PDFs
DOCUMENTS_PATH = './documents/'

def check_documents():
    """Check if documents exist and are properly indexed"""
    if not os.path.exists(DOCUMENTS_PATH):
        print("Documents directory does not exist")
        return False
    
    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in documents directory")
        return False
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"- {pdf}")
    return True

# Function to initialize the documents by downloading and indexing
def initialize_documents():
    # Example of downloading documents (use your Google Drive links)
    document_urls = [
        'https://drive.google.com/file/d/1vKxIUi_aYJAMxf5nwMlhmG_yvIb4GLPh/view?usp=drive_link', 
        'https://drive.google.com/file/d/1ulmaX0P8dE1W9INqYrl_KXUzjxFM3CWD/view?usp=drive_link'
    ]

    # Create documents directory if it doesn't exist
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        print(f"Created documents directory at {DOCUMENTS_PATH}")

    # Download and index documents
    for i, url in enumerate(document_urls):
        try:
            # Extract file ID for temporary path
            file_id = url.split('/d/')[1].split('/')[0]
            temp_path = os.path.join(DOCUMENTS_PATH, f'temp_{file_id}.pdf')
            
            print(f"\nProcessing document {i+1}...")
            print(f"URL: {url}")
            print(f"File ID: {file_id}")
            
            # Download will handle the original filename
            if not download_pdf_from_drive(url, temp_path):
                print(f"Failed to download document {i+1}")
                continue
            
            # Find the actual downloaded file.
            downloaded_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
            pdf_file = next((f for f in downloaded_files if f != f'temp_{file_id}.pdf'), None)
            
            if pdf_file:
                pdf_path = os.path.join(DOCUMENTS_PATH, pdf_file)
                print(f"Found downloaded file: {pdf_file}")
                print(f"Extracting and processing {pdf_file}...")
                
                chunks = extract_text_from_pdf(pdf_path)
                if chunks:
                    print(f"Extracted {len(chunks)} chunks from {pdf_file}")
                    print(f"Indexing chunks...")
                    index_document(chunks)
                    print(f"{pdf_file} processed and indexed successfully.")
                else:
                    print(f"Warning: No text could be extracted from {pdf_file}")
                
                # Remove temporary file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"Removed temporary file: {temp_path}")
            else:
                print(f"Could not find downloaded file for document {i+1}")
        except Exception as e:
            print(f"Error processing document {i+1}: {str(e)}")
            continue

# Main function to take user input and get the answer
def main():
    print("NYAAY Bot - Your Indian Legal Assistant")
    print("Type 'exit' to quit or 'clear' to start a new conversation")

    # Check if documents exist and are properly indexed
    if not check_documents():
        print("\nInitializing documents...")
        initialize_documents()
    else:
        print("\nProcessing documents...")
        # Reinitialize the index
        from Model.faiss_index import initialize_index
        initialize_index()
        
        # Process and index all documents
        pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(DOCUMENTS_PATH, pdf_file)
            try:
                chunks = extract_text_from_pdf(pdf_path)
                if chunks:
                    print(f"Indexing {pdf_file}...")
                    index_document(chunks)
                else:
                    print(f"Warning: Could not process {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}")
                continue

    print("\nReady to answer your questions!")
    while True:
        # Take input from user
        query = input("\nEnter your question: ")

        if query.lower() == 'exit' or query.lower() == 'quit':
            break
        elif query.lower() == 'clear':
            from Model.phi2_model import clear_conversation
            clear_conversation()
            print("Conversation history cleared. Starting fresh!")
            continue

        try:
            # Retrieve relevant documents with metadata
            retrieved_docs = retrieve_documents(query)

            # Combine the relevant documents with their metadata
            relevant_text = '\n\n'.join(retrieved_docs)

            # Generate the answer using Phi model
            answer = generate_answer(query, relevant_text)

            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"\nError processing your question. Please try again.")

if __name__ == '__main__':
    main()
