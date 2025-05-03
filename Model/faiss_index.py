import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import re

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Or use any other embedding model of choice

# Store the actual documents with metadata
documents = []

# FAISS index
embedding_dimension = 384  # all-MiniLM-L6-v2 has 384 dimensions
faiss_index = None

def initialize_index():
    """Initialize or reset the FAISS index"""
    global faiss_index, documents
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    documents = []
    print("FAISS index initialized")

def save_index_state():
    """Save the current state of the index and documents"""
    if not os.path.exists('./index_state'):
        os.makedirs('./index_state')
    
    # Save documents
    with open('./index_state/documents.json', 'w') as f:
        json.dump(documents, f)
    
    # Save FAISS index
    if faiss_index is not None:
        faiss.write_index(faiss_index, './index_state/faiss_index.bin')
        print("Index state saved")

def load_index_state():
    """Load the saved state of the index and documents"""
    global faiss_index, documents
    
    try:
        # Load documents
        if os.path.exists('./index_state/documents.json'):
            with open('./index_state/documents.json', 'r') as f:
                documents = json.load(f)
        
        # Load FAISS index
        if os.path.exists('./index_state/faiss_index.bin'):
            faiss_index = faiss.read_index('./index_state/faiss_index.bin')
            print("Index state loaded")
            return True
    except Exception as e:
        print(f"Error loading index state: {str(e)}")
    
    return False

# Function to index documents
def index_document(chunks):
    global documents, faiss_index
    
    if not chunks:
        print("Warning: No chunks to index")
        return
    
    # Initialize index if not already done
    if faiss_index is None:
        initialize_index()
    
    # Prepare embeddings for all chunks
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts)
    
    # Verify the embedding dimension
    if embeddings.shape[1] != embedding_dimension:
        raise ValueError(f"Expected embedding dimension {embedding_dimension}, got {embeddings.shape[1]}")
    
    # Add to FAISS index
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    
    # Store the document chunks with metadata
    documents.extend(chunks)
    print(f"Indexed {len(chunks)} chunks successfully")
    
    # Save the updated state
    save_index_state()

def is_legal_question(query):
    """Determine if a question is legal-related and where to search."""
    query_lower = query.lower()

    constitution_keywords = [
        'article', 'constitution', 'fundamental right', 'directive principle', 
        'preamble', 'citizenship', 'supreme court', 'parliament', 'rights', 'duties'
    ]
    ipc_keywords = [
        'section', 'ipc', 'penal code', 'crime', 'punishment', 
        'murder', 'theft', 'assault', 'kidnapping', 'offence', 'imprisonment', 'criminal'
    ]

    matched_constitution = any(term in query_lower for term in constitution_keywords)
    matched_ipc = any(term in query_lower for term in ipc_keywords)

    if matched_constitution and not matched_ipc:
        return True, 'constitution'
    elif matched_ipc and not matched_constitution:
        return True, 'ipc'
    elif matched_constitution and matched_ipc:
        # If both detected, prioritize based on keywords
        if 'article' in query_lower or 'constitution' in query_lower:
            return True, 'constitution'
        else:
            return True, 'ipc'
    else:
        return False, None


# Function to retrieve relevant documents
def retrieve_documents(query, k=5):
    if faiss_index is None or not documents:
        print("Warning: No documents have been indexed")
        return ["No documents available for retrieval"]

    try:
        is_legal, doc_source = is_legal_question(query)
        if not is_legal:
            return ["I apologize, but I can only assist with legal-related questions."]

        print(f"\nDocument Source: {doc_source.upper()}")

        # Try exact match
        section_match = re.search(r'(Section|Article)\s+(\d+[A-Za-z]*)', query, re.IGNORECASE)
        if section_match:
            section_type = section_match.group(1)
            section_num = section_match.group(2)
            search_key = f"{section_type.capitalize()} {section_num}"

            matched_docs = [doc for doc in documents if doc.get('section', '').lower() == search_key.lower()]
            if matched_docs:
                print(f"Found {len(matched_docs)} exact matches for {search_key}")
                # ⬇️ Combine all related chunks
                combined_text = "\n\n".join([format_chunk(doc) for doc in matched_docs])
                return [combined_text]

            print(f"No exact match found for {search_key}, using semantic search...")

        # Semantic search fallback
        print(f"Using semantic search for {doc_source}")
        query_embedding = model.encode([query])

        k = min(k, len(documents))
        distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)

        retrieved_docs = []
        for idx in indices[0]:
            if 0 <= idx < len(documents):
                chunk = documents[idx]
                if (doc_source == 'ipc' and not chunk.get('is_ipc', False)) or \
                   (doc_source == 'constitution' and not chunk.get('is_constitution', False)):
                    continue
                retrieved_docs.append(format_chunk(chunk))

        if retrieved_docs:
            # ⬇️ Combine top k chunks
            combined_text = "\n\n".join(retrieved_docs)
            return [combined_text]
        else:
            return ["No relevant sections found. Please try rephrasing your question."]
    except Exception as e:
        print(f"Error in retrieve_documents: {str(e)}")
        return ["An error occurred while retrieving documents. Please try again."]


def format_chunk(doc):
    text = doc.get('text', '')
    section = doc.get('section', 'Unknown')
    source = doc.get('source', '')
    citations = doc.get('citations', 'No specific citations')

    prefix = ""
    if doc.get('is_ipc', False):
        prefix = f"Section {section} of the Indian Penal Code:\n\n"
    elif doc.get('is_constitution', False):
        prefix = f"Article {section} of the Indian Constitution:\n\n"
    else:
        prefix = f"Section {section} from {source}:\n\n"

    return prefix + text + (f"\n\nRelated Citations: {citations}" if citations != "No specific citations" else '')

# Initialize the index on module import
if not load_index_state():
    initialize_index()
