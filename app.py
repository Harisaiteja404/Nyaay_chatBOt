from flask import Flask, request, jsonify
from Model.faiss_index import retrieve_documents, initialize_index, index_document
from Model.utils import extract_text_from_pdf
from Model.phi2_model import generate_answer, clear_conversation
import os
import logging
from werkzeug.exceptions import BadRequest, InternalServerError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data:
            raise BadRequest('No JSON data provided')
            
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        retrieved_chunks = retrieve_documents(question)
        context = '\n\n'.join(retrieved_chunks)
        answer = generate_answer(question, context)
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    try:
        clear_conversation()
        return jsonify({'status': 'conversation cleared'})
    except Exception as e:
        logger.error(f"Error in clear endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/init', methods=['POST'])
def init():
    try:
        if not os.path.exists('./documents'):
            return jsonify({'error': 'Documents directory not found'}), 404
            
        initialize_index()
        processed_files = 0
        failed_files = 0
        
        for filename in os.listdir('./documents'):
            if filename.endswith('.pdf'):
                try:
                    path = os.path.join('./documents', filename)
                    chunks = extract_text_from_pdf(path)
                    if chunks:
                        index_document(chunks)
                        processed_files += 1
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}")
                    failed_files += 1
                    
        return jsonify({
            'status': 'index initialized',
            'processed_files': processed_files,
            'failed_files': failed_files
        })
    except Exception as e:
        logger.error(f"Error in init endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Consider using environment variables for host and port
    app.run(host='0.0.0.0', port=8080)
