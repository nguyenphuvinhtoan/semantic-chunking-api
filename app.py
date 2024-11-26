from flask import Flask, request, jsonify
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env.dev')

app = Flask(__name__)

@app.route('/chunk', methods=['POST'])
def chunk_transcript():
    try:
        # Check if file is present in request
        if 'document' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['document']

        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        allowed_extensions = {'txt', 'pdf'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'File type not supported. Please upload .txt or .pdf files'}), 400

        sentences_per_group = request.form.get('sentences_per_group', 2, type=int)
        
        # Create semantic chunker
        chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        
        # Read the file content based on file type
        if file_extension == 'txt':
            content = file.read().decode('utf-8')
            # Get chunks using the codelight version which is more efficient
            chunks = chunker.create_documents_codelight([content], sentences_per_group=sentences_per_group)    
        else:  # pdf
            import tempfile
            import os
            
            # Save the file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            
            try:
                loader = PyPDFLoader(temp_path)
                content = loader.load()
                chunks = chunker.create_documents_codelight([d.page_content for d in content])
            finally:
                # Clean up temporary file
                os.remove(temp_path)
                os.rmdir(temp_dir)
        
        total_chunks = len(chunks)
        
        # Format response
        response = {
            'chunks': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in chunks
            ],
            'num_chunks': total_chunks
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)