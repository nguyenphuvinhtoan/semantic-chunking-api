import json
import mimetypes
import os
import re
import tempfile
from functools import wraps

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from werkzeug.datastructures import FileStorage

# Load environment variables from .env file
load_dotenv('.env.dev')

app = Flask(__name__)

# Constants
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SENTENCES_PER_GROUP = 50
DEFAULT_PLATFORM = 'OpenAI'

# Utility functions and decorators
def validate_file_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        has_file = 'document' in request.files
        has_url = 'documentUrl' in request.form
        if not (has_file or has_url):
            return jsonify({'error': 'Either document file or documentUrl must be provided'}), 400
            
        if has_file:
            file = request.files['document']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            file_extension = file.filename.rsplit('.', 1)[1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                return jsonify({'error': 'File type not supported. Please upload .txt or .pdf files'}), 400
            
        return f(*args, **kwargs)
    return decorated_function

def get_llm(platform):
    if platform == 'OpenAI':
        return ChatOpenAI(model="gpt-3.5-turbo")
    elif platform == 'Ollama':
        return OllamaLLM(model="llama3", base_url="http://localhost:11434")
    raise ValueError('Unsupported platform. Use "OpenAI" or "Ollama"')

def get_embeddings(platform):
    if platform == 'OpenAI':
        return OpenAIEmbeddings(model="text-embedding-3-large")
    elif platform == 'Ollama':
        return OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")
    raise ValueError('Unsupported platform. Use "OpenAI" or "Ollama"')

def read_file_content(file):
    file_extension = file.filename.rsplit('.', 1)[-1].lower()
    if file_extension == 'txt':
        content = file.read().decode('utf-8')
        return [Document(page_content=content)]
    else:  # pdf
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        try:
            loader = PyPDFLoader(temp_path)
            return loader.load()
        finally:
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
def download_and_save_file(url, save_path=None):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get filename from URL or Content-Disposition header
    filename = None
    if "Content-Disposition" in response.headers:
        content_disp = response.headers["Content-Disposition"]
        matches = re.findall("filename=(.+)", content_disp)
        if matches:
            filename = matches[0].strip('"')
    
    if not filename:
        filename = url.split('/')[-1]

        if not filename:
            # Use mimetypes to determine extension from content-type
            content_type = response.headers.get('content-type', '').split(';')[0]
            ext = mimetypes.guess_extension(content_type) or ''
            if not ext and 'pdf' in content_type.lower():
                ext = '.pdf'
            elif not ext and 'text' in content_type.lower():
                ext = '.txt'
            filename = f"downloaded_file{ext}"
            
    if '.pdf' not in filename or '.txt' not in filename:
        # Use mimetypes to determine extension from content-type
        content_type = response.headers.get('content-type', '').split(';')[0]
        ext = mimetypes.guess_extension(content_type) or ''
        if not ext and 'pdf' in content_type.lower():
            ext = '.pdf'
        elif not ext and 'text' in content_type.lower():
            ext = '.txt'
        filename = f"{filename}{ext}"
    
    # Create 'downloaded' directory if it doesn't exist
    download_dir = os.path.join(os.path.dirname(__file__), 'downloaded')
    os.makedirs(download_dir, exist_ok=True)
    
    # If no save_path provided, save to downloaded directory
    if not save_path:
        save_path = os.path.join(download_dir, filename)
    
    # Save the file
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    return save_path

def process_input_document():
    """Helper function to process either uploaded file or URL and return docs"""
    temp_path = None
    try:
        if 'document' in request.files:
            file = request.files['document']
            return read_file_content(file), None
        
        file_url = request.form.get('documentUrl')
        if not file_url:
            raise ValueError('No file or valid URL provided')
            
        temp_path = download_and_save_file(file_url)
        with open(temp_path, 'rb') as f:
            file = FileStorage(f, filename=os.path.basename(temp_path))
            return read_file_content(file), temp_path
            
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


# Error handling middleware
@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({'error': str(error)}), 500

# Routes
@app.route('/chunk', methods=['POST'])
@validate_file_request
def chunk_transcript():
    temp_path = None
    try:
        docs, temp_path = process_input_document()
        sentences_per_group = request.form.get('sentences_per_group', DEFAULT_SENTENCES_PER_GROUP, type=int)
        platform = request.form.get('platform', DEFAULT_PLATFORM, type=str)
        embeddings = get_embeddings(platform)
        chunker = SemanticChunker(embeddings=embeddings)
        
        chunks = chunker.create_documents_codelight([d.page_content for d in docs], sentences_per_group=sentences_per_group)
        
        return jsonify({
            'chunks': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in chunks],
            'num_chunks': len(chunks)
        })  
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/summarize', methods=['POST'])
@validate_file_request
def summarize_document():
    temp_path = None  
    try: 
        docs, temp_path = process_input_document()
        
        method = request.form.get('method', 'stuff')
        chunk_size = request.form.get('chunk_size', DEFAULT_CHUNK_SIZE, type=int)
        platform = request.form.get('platform', DEFAULT_PLATFORM, type=str)
        
        llm = get_llm(platform)
        
        if method == 'stuff':
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that creates concise summaries.
                Write a clear and informative summary of the following text:
                
                {context}
                
                Focus on the main points and key information.""")
            ])
            chain = create_stuff_documents_chain(llm, prompt)
            summary = chain.invoke({"context": docs})
        else:
            # Split documents into chunks
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, 
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)
            
            # Map: Summarize each chunk
            map_prompt = ChatPromptTemplate.from_messages([
                ("system", """Write a concise summary of the following text segment:
                
                {context}
                
                Focus on key points and main ideas.""")
            ])
            map_chain = create_stuff_documents_chain(llm, map_prompt)
            
            # Create summaries for each chunk
            summaries = []
            for doc in split_docs:
                summary = map_chain.invoke({"context": [doc]})
                summaries.append(summary)
            
            # Reduce: Combine all summaries
            reduce_prompt = ChatPromptTemplate.from_messages([
                ("system", """Combine these summaries into a coherent final summary:
                
                {context}
                
                Create a well-organized summary that captures the main themes and key points.""")
            ])
            reduce_chain = create_stuff_documents_chain(llm, reduce_prompt)
            summary = reduce_chain.invoke({"context": [Document(page_content=s) for s in summaries]})

        return jsonify({
            'summary': summary,
            'method': method,
            'platform': platform,
            'num_docs': len(docs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/chapters', methods=['POST'])
@validate_file_request
def get_chapters():
    temp_path = None
    try:
        docs, temp_path = process_input_document()
            
        platform = request.form.get('platform', DEFAULT_PLATFORM, type=str)
        llm = get_llm(platform)
        
        full_content = "\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                # Analyze the provided text to identify its chapters or main sections. For each chapter or section, provide a detailed breakdown using the exact JSON format outlined below.
                    - Identify chapters or sections from the text.
                    - Assign a clear and concise title to each section.
                    - Provide the source or reference for the section's location, such as page number or specific location marker.
                    - Summarize the section content with a detailed description, consisting of at least two to three sentences that cover the main points and key concepts.
                    - Use sequential numbering for the sections, starting with position number 1.
                    - If no distinct chapters or sections are evident, infer logical divisions based on the text's structure.

                    Return only the JSON array, adhering strictly to the format specified.

                # Output Format:
                [
                    {{
                        "title": "Clear and concise title of the section",
                        "source": "Page number or location reference (e.g., page 1, section start, etc.)",
                        "content": "Detailed description of the chapter's content (2-3 sentences minimum, summarizing main points and key concepts)",
                        "position": "Sequential number indicating the chapter's order (starting from 1)"
                    }}
                ]

                # Notes:

                - Ensure the summaries are meaningful with comprehensive coverage of key content points.
                - Accurately locate sections using the appropriate source markers.
                - Maintain the JSON format without deviations; no additional commentary beyond the JSON array is needed.
                """),
            ("user", "Text to analyze: \n{context}")
            ])

        # Create and run the chain
        chain = create_stuff_documents_chain(llm, prompt)
        
        result = chain.invoke({"context": [Document(page_content=full_content)]})

        # Extract JSON array using regex
        match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)

        if not match:
            raise ValueError("No JSON array found in response.")
        
        json_array_str = match.group(0)

        chapters = json.loads(json_array_str)

        return jsonify({
            'chapters': chapters,
            'num_chapters': len(chapters)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/flashcards', methods=['POST'])
@validate_file_request
def generate_flashcards():
    temp_path = None
    try:
        docs, temp_path = process_input_document()
        platform = request.form.get('platform', DEFAULT_PLATFORM, type=str)
        
        llm = get_llm(platform)
        full_content = "\n".join([doc.page_content for doc in docs])
        
        # First, determine the suitable number of flashcards based on content
        sizing_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the text and determine a suitable number of flashcards to create.
            Consider:
            - The length and complexity of the content
            - The number of distinct concepts or key points
            - A reasonable learning load (typically 5-15 cards per study session)
            
            Return only a number."""),
            ("user", "Text to analyze: \n{context}")
        ])
        
        sizing_chain = create_stuff_documents_chain(llm, sizing_prompt)
        num_cards_result = sizing_chain.invoke({"context": [Document(page_content=full_content)]})
        # Extract the number from the response
        num_cards = min(max(int(re.search(r'\d+', num_cards_result).group()), 5), 15)
        
        # Generate flashcards
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful teaching assistant that creates educational flashcards.
            Generate appropriate flashcards from the following text. Each flashcard should:
            - Have a clear, focused question that tests understanding
            - Provide a concise, accurate answer
            - Include a brief explanation that helps understand the concept
            - Cover different aspects and difficulty levels
            
            Return the flashcards in this exact JSON format:
            [
                {{
                    "question": "The question to test knowledge",
                    "answer": "The correct answer",
                    "explanation": "A helpful explanation of why this is correct"
                }}
            ]
            
            Focus on key concepts, important details, and ensure comprehensive coverage of the material."""),
            ("user", "Text to analyze: \n{context}")
        ])
        
        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({
            "context": [Document(page_content=full_content)]
        })
        # Extract JSON array using regex
        match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response.")
        
        flashcards = json.loads(match.group(0))
        
        return jsonify({
            'flashcards': flashcards,
            'num_cards': len(flashcards)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(port=8000, debug=True)