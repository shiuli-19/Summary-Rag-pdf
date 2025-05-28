import os
import io
import re
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Union, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, render_template, request, jsonify, url_for, session
from werkzeug.utils import secure_filename
import uuid
import time

# Set up Flask app
app = Flask(__name__)
app.secret_key = "pdf_summarizer_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}  # You can add other formats if needed

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class EnhancedPDFSummarizer:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "google/gemma-2b-it",  
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the enhanced PDF summarizer with specified models.
        
        Args:
            embedding_model_name: Model name for text embeddings
            llm_model_name: Model name for the large language model
            device: Device to run models on (cuda or cpu)
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize embedding model for semantic search
        print(f"Loading embedding model {embedding_model_name}...")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
        
        # Initialize LLM for summarization and QA
        print(f"Loading LLM model {llm_model_name}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        
        # Use the text-generation pipeline
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.llm_tokenizer,
            device=0 if device == "cuda" else -1,  # Assign to GPU or CPU
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Image.Image]]:
        """
        Extract text from a PDF file. If text extraction fails, save pages as images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing extracted text and list of page images (if text extraction failed)
        """
        print(f"Processing PDF: {pdf_path}")
        
        extracted_text = ""
        images = []
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Try to extract text
                page_text = page.get_text()
                
                # If the page has very little text, it might be an image/scanned page
                if len(page_text.strip()) < 50:
                    print(f"Page {page_num+1} has little text, saving as image for OCR...")
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                else:
                    extracted_text += page_text + "\n\n"
            
            return extracted_text, images
        
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return "", []
    
    def perform_ocr(self, images: List[Image.Image]) -> str:
        """
        Perform OCR on a list of images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Extracted text from OCR
        """
        if not images:
            return ""
        
        print(f"Performing OCR on {len(images)} images...")
        ocr_text = ""
        
        for i, img in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(img)
                ocr_text += page_text + "\n\n"
                print(f"Completed OCR for image {i+1}/{len(images)}")
            except Exception as e:
                print(f"Error performing OCR on image {i+1}: {e}")
        
        return ocr_text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive newlines and whitespaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove header/footer patterns if detected
        # This would need customization based on your specific PDFs
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1024, chunk_overlap: int = 100) -> List[str]:
        """
        Split text into chunks of specified size.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        print(f"Chunking text with chunk size {chunk_size}, overlap {chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text chunks
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        # Process in batches to avoid OOM errors
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and get embeddings
            inputs = self.embedding_tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        return all_embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            FAISS index
        """
        vector_dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        index.add(embeddings)
        return index
    
    def similarity_search(
        self, 
        query: str, 
        index: faiss.IndexFlatL2, 
        chunks: List[str], 
        embeddings: np.ndarray, 
        k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform similarity search for query against indexed chunks.
        
        Args:
            query: Query text
            index: FAISS index
            chunks: Original text chunks
            embeddings: Array of embeddings
            k: Number of similar chunks to retrieve
            
        Returns:
            List of dictionaries with chunks and their similarity scores
        """
        # Get query embedding
        inputs = self.embedding_tokenizer(
            [query], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            query_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_embedding = query_embedding.cpu().numpy()
        
        # Search for similar chunks
        distances, indices = index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            results.append({
                "chunk": chunks[idx],
                "similarity": 1.0 / (1.0 + distance)  # Convert distance to similarity score
            })
        
        return results
    
    def generate_llm_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        result = self.llm_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]['generated_text']
        
        # Extract the generated response (remove the prompt if possible)
        if prompt in result:
            response = result.split(prompt, 1)[1].strip()
        else:
            # If exact matching doesn't work, use a more general approach
            # Find where the model's response actually starts
            response = result.strip()
        
        return response
    
    def answer_query(self, query: str, similar_chunks: List[Dict[str, Union[str, float]]]) -> str:
        """
        Generate a structured answer to a specific query based on similar chunks.
        
        Args:
            query: User query
            similar_chunks: List of relevant chunks with similarity scores
            
        Returns:
            Generated answer
        """
        # Combine relevant chunks into a context
        context = ""
        for item in similar_chunks:
            context += item["chunk"] + " "
        
        # Format as a question answering task with structured output instructions
        prompt = f"""
You are an expert document analyst. Based on the following information from a document, please provide a clear, well-structured answer to this question:

QUESTION: {query}

INFORMATION:
{context}

Format your answer as follows:
1. Start with a direct answer to the question
2. Include relevant key points in bullet form
3. Use headings where appropriate to organize different aspects
4. Include specific details from the document that support your answer

Remember to be concise and focus only on information that directly answers the question.
"""
        
        # Generate answer using the LLM
        answer = self.generate_llm_response(prompt, max_tokens=512)
        
        return answer
    
    def summarize_text(self, text: str, format_type: str = "structured") -> str:
        """
        Generate a well-structured summary of the provided text.
        
        Args:
            text: Text to summarize
            format_type: Type of formatting ("structured", "bullet_points", "executive")
            
        Returns:
            Generated structured summary
        """
        # Check if the text is too long for processing at once
        if len(text.split()) > 4000:
            print("Text is too long for direct summarization, breaking into parts...")
            chunks = self.chunk_text(text, chunk_size=4000, chunk_overlap=200)
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                chunk_summary = self._generate_chunk_summary(chunk, format_type)
                chunk_summaries.append(chunk_summary)
            
            combined_text = "\n\n".join(chunk_summaries)
            
            # Combine the summaries of the chunks
            prompt = f"""
You are an expert document analyst. I have split a long document into sections and summarized each section. Please create a cohesive {format_type} summary that combines these section summaries into a unified document:

SECTION SUMMARIES:
{combined_text}

Create a {format_type} summary with:
1. An informative title reflecting the main topic
2. An executive summary (2-3 sentences)
3. Main sections with clear headings
4. Bullet points for key information
5. A brief conclusion
"""
            final_summary = self.generate_llm_response(prompt, max_tokens=1024)
            return final_summary
        
        # For text that can be processed at once
        return self._generate_chunk_summary(text, format_type)
    
    def _generate_chunk_summary(self, text: str, format_type: str) -> str:
        """
        Generate a summary for a chunk of text.
        
        Args:
            text: Text chunk to summarize
            format_type: Type of formatting
            
        Returns:
            Formatted summary
        """
        if format_type == "structured":
            prompt = f"""
You are an expert document analyst. Create a comprehensive, well-structured summary of the following text:

TEXT:
{text}

Format your summary as follows:
1. Start with an informative title
2. Include an executive summary (2-3 sentences)
3. Organize main content into clear sections with headings
4. Use bullet points for key information under each section
5. Include a brief conclusion

Focus on the most important information and main points.
"""
        elif format_type == "bullet_points":
            prompt = f"""
You are an expert document analyst. Create a bullet-point summary of the following text:

TEXT:
{text}

Format your summary as follows:
1. Start with an informative title
2. Include 5-7 main bullet points covering the core information
3. Use sub-bullets for supporting details where needed
4. Group related points together in clear sections
5. Keep bullet points concise and informative

Focus on the most important information.
"""
        elif format_type == "executive":
            prompt = f"""
You are an expert document analyst. Create a concise executive summary of the following text:

TEXT:
{text}

Format your summary as follows:
1. Start with a clear title
2. Begin with a 1-2 sentence overview 
3. List 3-5 key takeaways as bullet points
4. Include any critical data points or findings
5. End with a brief statement of implications or next steps

Keep the summary focused, actionable, and under 300 words.
"""
        else:  # Default to structured
            prompt = f"""
You are an expert document analyst. Create a well-organized summary of the following text:

TEXT:
{text}

Format your summary to include:
1. An informative title
2. A brief overview
3. Key points organized in a clear structure
4. Important details from the text
5. A brief conclusion

Focus on providing a clear and comprehensive understanding of the main content.
"""
        
        # Generate the summary
        summary = self.generate_llm_response(prompt, max_tokens=1024)
        return summary
    
    def process_pdf(
        self, 
        pdf_path: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        format_type: str = "structured"
    ) -> Dict:
        """
        Process a PDF file for summarization and prepare for query answering.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            format_type: Type of summary formatting
            
        Returns:
            Dictionary with processing results and data structures for querying
        """
        # Extract text from PDF
        extracted_text, images = self.extract_text_from_pdf(pdf_path)
        
        # If text extraction failed or insufficient text was found, perform OCR
        if len(extracted_text.strip()) < 100 and images:
            ocr_text = self.perform_ocr(images)
            extracted_text += ocr_text
        
        # Clean the text
        cleaned_text = self.clean_text(extracted_text)
        
        if not cleaned_text:
            return {"error": "No text could be extracted from the PDF"}
        
        # Chunk the text
        chunks = self.chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Get embeddings
        embeddings = self.get_embeddings(chunks)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Generate a summary
        summary_text = self.summarize_text(cleaned_text, format_type=format_type)
        
        return {
            "success": True,
            "summary": summary_text,
            "chunks": chunks,
            "embeddings": embeddings,
            "index": index,
            "text": cleaned_text
        }

# Initialize the enhanced summarizer
summarizer = EnhancedPDFSummarizer()

# Dictionary to store processed documents
processed_docs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400
    
    # Generate a unique ID for this upload
    doc_id = str(uuid.uuid4())
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{doc_id}_{filename}")
    file.save(file_path)
    
    # Process the PDF (in a production environment, consider running this in a background task)
    try:
        # Get chunk size and format type from form or use defaults
        chunk_size = int(request.form.get('chunk_size', 1024))
        format_type = request.form.get('format_type', 'structured')
        
        result = summarizer.process_pdf(
            file_path, 
            chunk_size=chunk_size,
            format_type=format_type
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Store the processed document data
        processed_docs[doc_id] = {
            'chunks': result['chunks'],
            'embeddings': result['embeddings'],
            'index': result['index'],
            'summary': result['summary'],
            'filename': filename,
            'text': result['text']
        }
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'summary': result['summary'],
            'filename': filename,
            'processing_time': f"{processing_time:.2f} seconds"
        })
    
    except Exception as e:
        return jsonify({'error': f"Error processing PDF: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query_document():
    # Get the query and document ID
    query = request.form.get('query')
    doc_id = request.form.get('doc_id')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    if not doc_id or doc_id not in processed_docs:
        return jsonify({'error': 'Document not found or not processed yet'}), 404
    
    try:
        # Get document data
        doc_data = processed_docs[doc_id]
        
        # Perform similarity search
        similar_chunks = summarizer.similarity_search(
            query, 
            doc_data['index'], 
            doc_data['chunks'], 
            doc_data['embeddings']
        )
        
        # Generate answer
        answer = summarizer.answer_query(query, similar_chunks)
        
        # Return the answer and relevant chunks
        return jsonify({
            'success': True,
            'answer': answer,
            'relevant_chunks': [chunk['chunk'] for chunk in similar_chunks[:3]]
        })
    
    except Exception as e:
        return jsonify({'error': f"Error answering query: {str(e)}"}), 500

@app.route('/get_summary/<doc_id>')
def get_summary(doc_id):
    if doc_id not in processed_docs:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify({
        'success': True,
        'summary': processed_docs[doc_id]['summary'],
        'filename': processed_docs[doc_id]['filename']
    })

@app.route('/reformat_summary', methods=['POST'])
def reformat_summary():
    doc_id = request.form.get('doc_id')
    format_type = request.form.get('format_type', 'structured')
    
    if not doc_id or doc_id not in processed_docs:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        # Get the full text and generate a new summary with the specified format
        doc_text = processed_docs[doc_id]['text']
        new_summary = summarizer.summarize_text(doc_text, format_type=format_type)
        
        # Update the stored summary
        processed_docs[doc_id]['summary'] = new_summary
        
        return jsonify({
            'success': True,
            'summary': new_summary
        })
    
    except Exception as e:
        return jsonify({'error': f"Error reformatting summary: {str(e)}"}), 500

# Clear a document from memory
@app.route('/clear/<doc_id>', methods=['POST'])
def clear_document(doc_id):
    if doc_id in processed_docs:
        # Remove the file
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(f"{doc_id}_"):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Remove from memory
        del processed_docs[doc_id]
        
        return jsonify({'success': True, 'message': 'Document cleared successfully'})
    
    return jsonify({'error': 'Document not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)