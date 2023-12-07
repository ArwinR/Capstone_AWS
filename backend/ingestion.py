
# Enhanced Ingestion Script

# Importing required libraries
import os
import logging
import sys
import pdfplumber  # Improved PDF text extraction
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
load_dotenv()  # Load environment variables from .env file
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
mistral_repo = 'mistralai/Mistral-7B-Instruct-v0.1'

# Tokenizer
embedd_model = 'BAAI/bge-reranker-large'
model_kwargs = {"device": 'cpu'}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

def ingest_doc(doc_path, file_name):
    # Create output directory for vector databases
    outdir = "./backend/vector_databases/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Creating database path
    db_path = os.path.join(outdir, file_name)
    logging.info('Database Path: %s', db_path)

    # Check if the database already exists
    if not os.path.exists(db_path):
        try:
            # Enhanced PDF text extraction using pdfplumber
            with pdfplumber.open(doc_path) as pdf:
                raw_doc = ""
                for page in pdf.pages:
                    raw_doc += page.extract_text() + "\n\n"  # Extract text from each page

            # Split and store vectors
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=20, separators=["\n\n", "\n", " ", ""]
            )
            all_splits = text_splitter.split_documents(raw_doc)

            # Creating vector store
            vectorstore = Chroma.from_documents(
                documents=all_splits, embedding=embeddings, persist_directory=db_path
            )
        except Exception as e:
            logging.error('Error occurred during PDF processing: %s', e)
            sys.exit(1)
    else:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    return vectorstore

# Example usage
# ingest_doc('path_to_pdf_document', 'example_vector_db')
