"""
Complete RAG (Retrieval-Augmented Generation) Pipeline

This module provides a comprehensive RAG pipeline that:
- Loads Amazon product data from CSV with product labels from header
- Creates structured documents with key-value pairs using product labels
- Stores vector embeddings in PostgreSQL with PGVector extension
- Retrieves context using semantic similarity
- Generates responses using local Ollama LLM (llama3.2:latest)
- Uses OpenAI's text-embedding-3-small for embeddings
"""

import os
import pandas as pd
from typing import Optional, List
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline for Product Question Answering
    
    This class handles:
    - Loading and preprocessing product data from CSV
    - Creating structured documents with product labels
    - Building and querying a PGVector-backed semantic search index
    - Generating responses using local Ollama LLM
    """
    os.environ["OPENAI_API_KEY"] = ""
    def __init__(
        self,
        csv_file_path: Optional[str] = None,
        pdf_file_path: Optional[str] = None,
        pgvector_connection_string: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "llama3.2:latest",
        collection_name: str = "documents",
        table_name: str = "documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 10,
    ):
        """
        Initialize the RAG Pipeline
        
        Args:
            csv_file_path: Path to the Amazon sales CSV file
            pgvector_connection_string: PostgreSQL connection string with PGVector
            ollama_base_url: Base URL for local Ollama instance
            embedding_model: OpenAI embedding model to use
            llm_model: Ollama LLM model to use
            collection_name: PGVector collection name
            table_name: PostgreSQL table name
            chunk_size: Character size for document chunks
            chunk_overlap: Character overlap between chunks
            top_k: Number of top results to retrieve for context
        """
        # Data paths
        self.csv_file_path = csv_file_path or os.path.join(
            Path(__file__).parent.parent,
            "data",
            "Amazon_Sales_6.csv"
        )
        self.pdf_file_path = ["data/Customer Support Escalation Policy.pdf",
                              "data/Fraud Prevention Policy.pdf",
                              "data/Return and Refund Policy.pdf",
                              "data/Shipping and Delivery Policy.pdf",
                              "data/App Privacy Policy.pdf"]       
        # Vector store configuration
        self.pgvector_connection_string = (
            pgvector_connection_string or
            os.getenv(
                "PGVECTOR_CONNECTION_STRING",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/shopassist_db"
            )
        )
        self.collection_name = collection_name
        self.table_name = table_name
        
        # LLM configuration
        self.ollama_base_url = (
            ollama_base_url or
            os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        )
        self.llm_model = llm_model
        
        # Embedding configuration
        self.embedding_model_name = embedding_model
        
        # Chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components
        self._embeddings = None
        self._llm = None
        self._vectorstore = None
        self._product_labels = None
        
        logger.info("RAG Pipeline initialized")
        logger.info(f"CSV Path: {self.csv_file_path}")
        logger.info(f"PGVector Connection: {self.pgvector_connection_string}")
        logger.info(f"Ollama Base URL: {self.ollama_base_url}")
    
    @property
    def embeddings(self):
        """Lazy load embeddings model"""
        if self._embeddings is None:
            logger.info(f"Loading embeddings model: {self.embedding_model_name}")
            self._embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        return self._embeddings
    
    @property
    def llm(self):
        """Lazy load LLM model"""
        if self._llm is None:
            logger.info(f"Loading LLM model: {self.llm_model} from {self.ollama_base_url}")
            try:
                self._llm = ChatOllama(
                    model=self.llm_model,
                    base_url=self.ollama_base_url,
                    temperature=0,
                )
                logger.info("LLM model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.ollama_base_url}. "
                    f"Ensure Ollama is running and {self.llm_model} is available."
                ) from e
        return self._llm
    
    @property
    def vectorstore(self):
        """Lazy load vectorstore"""
        if self._vectorstore is None:
            logger.info("Loading PGVector store from database")
            try:
                self._vectorstore = PGVector(
                    collection_name=self.collection_name,
                    connection_string=self.pgvector_connection_string,
                    embedding_function=self.embeddings,                                        
                )
                logger.info("PGVector store loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PGVector store: {e}")
                raise RuntimeError(
                    f"Could not connect to PGVector at {self.pgvector_connection_string}. "
                    f"Ensure PostgreSQL with pgvector is running."
                ) from e
        return self._vectorstore
    
    def load_csv_data(self) -> pd.DataFrame:
        """
        Load CSV file and return DataFrame
        
        Returns:
            pandas DataFrame with product data
        """
        logger.info(f"Loading CSV data from: {self.csv_file_path}")
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        df = pd.read_csv(self.csv_file_path)
        logger.info(f"Loaded {len(df)} product records with columns: {list(df.columns)}")
        return df
    
    def extract_product_labels(self, df: pd.DataFrame) -> List[str]:
        """
        Extract product labels from the first row (header) of the DataFrame
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of column names (product labels)
        """
        labels = df.columns.tolist()
        self._product_labels = labels
        logger.info(f"Extracted {len(labels)} product labels: {labels}")
        return labels
    
    def load_pdf_data(self) :
        """
        Load and extract text from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of text documents (one per page)
        """
        logger.info("Loading PDF documents...")      
        
        documents = []
        
        # Load all PDFs using PyMuPDFLoader
        for pdf_path in self.pdf_file_path:
            try:
                if os.path.exists(pdf_path):
                    logger.info(f"Loading PDF with PyMuPDFLoader: {pdf_path}")
                    loader = PyMuPDFLoader(pdf_path)                                   
                    # Load documents from this PDF
                    pdf_docs = loader.load()
                    documents.extend(pdf_docs)
                    logger.info(f"Loaded {len(pdf_docs)} documents from {pdf_path}")
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_path}: {e}")
                continue
        return documents
            
    def convert_rows_to_documents(self, df: pd.DataFrame, labels: List[str]) -> List[str]:
        """
        Convert each DataFrame row into a structured text document using product labels
        
        Each document contains key-value pairs with all product attributes.
        Format: {label1}: {value1}, {label2}: {value2}, ...
        
        Args:
            df: pandas DataFrame with product data
            labels: List of column names (product labels)
            
        Returns:
            List of formatted text documents
        """
        documents = []
        
        for idx, row in df.iterrows():
            # Create key-value pairs for each column using product labels
            doc_text = ", ".join(
                f"{label}: {str(row[label]).strip()}"
                for label in labels
                if pd.notna(row[label])
            )
            documents.append(doc_text)
        
        logger.info(f"Converted {len(documents)} rows into documents")
        return documents
    
    def chunk_documents(self, documents: List[str]) -> List[Document]:
        """
        Split documents into smaller chunks using RecursiveCharacterTextSplitter
        
        Args:
            documents: List of text documents
            
        Returns:
            List of LangChain Document objects (chunks)
        """
        logger.info(
            f"Chunking documents with size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[", ", " ", ""]
        )
        
        chunked_documents = splitter.create_documents(documents)
        logger.info(f"Created {len(chunked_documents)} document chunks")
        return chunked_documents
    
    def build_rag_pipeline(self) -> dict:
        """
        Build and populate the RAG pipeline
        
        This function:
        1. Loads CSV product data
        2. Extracts product labels from headers
        3. Converts rows to structured documents with key-value pairs
        4. Chunks documents for better retrieval
        5. Generates embeddings and stores in PGVector
        
        Returns:
            Dictionary with pipeline build statistics
        """
        logger.info("=" * 60)
        logger.info("Starting RAG Pipeline Build")
        logger.info("=" * 60)
        
        try:
            # Load CSV data
            df = self.load_csv_data()
            
            # Extract product labels from header
            labels = self.extract_product_labels(df)
            
            # Convert rows to documents with key-value pairs
            documents = self.convert_rows_to_documents(df, labels)
            pdf_documents = self.load_pdf_data()
            documents.extend([doc.page_content for doc in pdf_documents])   
                        
            # Chunk documents
            chunked_documents = self.chunk_documents(documents)
            
            # Store in PGVector
            logger.info(f"Storing {len(chunked_documents)} chunks in PGVector...")
            try:
                self._vectorstore = PGVector.from_documents(
                    documents=chunked_documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_string=self.pgvector_connection_string,                                        
                    pre_delete_collection=True,  # Set to True to rebuild from scratch
                )
                logger.info("Successfully stored embeddings in PGVector")
            except Exception as e:
                logger.error(f"Error storing in PGVector: {e}")
                raise
            
            stats = {
                "status": "success",
                "total_products": len(df),
                "total_documents": len(documents),
                "total_chunks": len(chunked_documents),
                "product_labels": labels,
                "collection_name": self.collection_name,                
            }
            
            logger.info("=" * 60)
            logger.info("RAG Pipeline Build Completed Successfully!")
            logger.info(f"Statistics: {stats}")
            logger.info("=" * 60)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error building RAG pipeline: {e}")
            raise
    
    def query_rag_pipeline(self, query: str) -> str:
        """
        Query the RAG pipeline and generate response
        
        This function:
        1. Retrieves top-k matching documents from PGVector based on query
        2. Formats the context from retrieved documents
        3. Sends context + query to local Ollama LLM
        4. Returns the LLM response
        
        Args:
            query: User's natural language question
            
        Returns:
            LLM-generated response using retrieved context
        """
        logger.info("=" * 60)
        logger.info(f"Processing Query: {query}")
        logger.info("=" * 60)
        
        try:
            # Retrieve relevant documents from PGVector
            logger.info(f"Retrieving top {self.top_k} documents from PGVector...")
            retriever = self.vectorstore.as_retriever(top_k=10)           
            
            retrieved_docs = retriever.invoke(query)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Format context from retrieved documents
            context = "\n\n".join(
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            )
            
            logger.info("Context retrieved:\n" + context[:200] + "...")
            
            # Create prompt with context and query
            prompt = ChatPromptTemplate.from_template(
                """Use the following product information to answer the user's question. 
Answer only using the provided context. If the information is not available in the context, 
say "I don't have enough information to answer that question. Always say "thanks for asking!" at the end of the answer."

Context:
{context}

Question: {question}

Helpful Answer:"""
            )
            
            # Invoke LLM with context
            logger.info("Sending context and query to LLM...")
            result = self.llm.invoke(
                prompt.format_messages(context=context, question=query)
            )
            
            response = result.content
            logger.info(f"LLM Response: {response[:200]}...")
            logger.info("=" * 60)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {e}")
            raise
    
    def rebuild_vectorstore(self) -> dict:
        """
        Rebuild the entire vectorstore from scratch
        
        Useful for refreshing the index if CSV data has changed.
        
        Returns:
            Dictionary with rebuild statistics
        """
        logger.info("Rebuilding vectorstore...")
        self._vectorstore = None  # Clear cached vectorstore
        return self.build_rag_pipeline()


# Convenience functions for easy access
_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


def build_rag_pipeline() -> dict:
    """Build the RAG pipeline"""
    pipeline = get_rag_pipeline()
    return pipeline.build_rag_pipeline()


def query_rag_pipeline(query: str) -> str:
    """Query the RAG pipeline"""
    pipeline = get_rag_pipeline()
    return pipeline.query_rag_pipeline(query)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Build the pipeline (indexes all products)
    print("\n" + "=" * 60)
    print("Building RAG Pipeline...")
    print("=" * 60)
    stats = pipeline.build_rag_pipeline()
    print(f"\nPipeline built successfully!")
    print(f"Total products: {stats['total_products']}")
    print(f"Total chunks: {stats['total_chunks']}")
    
    # Example queries
    sample_queries = [
        "Get information about electronics products rating>4?",        
    ]
    
    print("\n" + "=" * 60)
    print("Example Queries:")
    print("=" * 60)
    for sample_query in sample_queries:
        print(f"\nQuery: {sample_query}")
        try:
            response = pipeline.query_rag_pipeline(sample_query)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
