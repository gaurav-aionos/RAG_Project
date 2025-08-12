import os
import tempfile
from typing import List, Tuple, Any, Optional
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with configurations."""
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = 'openai/gpt-oss-120b'
        self.embedding_model_name = 'thenlper/gte-large'
        
        # Prompt templates
        self.qna_system_message = """
        You are a helpful AI assistant.
        User input will have the context required to answer user questions.
        The context will begin with: ###Context.
        Only answer using the context provided; if not found, say "I don't know".
        """
        
        self.qna_user_message_template = """
        ###Context
        {context}

        ###Question
        {question}
        """
    
    def build_vectorstore_in_memory(self, pdf_files: List[Any]) -> Tuple[Any, int, int]:
        """
        Build vector store from uploaded PDF files in memory (no persistence).
        
        Args:
            pdf_files: List of uploaded PDF file objects
            
        Returns:
            Tuple of (vectorstore, page_count, chunk_count)
        """
        all_docs = []
        temp_files = []

        try:
            # Process each PDF file using temporary files
            for idx, pdf_file in enumerate(pdf_files, start=1):
                # Create a temporary file for each PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(pdf_file.getvalue())  # Use getvalue() instead of read()
                    temp_path = temp_file.name
                    temp_files.append(temp_path)

                # Load the PDF from temporary file
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name='cl100k_base',
                chunk_size=512,
                chunk_overlap=16
            )
            chunks = text_splitter.split_documents(all_docs)

            # Create in-memory vector store (no persistence)
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                # No persist_directory = in-memory only
            )
            
            return vectorstore, len(all_docs), len(chunks)
            
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass  # File already deleted or doesn't exist
    
    def make_prediction(self, vectorstore: Any, user_input: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        Generate prediction based on user input and in-memory vector store.
        
        Args:
            vectorstore: The in-memory vector store to query
            user_input: User's question
            k: Number of relevant documents to retrieve
            
        Returns:
            Tuple of (prediction, context_list)
        """
        # Create retriever directly from the vectorstore
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': k}
        )
        
        relevant_document_chunks = retriever.get_relevant_documents(user_input)
        context_list = [d.page_content for d in relevant_document_chunks]
        context_for_query = ". ".join(context_list)

        prompt = [
            {'role': 'system', 'content': self.qna_system_message},
            {'role': 'user', 'content': self.qna_user_message_template.format(
                context=context_for_query,
                question=user_input
            )}
        ]

        client = Groq(api_key=self.api_key)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=0
            )
            prediction = response.choices[0].message.content.strip()
        except Exception as e:
            prediction = f"❌ Error: {e}"

        return prediction, context_list


# Create a singleton instance
rag_pipeline = RAGPipeline()
