import streamlit as st
import sys
import os

# Add the parent directory to the path to import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_pipeline import rag_pipeline


class StudyMateUI:
    def __init__(self):
        """Initialize the Streamlit UI."""
        self.setup_page_config()
        self.setup_custom_css()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="StudyMate AI", 
            layout="wide", 
            page_icon="📚"
        )
    
    def setup_custom_css(self):
        """Apply custom CSS styling (Light Theme with Indigo Blue)."""
        st.markdown("""
        <style>
            body {
                background-color: #FFFFFF;
                color: #1A1A1A;
            }
            .main-title {
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                color: #3F51B5; /* Indigo Blue */
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                font-size: 18px;
                color: #555555;
                margin-bottom: 30px;
            }
            .stButton>button {
                background: #3F51B5; /* Indigo Blue */
                color: #FFFFFF;
                border-radius: 10px;
                padding: 8px 20px;
                font-size: 16px;
                border: none;
                font-weight: bold;
            }
            .stButton>button:hover {
                background: #303F9F; /* Darker Indigo */
                color: white;
            }
            .stTextInput>div>div>input {
                border-radius: 8px;
                border: 1px solid #3F51B5;
                background-color: #FFFFFF;
                color: #1A1A1A;
                padding: 8px;
            }
            .qa-box {
                background-color: #F5F5F5;
                border: 1px solid #3F51B5;
                border-radius: 10px;
                padding: 12px 15px;
                margin-bottom: 15px;
            }
            .qa-box .question {
                color: #3F51B5;
                font-weight: bold;
                margin-bottom: 8px;
            }
            .qa-box .answer {
                color: #1A1A1A;
            }
            .reset-success {
                background-color: #E8F0FE;
                border: 1px solid #3F51B5;
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0;
                color: #3F51B5;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
        if "pdf_uploaded" not in st.session_state:
            st.session_state.pdf_uploaded = False
        if "page_count" not in st.session_state:
            st.session_state.page_count = 0
        if "chunk_count" not in st.session_state:
            st.session_state.chunk_count = 0
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "question_input" not in st.session_state:
            st.session_state.question_input = ""
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
    
    def reset_session(self):
        """Reset the entire session - clear all data and return to initial state."""
        st.session_state.vectorstore = None
        st.session_state.pdf_uploaded = False
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.session_state.question_input = ""
        st.session_state.uploaded_files = []
        st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with file upload and controls."""
        with st.sidebar:
            st.image(
                "https://images.crunchbase.com/image/upload/c_pad,h_256,w_256,f_auto,q_auto:eco,dpr_1/0a0d9095f6f24873aafd805a61fd71db", 
                width=200,
            )
            st.markdown("<h2 style='color:#3F51B5;'>📚 StudyMate AI</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color:#555555;'>Your intelligent learning companion</p>", unsafe_allow_html=True)
            
            if st.session_state.pdf_uploaded:
                st.success(f"📄 Session Active: {st.session_state.page_count} pages loaded")
            else:
                st.info("📂 No PDFs loaded - Upload to start")
            
            pdf_files = None
            if not st.session_state.pdf_uploaded:
                pdf_files = st.file_uploader(
                    "Upload PDF files", 
                    type=["pdf"], 
                    accept_multiple_files=True,
                    key="pdf_uploader"
                )
            
            if st.button("🔄 Reset Session", type="primary"):
                self.reset_session()
            
            st.markdown("---")
            st.info("💡 **Session-based RAG**\n- No files saved to disk\n- All data in memory only\n- Reset clears everything")
            
            return pdf_files
    
    def process_pdfs(self, pdf_files):
        """Process uploaded PDF files in memory."""
        if pdf_files and not st.session_state.pdf_uploaded:
            with st.spinner("📚 Processing PDFs in memory..."):
                try:
                    vectorstore, pages, chunks = rag_pipeline.build_vectorstore_in_memory(pdf_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.page_count = pages
                    st.session_state.chunk_count = chunks
                    st.session_state.pdf_uploaded = True
                    st.session_state.uploaded_files = [f.name for f in pdf_files]
                    
                    st.success(f"✅ {pages} pages loaded | {chunks} chunks created (In Memory)")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")
                    st.session_state.pdf_uploaded = False
                    st.session_state.vectorstore = None
    
    def submit_question(self):
        """Handle question submission."""
        user_question = st.session_state.question_input.strip()
        if not user_question:
            return
        
        if not st.session_state.vectorstore:
            st.error("❌ No documents loaded. Please upload PDFs first.")
            return
        
        st.session_state.chat_history.append({
            "question": user_question, 
            "answer": "⏳ Generating response...", 
            "context": []
        })
        
        try:
            answer, context_list = rag_pipeline.make_prediction(st.session_state.vectorstore, user_question)
            st.session_state.chat_history[-1] = {
                "question": user_question, 
                "answer": answer, 
                "context": context_list
            }
        except Exception as e:
            st.session_state.chat_history[-1] = {
                "question": user_question, 
                "answer": f"❌ Error generating response: {str(e)}", 
                "context": []
            }
        
        st.session_state.question_input = ""
    
    def render_main_content(self):
        """Render the main content area."""
        st.markdown("<div class='main-title'>Welcome to StudyMate AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Upload your study materials, ask questions, and learn smarter 🎯</div>", unsafe_allow_html=True)

        if st.session_state.pdf_uploaded and st.session_state.vectorstore:
            st.info(f"📚 **Active Session**: {len(st.session_state.uploaded_files)} files loaded ({st.session_state.page_count} pages, {st.session_state.chunk_count} chunks)")
            
            st.text_input(
                "❓ Ask a question about your documents:", 
                key="question_input", 
                on_change=self.submit_question,
                placeholder="Type your question here and press Enter..."
            )
            
            if st.session_state.chat_history:
                st.markdown("### 💬 Conversation History")
                for chat in reversed(st.session_state.chat_history):
                    st.markdown(
                        f"""
                        <div class='qa-box'>
                            <div class='question'>Q: {chat['question']}</div>
                            <div class='answer'>A: {chat['answer']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("### 💬 Ready for Questions!")
                st.markdown("Ask any question about your uploaded documents above.")
                
        else:
            st.markdown("### 🚀 Get Started")
            st.markdown("""
            1. **Upload PDFs** using the sidebar file uploader  
            2. **Wait for processing** (files are processed in memory only)  
            3. **Ask questions** about your documents  
            4. **Reset session** anytime to start fresh  
            """)
            st.info("💡 This is a **session-based system** - no files are saved permanently!")
    
    def run(self):
        """Main method to run the Streamlit app."""
        pdf_files = self.render_sidebar()
        self.process_pdfs(pdf_files)
        self.render_main_content()


def main():
    app = StudyMateUI()
    app.run()


if __name__ == "__main__":
    main()
