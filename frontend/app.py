import streamlit as st
import sys
import os
import re

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
        """Apply custom CSS styling (IndiGo Pastel Blue Theme)."""
        st.markdown("""
        <style>
            body {
                background-color: #F8FAFF;
                color: #1A1A1A;
            }
            .main-title {
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                color: #1E3A8A; /* IndiGo Deep Blue */
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                font-size: 18px;
                color: #64748B;
                margin-bottom: 30px;
            }
            .stButton>button {
                background: linear-gradient(135deg, #3B82F6, #1E40AF); /* IndiGo Blue Gradient */
                color: #FFFFFF;
                border-radius: 12px;
                padding: 10px 24px;
                font-size: 16px;
                border: none;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
            }
            .stButton>button:hover {
                background: linear-gradient(135deg, #2563EB, #1D4ED8);
                transform: translateY(-1px);
                box-shadow: 0 6px 8px rgba(59, 130, 246, 0.4);
            }
            .stTextInput>div>div>input {
                border-radius: 10px;
                border: 2px solid #BFDBFE;
                background-color: #FFFFFF;
                color: #1A1A1A;
                padding: 12px;
            }
            .stTextInput>div>div>input:focus {
                border-color: #3B82F6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }
            .qa-box {
                background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
                border: 2px solid #BAE6FD;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
            }
            .qa-box .question {
                color: #1E40AF;
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 16px;
            }
            .qa-box .answer {
                color: #1F2937;
                line-height: 1.6;
            }
            .citation-box {
                background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                padding: 12px;
                margin: 8px 0;
                font-size: 14px;
            }
            .citation-header {
                font-weight: bold;
                color: #1E40AF;
                margin-bottom: 6px;
            }
            .quiz-container {
                background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
                border: 3px solid #93C5FD;
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(147, 197, 253, 0.2);
            }
            .quiz-header {
                font-size: 24px;
                font-weight: bold;
                color: #1E40AF;
                margin-bottom: 20px;
                text-align: center;
                padding: 12px;
                background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
                border-radius: 12px;
                border: 2px solid #93C5FD;
            }
            .quiz-question {
                background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
                border: 2px solid #7DD3FC;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(125, 211, 252, 0.2);
            }
            .question-title {
                font-weight: bold;
                font-size: 18px;
                color: #0F172A;
                margin-bottom: 15px;
                padding: 10px;
                background: rgba(59, 130, 246, 0.1);
                border-radius: 8px;
                border-left: 4px solid #3B82F6;
            }
            .quiz-option {
                margin: 8px 0;
                padding: 12px 16px;
                border-radius: 8px;
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                transition: all 0.2s ease;
            }
            .quiz-option:hover {
                background-color: #F9FAFB;
            }
            .quiz-option.correct {
                background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
                border: 2px solid #22C55E;
                font-weight: bold;
                color: #15803D;
                transform: scale(1.02);
                box-shadow: 0 2px 4px rgba(34, 197, 94, 0.2);
            }
            .quiz-correct-answer {
                color: #059669;
                font-weight: bold;
                margin-top: 15px;
                padding: 10px;
                background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
                border-radius: 8px;
                border-left: 4px solid #10B981;
            }
            .quiz-explanation {
                color: #374151;
                font-style: italic;
                margin-top: 12px;
                padding: 12px;
                background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
                border-radius: 8px;
                border-left: 4px solid #6B7280;
                line-height: 1.5;
            }
            .reset-success {
                background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
                border: 2px solid #93C5FD;
                border-radius: 12px;
                padding: 12px;
                margin: 12px 0;
                color: #1E40AF;
            }
            .mode-indicator {
                background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
                border: 2px solid #93C5FD;
                border-radius: 10px;
                padding: 12px 16px;
                margin: 12px 0;
                font-weight: bold;
                color: #1E40AF;
                text-align: center;
                box-shadow: 0 2px 4px rgba(147, 197, 253, 0.2);
            }
            .sidebar .element-container {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 8px;
                margin: 4px 0;
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
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "Q&A"
        if "quiz_topic" not in st.session_state:
            st.session_state.quiz_topic = ""
        if "num_questions" not in st.session_state:
            st.session_state.num_questions = 5
    
    def reset_session(self):
        """Reset the entire session - clear all data and return to initial state."""
        st.session_state.vectorstore = None
        st.session_state.pdf_uploaded = False
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.session_state.question_input = ""
        st.session_state.uploaded_files = []
        st.session_state.app_mode = "Q&A"
        st.session_state.quiz_topic = ""
        st.session_state.num_questions = 5
        st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with file upload and controls."""
        with st.sidebar:
            st.image(
                "https://images.crunchbase.com/image/upload/c_pad,h_256,w_256,f_auto,q_auto:eco,dpr_1/0a0d9095f6f24873aafd805a61fd71db", 
                width=200,
            )
            st.markdown("<h2 style='color:#1E40AF;'>📚 StudyMate AI</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748B;'>Your intelligent learning companion</p>", unsafe_allow_html=True)
            
            # Mode selection dropdown
            st.markdown("### 🎯 Mode Selection")
            app_mode = st.selectbox(
                "Choose your mode:",
                ["Q&A", "Q&A with Citations", "Quiz Generator"],
                index=["Q&A", "Q&A with Citations", "Quiz Generator"].index(st.session_state.app_mode),
                key="mode_selector"
            )
            st.session_state.app_mode = app_mode
            
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
            
            # Mode-specific controls
            if st.session_state.app_mode == "Quiz Generator" and st.session_state.pdf_uploaded:
                st.markdown("### 🎯 Quiz Settings")
                st.session_state.quiz_topic = st.text_input(
                    "Quiz Topic (optional):", 
                    value=st.session_state.quiz_topic,
                    placeholder="Leave empty for general quiz"
                )
                st.session_state.num_questions = st.slider(
                    "Number of Questions:", 
                    min_value=3, 
                    max_value=10, 
                    value=st.session_state.num_questions
                )
            
            st.markdown("---")
            st.info("💡 **Features**\n- Q&A: Basic question answering\n- Citations: Q&A with source references\n- Quiz: Generate practice questions")
            
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
        """Handle question submission based on current mode."""
        user_question = st.session_state.question_input.strip()
        if not user_question:
            return
        
        if not st.session_state.vectorstore:
            st.error("❌ No documents loaded. Please upload PDFs first.")
            return
        
        if st.session_state.app_mode == "Quiz Generator":
            # Generate quiz
            st.session_state.chat_history.append({
                "question": f"Generate quiz: {user_question}", 
                "answer": "⏳ Generating quiz...", 
                "context": [],
                "type": "quiz"
            })
            
            try:
                quiz = rag_pipeline.generate_quiz(
                    st.session_state.vectorstore, 
                    user_question, 
                    st.session_state.num_questions
                )
                st.session_state.chat_history[-1] = {
                    "question": f"Generate quiz: {user_question}", 
                    "answer": quiz, 
                    "context": [],
                    "type": "quiz"
                }
            except Exception as e:
                st.session_state.chat_history[-1] = {
                    "question": f"Generate quiz: {user_question}", 
                    "answer": f"❌ Error generating quiz: {str(e)}", 
                    "context": [],
                    "type": "quiz"
                }
        
        elif st.session_state.app_mode == "Q&A with Citations":
            # Q&A with citations
            st.session_state.chat_history.append({
                "question": user_question, 
                "answer": "⏳ Generating response with citations...", 
                "context": [],
                "type": "qa_citations"
            })
            
            try:
                answer, detailed_context = rag_pipeline.make_prediction_with_citations(
                    st.session_state.vectorstore, 
                    user_question
                )
                st.session_state.chat_history[-1] = {
                    "question": user_question, 
                    "answer": answer, 
                    "context": detailed_context,
                    "type": "qa_citations"
                }
            except Exception as e:
                st.session_state.chat_history[-1] = {
                    "question": user_question, 
                    "answer": f"❌ Error generating response: {str(e)}", 
                    "context": [],
                    "type": "qa_citations"
                }
        
        else:  # Regular Q&A
            st.session_state.chat_history.append({
                "question": user_question, 
                "answer": "⏳ Generating response...", 
                "context": [],
                "type": "qa"
            })
            
            try:
                answer, context_list = rag_pipeline.make_prediction(
                    st.session_state.vectorstore, 
                    user_question
                )
                st.session_state.chat_history[-1] = {
                    "question": user_question, 
                    "answer": answer, 
                    "context": context_list,
                    "type": "qa"
                }
            except Exception as e:
                st.session_state.chat_history[-1] = {
                    "question": user_question, 
                    "answer": f"❌ Error generating response: {str(e)}", 
                    "context": [],
                    "type": "qa"
                }
        
        st.session_state.question_input = ""
    
    def generate_quiz_from_topic(self):
        """Generate quiz from topic input."""
        if not st.session_state.vectorstore:
            st.error("❌ No documents loaded. Please upload PDFs first.")
            return
        
        topic = st.session_state.quiz_topic if st.session_state.quiz_topic.strip() else "general concepts"
        
        with st.spinner("🎯 Generating quiz..."):
            try:
                quiz = rag_pipeline.generate_quiz(
                    st.session_state.vectorstore, 
                    topic, 
                    st.session_state.num_questions
                )
                st.session_state.chat_history.append({
                    "question": f"Quiz on: {topic}", 
                    "answer": quiz, 
                    "context": [],
                    "type": "quiz"
                })
            except Exception as e:
                st.error(f"❌ Error generating quiz: {str(e)}")
    
    def parse_quiz(self, quiz_text):
        """Parse quiz text into structured format with improved correct answer detection."""
        questions = []
        
        # More flexible regex patterns to handle various formats
        patterns = [
            r'(?:\*\*)?Question (\d+)(?:\*\*)?:?\s*(.*?)(?=(?:\*\*)?Question \d+|$)',
            r'(\d+)\.\s*(.*?)(?=\d+\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, quiz_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    question_content = match.group(2).strip()
                    lines = [line.strip() for line in question_content.split('\n') if line.strip()]
                    
                    if not lines:
                        continue
                    
                    question_text = lines[0]
                    options = []
                    correct_answer = ""
                    explanation = ""
                    
                    i = 1
                    # Extract options (A), B), C), D) or A., B., C., D.)
                    while i < len(lines):
                        line = lines[i]
                        if re.match(r'^[ABCD][\)\.]', line):
                            options.append(line)
                        elif line.lower().startswith("correct answer"):
                            # More flexible correct answer extraction
                            answer_match = re.search(r'correct answer:?\s*([ABCD])', line, re.IGNORECASE)
                            if answer_match:
                                correct_answer = answer_match.group(1).upper()
                        elif line.lower().startswith("explanation"):
                            explanation = re.sub(r'^explanation:?\s*', '', line, flags=re.IGNORECASE)
                        i += 1
                    
                    if question_text and options and correct_answer:
                        questions.append({
                            "question": question_text,
                            "options": options,
                            "correct": correct_answer,
                            "explanation": explanation
                        })
                break
        
        return questions
    
    def render_quiz_display(self, chat):
        """Render quiz in a user-friendly format with proper correct answer highlighting."""
        quiz_questions = self.parse_quiz(chat['answer'])
        
        if not quiz_questions:
            # Fallback to raw text if parsing fails
            st.markdown(
                f"""
                <div class='quiz-container'>
                    <div class='quiz-header'>🎯 Quiz Topic: {chat['question'].replace('Generate quiz: ', '').replace('Quiz on: ', '')}</div>
                    <div style="padding: 15px; background: rgba(255,255,255,0.8); border-radius: 8px;">
                        {chat['answer'].replace('\n', '<br>')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            return
        
        # Display parsed quiz with enhanced styling
        st.markdown(
            f"""
            <div class='quiz-container'>
                <div class='quiz-header'>🎯 Quiz: {chat['question'].replace('Generate quiz: ', '').replace('Quiz on: ', '')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        for i, q in enumerate(quiz_questions, 1):
            # Create option HTML with correct answer highlighted
            options_html = ""
            for option in q['options']:
                option_letter = option[0].upper()  # Get the letter (A, B, C, D)
                if option_letter == q['correct'].upper():
                    options_html += f"<div class='quiz-option correct'>✅ {option} <strong>(CORRECT)</strong></div>"
                else:
                    options_html += f"<div class='quiz-option'>{option}</div>"
            
            st.markdown(
                f"""
                <div class='quiz-question'>
                    <div class='question-title'>Q{i}: {q['question']}</div>
                    {options_html}
                    <div class='quiz-correct-answer'>
                        ✅ <strong>Correct Answer: {q['correct']}</strong>
                    </div>
                    <div class='quiz-explanation'>
                        💡 <strong>Explanation:</strong> {q['explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def render_main_content(self):
        """Render the main content area."""
        st.markdown("<div class='main-title'>Welcome to StudyMate AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Upload your study materials, ask questions, and learn smarter 🎯</div>", unsafe_allow_html=True)

        # Display current mode
        mode_icons = {
            "Q&A": "💬", 
            "Q&A with Citations": "📝", 
            "Quiz Generator": "🎯"
        }
        st.markdown(
            f"<div class='mode-indicator'>{mode_icons.get(st.session_state.app_mode, '💬')} Current Mode: {st.session_state.app_mode}</div>", 
            unsafe_allow_html=True
        )

        if st.session_state.pdf_uploaded and st.session_state.vectorstore:
            st.info(f"📚 **Active Session**: {len(st.session_state.uploaded_files)} files loaded ({st.session_state.page_count} pages, {st.session_state.chunk_count} chunks)")
            
            # Input based on mode
            if st.session_state.app_mode == "Quiz Generator":
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text_input(
                        "🎯 Enter topic for quiz or leave empty for general quiz:", 
                        key="question_input", 
                        on_change=self.submit_question,
                        placeholder="Type quiz topic here and press Enter..."
                    )
                with col2:
                    if st.button("Generate Quiz", type="primary"):
                        self.generate_quiz_from_topic()
            else:
                input_placeholder = "Type your question here and press Enter..."
                if st.session_state.app_mode == "Q&A with Citations":
                    input_placeholder = "Ask a question (citations will be included)..."
                
                st.text_input(
                    "❓ Ask a question about your documents:", 
                    key="question_input", 
                    on_change=self.submit_question,
                    placeholder=input_placeholder
                )
            
            # Display conversation history
            if st.session_state.chat_history:
                st.markdown("### 💬 Conversation History")
                for chat in reversed(st.session_state.chat_history):
                    if chat.get("type") == "quiz":
                        # Enhanced Quiz Display
                        self.render_quiz_display(chat)
                        
                    elif chat.get("type") == "qa_citations":
                        # Q&A with citations display
                        st.markdown(
                            f"""
                            <div class='qa-box'>
                                <div class='question'>Q: {chat['question']}</div>
                                <div class='answer'>A: {chat['answer']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Display citations
                        if chat['context']:
                            st.markdown("**📚 Sources & Citations:**")
                            for i, ctx in enumerate(chat['context'], 1):
                                source_file = ctx.get('source', 'Unknown').split('/')[-1] if ctx.get('source') else 'Unknown'
                                st.markdown(
                                    f"""
                                    <div class='citation-box'>
                                        <div class='citation-header'>Source {i}: {source_file} (Page {ctx.get('page', 'Unknown')})</div>
                                        <div>{ctx.get('content', '')[:200]}{'...' if len(ctx.get('content', '')) > 200 else ''}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        # Regular Q&A display
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
                if st.session_state.app_mode == "Quiz Generator":
                    st.markdown("Generate practice quizzes based on your uploaded documents.")
                elif st.session_state.app_mode == "Q&A with Citations":
                    st.markdown("Ask questions and get answers with source citations.")
                else:
                    st.markdown("Ask any question about your uploaded documents.")
                
        else:
            st.markdown("### 🚀 Get Started")
            st.markdown("""
            1. **Upload PDFs** using the sidebar file uploader  
            2. **Select your mode** (Q&A, Citations, or Quiz)  
            3. **Wait for processing** (files are processed in memory only)  
            4. **Start learning** with questions or quizzes  
            5. **Reset session** anytime to start fresh  
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
