import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load env variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = 'openai/gpt-oss-120b'

# Prompt templates
qna_system_message = """
You are a helpful AI assistant.
User input will have the context required to answer user questions.
The context will begin with: ###Context.
Only answer using the context provided; if not found, say "I don't know".
"""

qna_user_message_template = """
###Context
{context}

###Question
{question}
"""

# -------- Functions --------
def build_vectorstore(pdf_files, persist_directory):
    os.makedirs(persist_directory, exist_ok=True)
    all_docs = []

    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(persist_directory, f"uploaded_{idx}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=512,
        chunk_overlap=16
    )
    chunks = text_splitter.split_documents(all_docs)

    embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore, len(all_docs), len(chunks)

def make_prediction(vectorstore, user_input, k=5):
    embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
    vectorstore_persisted = Chroma(
        persist_directory=vectorstore._persist_directory,
        embedding_function=embedding_model
    )
    retriever = vectorstore_persisted.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    relevant_document_chunks = retriever.get_relevant_documents(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    prompt = [
        {'role': 'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
        )}
    ]

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        prediction = f"❌ Error: {e}"

    return prediction, context_list

# -------- New: Auto-delete persisted DB on session end/reset --------
def delete_persisted_db(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

# -------- Streamlit UI Config --------
st.set_page_config(page_title="StudyMate AI", layout="wide", page_icon="📚")

# Custom CSS for yellow & dark grey theme
st.markdown("""
<style>
    body {
        background-color: #1E1E1E;
        color: #EAEAEA;
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #BBBBBB;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: #FFD700;
        color: #000000;
        border-radius: 10px;
        padding: 8px 20px;
        font-size: 16px;
        border: none;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #FFD700;
        background-color: #2A2A2A;
        color: #FFFFFF;
        padding: 8px;
    }
    /* Combined Q&A box styling */
    .qa-box {
        background-color: #2A2A2A;
        border: 1px solid #FFD700;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 15px;
    }
    .qa-box .question {
        color: #FFD700;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .qa-box .answer {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://media.istockphoto.com/id/2058507417/vector/artificial-intelligence-icon-sign-logo-in-the-circuit-line-style-ai-processor-vector-icon.jpg?s=612x612&w=0&k=20&c=0kb5zgMcapsLizKDzLP-Y72UyyVACOy2cEZC8hNIboE=", width=200)
    st.markdown("<h2 style='color:#FFD700;'>📚 StudyMate AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#CCCCCC;'>Your intelligent learning companion</p>", unsafe_allow_html=True)
    pdf_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
    persist_dir = "chroma_db"

    if st.button("🔄 Reset Session"):
        delete_persisted_db(persist_dir)  # Delete DB folder on reset
        st.session_state.clear()
        st.success("Session reset! Upload new PDFs.")
    st.markdown("---")
    st.info("💡 Uses RAG to answer from your PDFs.")

# Session states
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

# PDF Processing
if pdf_files and not st.session_state.pdf_uploaded:
    with st.spinner("📚 Processing PDFs..."):
        vectorstore, pages, chunks = build_vectorstore(pdf_files, persist_dir)
        st.session_state.vectorstore = vectorstore
        st.session_state.page_count = pages
        st.session_state.chunk_count = chunks
        st.session_state.pdf_uploaded = True
    st.success(f"✅ {pages} pages loaded | {chunks} chunks created.")

elif st.session_state.pdf_uploaded and len(st.session_state.chat_history) == 0:
    st.success(f"📄 PDFs loaded: {st.session_state.page_count} pages | {st.session_state.chunk_count} chunks")

# Question submission
def submit_question():
    user_question = st.session_state.question_input.strip()
    if not user_question:
        return
    st.session_state.chat_history.append({"question": user_question, "answer": "⏳ Generating...", "context": []})
    answer, context_list = make_prediction(st.session_state.vectorstore, user_question)
    st.session_state.chat_history[-1] = {"question": user_question, "answer": answer, "context": context_list}
    st.session_state.question_input = ""

# Main UI
st.markdown("<div class='main-title'>Welcome to StudyMate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your study materials, ask questions, and learn smarter 🎯</div>", unsafe_allow_html=True)

if st.session_state.vectorstore:
    st.text_input("❓ Ask a question:", key="question_input", on_change=submit_question)
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
    st.warning("📂 Please upload one or more PDFs to start.")

# -------- New: Cleanup DB when Streamlit session ends --------
if st.session_state.get("pdf_uploaded") is False:
    delete_persisted_db(persist_dir)
