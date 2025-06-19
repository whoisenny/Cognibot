from dotenv import load_dotenv
import streamlit as st

st.set_page_config(
    page_title="Cognibot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from openpipe import OpenAI
from typing import Optional
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import PyPDF2
import os
import faiss

from sentence_transformers import SentenceTransformer
import numpy as np
import datetime
import uuid

load_dotenv()

@st.cache_resource
def get_openpipe_client():
    api_key = os.getenv("OPENPIPE_API_KEY")
    if not api_key:
        st.error("OPENPIPE_API_KEY not found. Please check your .env file.")
        st.stop()
    return OpenAI(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# FAISS setup
EMBED_DIM = 384  # for all-MiniLM-L6-v2

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    st.session_state.metadata = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())[:8]

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def process_pdf_for_rag(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        chunks = chunk_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF for RAG: {str(e)}")
        return []

def add_documents_to_knowledge_base(chunks, filename):
    model = load_embedding_model()
    new_vectors = []
    new_metadata = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:
            embedding = model.encode([chunk])[0].astype('float32')
            new_vectors.append(embedding)
            new_metadata.append({
                'content': chunk,
                'source': filename,
                'chunk_id': i
            })
    
    if new_vectors:
        st.session_state.faiss_index.add(np.array(new_vectors))
        st.session_state.metadata.extend(new_metadata)

def retrieve_relevant_chunks(query, top_k=3):
    if st.session_state.faiss_index.ntotal == 0:
        return []

    model = load_embedding_model()
    query_vector = model.encode([query])[0].astype('float32').reshape(1, -1)

    D, I = st.session_state.faiss_index.search(query_vector, top_k)

    relevant_chunks = []
    for idx, distance in zip(I[0], D[0]):
        if idx < len(st.session_state.metadata):
            meta = st.session_state.metadata[idx]
            similarity = 1 - distance  # Optional: convert L2 distance to similarity
            relevant_chunks.append({
                'content': meta['content'],
                'source': meta['source'],
                'similarity': similarity
            })
    return relevant_chunks

def add_to_chat_history(user_message: str, ai_response: str, relevant_chunks: list = None):
    chat_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'session_id': st.session_state.current_session_id,
        'user_message': user_message,
        'ai_response': ai_response,
        'relevant_chunks': relevant_chunks or []
    }
    st.session_state.chat_history.append(chat_entry)

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.current_session_id = str(uuid.uuid4())[:8]

def export_chat_history():
    if not st.session_state.chat_history:
        return "No chat history to export."
    export_text = f"Cognibot Chat History\nExported on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for i, entry in enumerate(st.session_state.chat_history, 1):
        export_text += f"--- Conversation {i} ---\n"
        export_text += f"Time: {entry['timestamp']}\n"
        export_text += f"User: {entry['user_message']}\n"
        export_text += f"Cognibot: {entry['ai_response']}\n\n"
    return export_text

def export_to_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    for entry in chat_history:
        pdf.multi_cell(0, 10, f"You: {entry['user_message']}")
        pdf.multi_cell(0, 10, f"Bot: {entry['ai_response']}")
        pdf.ln()
    pdf.output("chat_summary.pdf")

def get_ai_response(prompt: str) -> Optional[str]:
    try:
        client = get_openpipe_client()
        relevant_chunks = []
        if st.session_state.faiss_index.ntotal > 0:
            relevant_chunks = retrieve_relevant_chunks(prompt, top_k=3)
        if relevant_chunks:
            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
            rag_prompt = f"""Context from uploaded documents:
{context}

User question: {prompt}

Please answer the question using both your knowledge and the provided context. If the context is relevant, reference it in your answer."""
            with st.expander("📋 Retrieved Context from Your Documents"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"**Source {i}:** {chunk['source']} (Similarity: {chunk['similarity']:.2f})")
                    st.write(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                    st.divider()
        else:
            rag_prompt = prompt
        completion = client.chat.completions.create(
            model="openpipe:eni-final-year",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0,
            openpipe={
                "tags": {
                    "prompt_id": "counting",
                    "any_key": "any_value"
                }
            }
        )
        ai_response = completion.choices[0].message.content
        add_to_chat_history(prompt, ai_response, relevant_chunks)
        return ai_response
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None

def generate_diagram():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_title("Simple waveform")
    return fig

def main():
    st.markdown(
    """
    <h1 style='text-align: center; color: #FFFFFF; font-family:  Helvetica, sans-serif !important; font-size: 60px;
               margin-bottom: 10px; margin-top: 60px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
        Cognibot🤖
    </h1>
    """, 
    unsafe_allow_html=True
    )
    st.markdown("""
    Welcome to your AI-powered Computer Engineering assistant! 
    I can help you with:
    - Programming concepts and code explanations
    - Computer architecture questions
    - Digital logic and circuit design
    - Operating systems concepts
    - And much more!
    """)

    with st.sidebar:
        st.sidebar.title("🔧 Navigation")
        section = st.sidebar.selectbox("Go to", ["Chat", "Chat History", "Knowledge Base", "Settings"])
        st.markdown("---")
        st.markdown("### 💬 Current Session")
        st.write(f"Session ID: {st.session_state.current_session_id}")
        st.write(f"Messages: {len(st.session_state.chat_history)}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Chat"):
                clear_chat_history()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.chat_history = []
                st.session_state.metadata = []
                st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)
                st.success("All data cleared!")
                st.rerun()
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        st.write(f"Documents loaded: {len(set([doc['source'] for doc in st.session_state.metadata]))}")
        st.write(f"Total chunks: {len(st.session_state.metadata)}")
        if st.button("Clear Knowledge Base"):
            st.session_state.metadata = []
            st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)
            st.success("Knowledge base cleared!")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI assistant is specifically trained to help undergraduate Computer Engineering Students.
        Upload documents to enhance responses with your specific materials!
        """)

    if section == "Chat":
        # --- Display chat history ONCE, at the top ---
        st.markdown("### 💬 Chat History")
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.write(f"**You:** {entry['user_message']}")
                st.write(f"**Cognibot:** {entry['ai_response']}")
                st.markdown("---")
        else:
            st.info("No messages yet. Start the conversation below!")

        st.write("")

        # --- Prompt guide for students ---
        st.markdown("""
        **Try asking me things like:**
        - Explain the concept of pipelining in computer architecture.
        - Summarize the OSI model.
        - Generate 5 quiz questions about microprocessors.
        - List key formulas for digital logic design.
        - How does a multiplexer work?
        """)

        st.write("")

        # --- File uploader ---
        uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'], key="fixed_upload")
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type == 'application/pdf':
                st.success("PDF uploaded successfully!")
                existing_sources = [doc['source'] for doc in st.session_state.metadata]
                if uploaded_file.name not in existing_sources:
                    with st.spinner(f"Processing {uploaded_file.name} for knowledge base..."):
                        chunks = process_pdf_for_rag(uploaded_file)
                        if chunks:
                            add_documents_to_knowledge_base(chunks, uploaded_file.name)
                            st.success(f"Added {uploaded_file.name} to knowledge base! You can now ask questions about this document.")
                else:
                    st.info(f"{uploaded_file.name} already in knowledge base")
            elif file_type.startswith('image/'):
                st.success("Image uploaded successfully!")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # --- Regular chat input and action buttons in one row ---
        user_input = st.text_input("Ask a question", key="chat_input")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    with st.spinner("Cognibot is thinking..."):
                        get_ai_response(user_input)
                    st.rerun()
        with col2:
            if st.button("Generate Diagram"):
                fig = generate_diagram()
                st.pyplot(fig)
        with col3:
            if st.button("Export PDF Summary"):
                export_to_pdf(st.session_state.chat_history)
                with open("chat_summary.pdf", "rb") as f:
                    st.download_button("Download PDF", f, "chat_summary.pdf")

    elif section == "Chat History":
        st.markdown("### 📜 Chat History Management")
        if not st.session_state.chat_history:
            st.info("No chat history yet. Start a conversation to see it here!")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", len(st.session_state.chat_history))
            with col2:
                total_words = sum(len(entry['user_message'].split()) + len(entry['ai_response'].split()) 
                                for entry in st.session_state.chat_history)
                st.metric("Total Words", total_words)
            with col3:
                rag_used = sum(1 for entry in st.session_state.chat_history if entry['relevant_chunks'])
                st.metric("RAG Used", rag_used)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export Chat History"):
                    export_text = export_chat_history()
                    st.download_button(
                        label="Download as TXT",
                        data=export_text,
                        file_name=f"cognibot_chat_{st.session_state.current_session_id}.txt",
                        mime="text/plain"
                    )
            with col2:
                if st.button("🗑️ Clear History"):
                    clear_chat_history()
                    st.success("Chat history cleared!")
                    st.rerun()
            st.markdown("### 📋 Detailed History")
            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Conversation {len(st.session_state.chat_history) - i + 1} - {entry['timestamp']}"):
                    st.write("**User:**")
                    st.write(entry['user_message'])
                    st.write("**Cognibot:**")
                    st.write(entry['ai_response'])
                    if entry['relevant_chunks']:
                        st.write("**Document Sources Used:**")
                        for chunk in entry['relevant_chunks']:
                            st.write(f"- {chunk['source']} (Similarity: {chunk['similarity']:.2f})")
    elif section == "Knowledge Base":
        st.markdown("### 📚 Knowledge Base Management")
        if not st.session_state.metadata:
            st.info("No documents in knowledge base. Upload PDFs to add them!")
        else:
            sources = list(set([doc['source'] for doc in st.session_state.metadata]))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(sources))
            with col2:
                st.metric("Total Chunks", len(st.session_state.metadata))
            st.markdown("### 📄 Document Details")
            for source in sources:
                source_chunks = [doc for doc in st.session_state.metadata if doc['source'] == source]
                with st.expander(f"📄 {source} ({len(source_chunks)} chunks)"):
                    for i, chunk in enumerate(source_chunks[:3]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                        st.write("---")
                    if len(source_chunks) > 3:
                        st.write(f"... and {len(source_chunks) - 3} more chunks")
    elif section == "Settings":
        st.markdown("### ⚙️ Settings")
        st.markdown("#### 💬 Chat Settings")
        if st.button("🆕 Start New Session"):
            clear_chat_history()
            st.success("New session started!")
            st.rerun()
        st.markdown("#### 🗄️ Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History Only"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        with col2:
            if st.button("Clear Knowledge Base Only"):
                st.session_state.metadata = []
                st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)
                st.success("Knowledge base cleared!")
        if st.button("🗑️ Clear Everything", type="primary"):
            st.session_state.chat_history = []
            st.session_state.metadata = []
            st.session_state.faiss_index = faiss.IndexFlatL2(EMBED_DIM)
            clear_chat_history()
            st.success("Everything cleared!")
            st.rerun()

if __name__ == "__main__":
    main()