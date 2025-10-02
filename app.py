import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW: Import from langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import tempfile

# --- UI Configuration ---
st.set_page_config(page_title="Query Your PDF Notes", layout="wide")
st.title("📄 Query Your PDF Notes")

# --- Helper Function to Create the RAG Chain ---
def create_rag_chain(pdf_file):
    """Creates a Retrieval-Augmented Generation chain from a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # UPDATED: No change in logic, just the import source is new
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # UPDATED: Use HuggingFaceEndpoint instead of HuggingFaceHub
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.7,
            task="conversational",
            max_new_tokens=500
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
        )
        
        os.remove(tmp_file_path)
        return qa_chain

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        return None

# --- Streamlit App Logic ---
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Upload your PDF notes using the file uploader.\n"
    "2. Wait for the document to be processed.\n"
    "3. Ask questions about your document in the text box."
)

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        with st.spinner("Processing your PDF... This may take a moment."):
            st.session_state.qa_chain = create_rag_chain(uploaded_file)
            if st.session_state.qa_chain:
                st.success("Document processed successfully! You can now ask questions.")

if st.session_state.qa_chain is not None:
    query = st.text_input("Ask a question about the document:", placeholder="What is the main topic of my notes?")
    if query:
        with st.spinner("Thinking..."):
            try:
                # UPDATED: Use .invoke() instead of calling the chain directly
                result = st.session_state.qa_chain.invoke({"query": query})
                st.write("### Answer")
                st.write(result["result"])
            except Exception as e:
                st.error(f"Failed to get an answer: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Powered by LangChain, Streamlit, and Hugging Face.")
