import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os
import tempfile

# --- UI Configuration ---
st.set_page_config(page_title="Query Your PDF Notes", layout="wide")
st.title("📄 Query Your PDF Notes")

# --- Hugging Face API Token ---
# It's recommended to set this as an environment variable or Streamlit secret
# For local testing, you can uncomment the line below and paste your token
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Helper Function to Create the RAG Chain ---
def create_rag_chain(pdf_file):
    """Creates a Retrieval-Augmented Generation chain from a PDF file."""
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # 1. Load the document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # 3. Create embeddings (CPU-friendly)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 4. Create vector database
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # 5. Initialize the LLM from Hugging Face Hub
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
        )

        # 6. Create the RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
        )
        
        # Clean up the temporary file
        os.remove(tmp_file_path)

        return qa_chain

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        # Clean up in case of error
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

# Use session_state to store the chain and avoid re-processing
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
                result = st.session_state.qa_chain({"query": query})
                st.write("### Answer")
                st.write(result["result"])
            except Exception as e:
                st.error(f"Failed to get an answer: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Powered by LangChain, Streamlit, and Hugging Face.")
