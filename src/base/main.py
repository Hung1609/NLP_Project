import streamlit as st
import logging
import os
import tempfile
import shutil
import ollama
import warnings
import torch
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import Tuple, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Set up
st.set_page_config(
    page_title="NLP Minor Project RAG Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

st.cache_resource(show_spinner=True)
def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

def create_vector_db(file_upload):
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    logger.info(f"Document split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = PineconeStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    logger.info("Vector DB created in Pinecone")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def check_pinecone_index(index_name: str, dimension: int = 768) -> None:
    logger.info(f"Checking if Pinecone index '{index_name}' exists...")
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        logger.info(f"Index '{index_name}' does not exist. Creating new index...")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        logger.info(f"Index '{index_name}' created successfully.")
    else:
        logger.info(f"Index '{index_name}' already exists.")

def process_question(question: str, vector_db: Pinecone, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    # Initialize Ollama
    llm = ChatOllama(model=selected_model)
    # Prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. 
        Your task is to generate 2 different versions of the given question to improve document retrieval:
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    RAG_template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(RAG_template)
    chain = ({"context": retriever, "question": RunnablePassthrough()}|prompt|llm|StrOutputParser())

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def delete_pinecone_index(index_name: str) -> None:
    logger.info(f"Deleting Pinecone index '{index_name}'...")
    try:
        pc.delete_index(index_name)
        # Clear session state
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
            
        st.success("Index and temporary files deleted successfully.")
        logger.info(f"Index '{index_name}' deleted successfully.")
    except Exception as e:
        st.error(f"Error deleting index: {str(e)}")
        logger.error(f"Error deleting index '{index_name}': {e}")

def main() -> None:
    if "initialized" not in st.session_state:
        check_pinecone_index(INDEX_NAME)
        # Get models
        models_info = ollama.list()
        st.session_state["available_models"] = extract_model_names(models_info)
        st.session_state["initialized"] = True

    # Define layout    
    st.subheader("NLP Minor Project RAG Assistant", divider="gray", anchor=False)
    col1, col2 = st.columns([1, 3])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "question_cache" not in st.session_state:
        st.session_state["question_cache"] = {}

    # Model selection
    if st.session_state["available_models"]:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system", 
            st.session_state["available_models"],
            key="model_select"
        )

    use_sample = col1.toggle(
        "Use sample PDF (Sample data)", 
        key="sample_checkbox"
    )
    
    # Clear vector DB when switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            try:
                pc.delete_index(INDEX_NAME)
                logger.info(f"Pinecone index '{INDEX_NAME}' deleted successfully.")
            except Exception as e:
                logger.exception(f"Error deleting index '{INDEX_NAME}': {e}")
                st.error(f"Error deleting index: {str(e)}")
                
            st.session_state["vector_db"] = None
            if "file_upload" in st.session_state:
                st.session_state.pop("file_upload")

        st.session_state["use_sample"] = use_sample

    if use_sample:
        sample_path = "../documents/all_data.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = PineconeStore.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        index_name=INDEX_NAME,
                    )
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        file_upload = col1.file_uploader(
            "Upload a PDF file", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )

        if file_upload:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    # Store file in session state
                    st.session_state["file_upload"] = file_upload

    # Delete index button
    delete_index = col1.button(
        "Delete index", 
        type="secondary",
        key="delete_button"
    )

    if delete_index:
        delete_pinecone_index()

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            with message_container.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user's message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            if prompt in st.session_state["question_cache"]:
                                response = st.session_state["question_cache"][prompt]
                            else:
                                response = process_question(
                                    prompt, st.session_state["vector_db"], selected_model
                                )
                                st.session_state["question_cache"][prompt] = response
                            
                            st.markdown(response)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                        else:
                            st.warning("Please upload a PDF file first.")

            except Exception as e:
                st.error(e)
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state.get("vector_db") is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")

if __name__ == "__main__":
    main()