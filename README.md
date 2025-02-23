# NLP Minor Project

## Group Member
- Phan Lạc Hưng - 22BI13186
- Nguyễn Anh Đức - 22BI13089
- Nguyễn Hoàng Duy - 22BI13122
- Cao Thái Hà - 22BI13136
- Đỗ Lê Hoàng Việt - 22BI13463
- Nguyễn Hà Trung - 22BI13436

## Overview
This project aims to build a local chatbot using Retrieval-Augmented Generation (RAG) with the LangChain framework. It processes data from local PDF files and stores vector embeddings in Pinecone for retrieval. The chatbot uses "llama2" for generating responses and "nomic-embed-text" for embeddings. A Streamlit interface allows users to upload PDFs, process data, and interact with the chatbot.

## Deployment

### Preparation
- Set up a virtual environment using the command: `python -m venv your_environment_name`
- Active the environment:
 - On Windows: `your_environment_name\Scripts\activate`
 - On macOS/Linux: `source your_environment_name/bin/activate`
- Run `pip install -r requirements.txt` to install all dependencies

### Install Ollama
- Access: https://ollama.com/download
- Choose the version that matches your operating system
- To check if Ollama is installed successfully, open CMD and type:`ollama`
- To download the models, type`ollama run nomic-embed-text` and `ollama run llama2`

### Prepare vector database
- Access: https://www.pinecone.io/
- Create an account
- Copy your API key and set it in the ".env" file

### Run the application
- Navigate to the project directory and run:
```python 
streamlit run main.py
```
- Open the UI and process the data first. You can either:
 - Use our provided sample PDF file.
 - Upload a PDF file from your computer.
- Choose "llama2" model since "nomic-embed-text" does not support generation.
- After testing, you can delete the index to free up Pinecone memory or browse another file.

### References
- Streamlit Documentation: https://docs.streamlit.io/
- LangChain: https://python.langchain.com/docs/introduction/
- Pinecone Docs: https://docs.pinecone.io/guides/get-started/overview