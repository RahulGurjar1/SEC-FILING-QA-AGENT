# Financial SEC Filings Q&A System

This project is a Question-Answering system that uses a Retrieval-Augmented Generation (RAG) architecture to answer complex financial questions based on the content of public companies' SEC filings.

## Overview

The system operates in two main phases:

1.  **Data Pipeline (Offline):** A series of scripts downloads, parses, and indexes SEC filings. It specifically extracts the "Risk Factors" and "Management's Discussion and Analysis (MD&A)" sections, chunks them, and stores them as vector embeddings in a ChromaDB database.
2.  **QA Application (Online):** A Streamlit web application allows you to ask questions in natural language. The app retrieves the most relevant text chunks from the database and uses the Google Gemini LLM to generate a concise, accurate answer, complete with citations to the source documents.

## Tech Stack & Rationale

*   **Python**: Core programming language.
*   **Streamlit**: Powers the interactive web UI.
    *   *Why?* Enables rapid development of data-centric applications with simple Python scripts.
*   **Sentence-Transformers**: Generates vector embeddings for semantic search.
    *   *Why?* The `all-MiniLM-L6-v2` model provides a fantastic balance of high performance and small size, making it ideal for running locally.
*   **ChromaDB**: The vector database for storing and retrieving text chunks.
    *   *Why?* It is an open-source, lightweight, and persistent vector store that is extremely easy to set up and use for local development.
*   **Google Gemini**: The Large Language Model used for generating answers.
    *   *Why?* A powerful and capable LLM that can follow strict instructions to synthesize information from provided context.
*   **LangChain**: Used for its robust text-splitting utilities.
    *   *Why?* Provides standardized, battle-tested components that are essential for building reliable LLM applications.
*   **BeautifulSoup & Regex**: For parsing HTML and extracting specific sections.
    *   *Why?* A practical and effective combination for targeted information extraction from semi-structured HTML documents.
*   **Pandas & PyArrow**: For data manipulation and storage in the Parquet format.
    *   *Why?* Parquet is a highly efficient columnar storage format that is perfect for tabular data like our processed text chunks.


## Prerequisites

*   Python 3.8+
*   API keys for:
    *   Google AI (for Gemini)
    *   SEC-API.io

## Process to run the Project
### 1. Clone the Repository
git clone https://github.com/RahulGurjar1/SEC-FILING-QA-AGENT.git
cd sec_filings_qa

### 2. Install Dependencies: It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Ubantu (check out if you are working on a different OS)
pip install -r requirements.txt
```
### 3. Set Up Environment Variables
Create a file named .env in the root of the project directory and add your API keys:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
SEC_API_KEY="your_sec_api_io_key_here"
USER_AGENT_EMAIL="your_email@example.com" # Required by SEC for API access
```
### 4. Run the Data Pipeline
Execute the following scripts in order to build the vector database. This only needs to be done once.

1. Download filings from sec-api.io
```bash
python3 scripts/downloader.py
```
2. Parse HTML files and create chunks
```bash
python3 scripts/parser.py
```
3. Create embeddings and index them in ChromaDB
```bash
python3 scripts/indexer.py
```
### 5. Run the Streamlit Application
```bash
streamlit run app.py
```
You can now access the Q&A interface in your web browser at localhost url.