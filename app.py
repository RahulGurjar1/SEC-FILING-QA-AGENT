import os
import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from scripts.query_processor import parse_query

load_dotenv()

class FinancialQASystem:
    def __init__(self, tickers):
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.tickers = tickers

        @st.cache_resource
        def get_embedding_model():
            return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        @st.cache_resource
        def get_chroma_collection():
            client = chromadb.PersistentClient(path="chroma_db")
            return client.get_collection(name="sec_filings")

        self.embedding_model = get_embedding_model()
        self.chroma_collection = get_chroma_collection()

        genai.configure(api_key=google_api_key)
        self.llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def retrieve_context(self, query_text):
        parsed = parse_query(query_text)

        query_args = {
            "query_texts": [query_text],
            "n_results": 15
        }

        if parsed.get('tickers'):
            metadata_filter = {
                'ticker': {
                    '$in': parsed['tickers']
                }
            }
            query_args['where'] = metadata_filter

        results = self.chroma_collection.query(**query_args)

        context_chunks = results.get('documents', [[]])[0]
        metadata_list = results.get('metadatas', [[]])[0]

        prompt_context = ""
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadata_list)):
            chunk_id = meta.get('chunk_id', 'N/A') 
            
            source = f"[{meta.get('ticker')}-{meta.get('filing_date')}-{meta.get('filing_type')}-{chunk_id}]"
            prompt_context += f"{source}: {chunk}\n\n"

        return prompt_context

    def generate_answer(self, user_question, context):
        prompt = f"""
        You are a financial research assistant.
        Given these source snippets with metadata, answer:
        {user_question}

        Sources:
        {context}

        Rules:
        1. Base your answer ONLY on the provided sources.
        2. For each fact you use, cite the source in the format: [ticker-date-form-chunkID].
        3. If answer is not in context, say: "I cannot answer this based on the provided context."
        """

        response = self.llm_model.generate_content(prompt)
        return response.text


# Streamlit Interface Setup for the Financial QA System
st.set_page_config(layout="wide")
st.title("SEC Filings QA Interface")
st.markdown("A question-answering system that analyzes SEC filings to answer complex financial research questions.")

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BRK-B', 'JNJ', 'PFE', 'AMZN', 'WMT', 'KO', 'XOM']
qa_system = FinancialQASystem(tickers=TICKERS)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Your Question")
    question = st.text_area("Please enter your question below:", height=150, placeholder="e.g., What are the primary revenue drivers for major technology companies, and how have they evolved?")

with col2:
    st.header("Answer")
    if st.button("Get Answer", type="primary"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing query and generating answer..."):
                context = qa_system.retrieve_context(question)
                if not context:
                    st.error("No relevant context found. Try a different query.")
                else:
                    answer = qa_system.generate_answer(question, context)
                    st.markdown(answer)
