import os
import logging
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# Initial Configuration
DATA_FILE = 'data/processed_chunks.parquet'
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'sec_filings'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 100
USE_GPU = torch.cuda.is_available()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_vector_db(overwrite: bool = False):
    # Validiation of data file
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found at {DATA_FILE}. Run python3 scripts/parser.py first.")
        return

    df = pd.read_parquet(DATA_FILE)
    logger.info(f"Loaded {len(df)} chunks from {DATA_FILE}.")

    # Initialization of embedding model
    device = 'cuda' if USE_GPU else 'cpu'
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}.")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Initialization of ChromaDB client and collection
    client = chromadb.PersistentClient(path=DB_PATH)
    if overwrite and client.exists_collection(name=COLLECTION_NAME):
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Dropped existing collection '{COLLECTION_NAME}'.")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"Using ChromaDB collection '{COLLECTION_NAME}'.")

    failures = 0
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Indexing batches"):
        batch = df.iloc[start:start+BATCH_SIZE]
        ids = batch['chunk_id'].tolist()
        docs = batch['text'].tolist()
        metas = batch.drop(columns=['chunk_id', 'text']).to_dict('records')
        try:
            embeddings = model.encode(docs, show_progress_bar=False).tolist()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metas
            )
        except Exception as e:
            logger.error(f"Batch starting at {start} failed: {e}")
            failures += 1

    logger.info(f"Indexing complete with {failures} failed batches.")
    total = collection.count()
    logger.info(f"Total documents in collection: {total}.")

    # Test Query
    test_query(collection)


def test_query(collection):
    query_text = "What are the main risks related to competition for Apple?"
    filters = {
        '$and': [
            {'ticker': {'$eq': 'AAPL'}},
            {'section': {'$eq': 'risk_factors'}}
        ]
    }
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3,
            where=filters
        )
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        if not docs:
            logger.warning("No results found for test query.")
            return

        logger.info(f"Test query: {query_text}")
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            logger.info(f"Result {i}: {meta['ticker']} {meta['filing_type']} on {meta['filing_date']}")
            logger.info(f"Excerpt: {doc[:200]}...")
    except Exception as e:
        logger.error(f"Test query failed: {e}")


if __name__ == '__main__':
    create_vector_db(overwrite=False)