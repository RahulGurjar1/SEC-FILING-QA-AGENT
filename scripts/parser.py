import os
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm


# initial configuration
DATA_DIR = 'data';
CHUNK_SIZE = 1000;
CHUNK_OVERLAP = 200;
SECTION_PATTERNS = {
    'risk_factors': {
        'start': [r'item\s*1a\s*\.\s*risk\s*factors'],
        'end': [r'item\s*1b\s*\.\s*unresolved\s*staff\s*comments']
    },
    'mda': {
        'start': [r'item\s*7\s*\.\s*management\'s\s*discussion\s*and\s*analysis'],
        'end': [r'item\s*7a\s*\.\s*quantitative\s*and\s*qualitative\s*disclosures']
    }
};

# helper functions
def find_section_positions(text, patterns):
    start_pos = -1;
    end_pos = len(text);
    for pattern in patterns['start']:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL);
        if match:
            start_pos = match.start();
            break;
    if start_pos == -1:
        return -1, -1;

    for pattern in patterns['end']:
        match = re.search(pattern, text[start_pos:], re.IGNORECASE | re.DOTALL);
        if match:
            end_pos = start_pos + match.start();
            break;
    return start_pos, end_pos;

def extract_section_text(html_content, patterns):
    soup = BeautifulSoup(html_content, 'lxml');
    full_text = ' '.join(p.get_text() for p in soup.find_all(['p', 'div', 'span', 'font']));
    start, end = find_section_positions(full_text, patterns);
    if start != -1:
        return full_text[start:end]
    return None

# main logic
def process_all_filings():
    all_chunks=[];
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    );
    file_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.html'):
                file_paths.append(os.path.join(root, file))

    print(f"Found {len(file_paths)} HTML files to process.")
    
    for file_path in tqdm(file_paths, desc="Processing Filings"):
        try:
            parts = file_path.split(os.sep)
            ticker, filing_type, file_name = parts[-3], parts[-2], parts[-1]
            date = file_name.split('_')[0]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            for section_name, patterns in SECTION_PATTERNS.items():
                section_text = extract_section_text(html_content, patterns)
                
                if section_text:
                    chunks = text_splitter.split_text(section_text)
                    
                    for i, chunk_text in enumerate(chunks):
                        chunk_data = {
                            'ticker': ticker,
                            'filing_type': filing_type,
                            'filing_date': date,
                            'section': section_name,
                            'chunk_id': f"{ticker}_{date}_{filing_type}_{section_name}_{i}",
                            'text': chunk_text,
                            'source_file': file_path
                        }
                        all_chunks.append(chunk_data)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    df = pd.DataFrame(all_chunks)
    
    output_path = os.path.join(DATA_DIR, 'processed_chunks.parquet')
    df.to_parquet(output_path)
    
    print(f"\nProcessing complete. Saved {len(df)} chunks to {output_path}")

if __name__ == '__main__':
    process_all_filings();