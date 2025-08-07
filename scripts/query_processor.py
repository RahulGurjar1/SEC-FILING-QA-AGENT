import re
from typing import List, Dict

FORM_TYPES = ['10-K', '10-Q', '8-K', 'DEF 14A', 'Form 4', 'Form 3', 'Form 5']
KNOWN_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BRK-B', 'JNJ', 'PFE', 'AMZN', 'WMT', 'KO', 'XOM']


def parse_query(query: str) -> Dict:
    query = query.strip()
    words = query.split()

    tickers = [w for w in words if w in KNOWN_TICKERS or (w.isupper() and len(w) <= 5)]
    tickers = list(set([t for t in tickers if t in KNOWN_TICKERS]))

    years = re.findall(r"\b(20[0-2][0-9])\b", query)
    forms = [form for form in FORM_TYPES if form.lower() in query.lower()]

    return {
        "tickers": tickers,
        "years": years,
        "forms": forms
    }

if __name__ == '__main__':
    test_query = "Compare AAPL and MSFT 10-K filings from 2022 and 2023"
    parsed = parse_query(test_query)
    print(parsed)
