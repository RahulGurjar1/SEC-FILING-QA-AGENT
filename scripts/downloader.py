import os
import time
import requests
from dotenv import load_dotenv

# loading environment variables from .env file in the project
load_dotenv()

# initial configuration
SEC_API_KEY = os.getenv('SEC_API_KEY');
if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY is not set in the environment variables.")

Headers = {
    'Authorization': SEC_API_KEY,
};

QUERY_API_URL = f'https://api.sec-api.io?token={SEC_API_KEY}'
DATA_DIR = 'data';
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BRK-B', 'JNJ', 'PFE', 'AMZN', 'WMT', 'KO', 'XOM']
FILING_TYPES = ['10-K', '10-Q', '8-K']
START_DATE = '2021-01-01'
USER_AGENT_EMAIL = os.getenv('USER_AGENT_EMAIL');
if not USER_AGENT_EMAIL:
    raise ValueError("USER_AGENT_EMAIL is not set in the environment variables.")

# main code
# fetches and saves SEC filings for a list of tickers
def fetch_and_save_filings():
    fetch_headers = {'User-Agent': USER_AGENT_EMAIL};
    print('SEC filing download process started');
    for ticker in TICKERS:
        for filing_type in FILING_TYPES:
            # query = {
            #     "query":{
            #         "query-string":{
            #             "query":f'ticker:{ticker} AND formType:"{filing_type}" AND filedAt:[{START_DATE} TO *]'
            #         }
            #     },
            #     "from":"0",
            #     "size":"50",
            #     "sort":[{"filedAt":{"order":"desc"}}]
            # }
            query = {
                "query": f'ticker:{ticker} AND formType:"{filing_type}" AND filedAt:[{START_DATE} TO *]',
                "from": "0",
                "size": "50",
                "sort": [{ "filedAt": { "order": "desc" }}]
            }

            try:
                # response = requests.post(QUERY_API_URL + '/v1/filings', json=query, headers=Headers);
                response = requests.post(QUERY_API_URL, json=query)
                response.raise_for_status();
                # filings_metadata = response.json()['filings']
                filings_metadata = response.json().get('filings', [])
                if not filings_metadata:
                    print(f"no filings found for {ticker} since {START_DATE}");
                    continue;
                print(f"found {len(filings_metadata)} {filing_type} filings for {ticker}. Download started...");
                for filing in filings_metadata:
                    report_url = filing['linkToFilingDetails'];
                    filed_at = filing['filedAt'].split('T')[0];
                    save_dir = os.path.join(DATA_DIR,ticker, filing_type);
                    os.makedirs(save_dir,exist_ok=True);
                    file_name = f"{filed_at}_{filing_type}.html";
                    file_path= os.path.join(save_dir,file_name);
                    if(os.path.exists(file_path)):
                        print(f"skipping alreay downloaded file: {file_path}");
                        continue
                    filing_response = requests.get(report_url, headers= fetch_headers)
                    filing_response.raise_for_status();
                    with open(file_path,'w', encoding='utf-8') as f:
                        f.write(filing_response.text);
                    print(f" -> Saved {file_path}");
                    time.sleep(1);
            except requests.exceptions.RequestException as err:
                print(f"error fetching data for {ticker} {filing_type}:{err}");
            except KeyError:
                print(f"couldn't parse response for {ticker} {filing_type}: Response: {response.text}")
    print("download process completed.")            

if __name__== '__main__':
    fetch_and_save_filings();


