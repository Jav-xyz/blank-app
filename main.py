from yfinance import download
from constants import companies
for company in companies:
    try:
        df = download(company, start="2002-01-01", end="2024-01-01", period="1mo")
        if (df is None):
            continue
        print('the data was fetch properly')
        df.to_csv(f"./stocks/{company}.csv")
    except:
        print(f"An error occured couldn't fetch company: {company}")
        pass
