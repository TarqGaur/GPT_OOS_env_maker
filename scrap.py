import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from requests.adapters import HTTPAdapter, Retry
from LLM import addhistory 



def scrap(search):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0 Safari/537.36"
    }

    params = {
        "engine": "google",
        "q": search,   
        "hl": "en",
        "gl": "us",
        "num": "2",     
        "api_key": "d71b08519c184d0b2d7d339e7c0e4dba161f48c0095680ae57419a6f70d3d501"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", [])[:3]

    total_data = []

    for i, result in enumerate(organic_results, start=1):
        url = result.get("link")
        title = result.get("title")
        print(f"\n🔗 Site {i}: {title}\nURL: {url}")
        
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        text = " ".join([p.get_text() for p in soup.find_all("p")])
        total_data.append(text)
    return total_data


