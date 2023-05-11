
import requests
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from bs4 import BeautifulSoup as bs


_ = load_dotenv(find_dotenv())


def number_to_words(num: int = 21023):
    url = "https://number-conversion-service.p.rapidapi.com/webservicesserver/NumberConversion.wso"

    headers = {
        "content-type": "application/xml",
        "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
        "X-RapidAPI-Host": "number-conversion-service.p.rapidapi.com",
    }
    
    payload = f"""<?xml version='1.0' encoding='utf-8'?>
    <soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/'>
      <soap:Body>
        <NumberToWords xmlns='http://www.dataaccess.com/webservicesserver/'>
          <ubiNum>{num}</ubiNum>
        </NumberToWords>
      </soap:Body>
    </soap:Envelope>"""
    response = requests.post(url, data=payload, headers=headers)
    soup = bs(response.text, features="lxml")
    result = soup.find("m:numbertowordsresult").text
    return pd.DataFrame({"query": num, "result": result}, index=[0])

number_to_words(481)

