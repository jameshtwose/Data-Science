import urllib, json
from urllib.request import urlopen
import pandas as pd


def main():
    url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    response = urlopen(url)
    data = json.loads(response.read())
    print(data)
    url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])
    return url

print(main())

# print(pd.read_csv(main()).head())
