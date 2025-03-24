import requests
import sys
from pprint import pprint
port = str(sys.argv[1]) if len(sys.argv) >= 2 else '5004'

url_wiki = f"http://127.0.0.1:{port}/queries"
query_list = ['Apple', "Microsoft", "Google"]
topk = 3
response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
if response.status_code == 200:
    pprint(response.json())
else:
    print(response.text)
