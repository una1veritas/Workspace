import requests
import json
import urllib.parse

api="https://www.quandl.com/api/v3/datatables/FXCM/H1?date=2019-04-03&symbol={0}&api_key=o-n5ZDbb-ewagY5j8VYf"
qcode = urllib.parse.quote("XNAS/AAPL")
resp = requests.get(api.format(qcode))
data_dict = json.loads(resp.text)
print(data_dict)
