import sys
import requests
import json
#sense_hat を使う場合，次の行の # を削除する
#from sense_hat import SenseHat

if 'sense_hat' in sys.modules:
    sense = SenseHat()
    sense.clear()
 
api ="https://api.bitflyer.jp/v1/ticker"
 
resp = requests.get(api)
data_dict = json.loads(resp.text) 
print(data_dict)
curpair = data_dict["product_code"].replace('_','/')
tstamp = data_dict["timestamp"].replace('T', ' ').split('.')[0]
buy = data_dict["best_bid"]
sell = data_dict["best_ask"]
msg = "{0} UTC  {1} BUY {2:,} SELL {3:,}".format(tstamp, curpair, buy, sell)
if 'sense_hat' in sys.modules:
    sense.show_message(msg)
else:
    print(msg)
