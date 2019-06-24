#coding: UTF-8
import requests
from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

temp = round(sense.get_temperature(), 1)

# LINE Notify
# https://notify-bot.line.me/ja/
#
# LINE Notify API Document
# https://notify-bot.line.me/doc/ja/
#
# Sample w/ Python
# https://blue-black.ink/?p=3935


url_notifyapi = "https://notify-api.line.me/api/notify"

# IoT_Dev_Prog
token = "C8ehzq5NuFNre6HwwD31A3mIxhN7iElDsAahP3nNHO6"
auth_headers = {"Authorization" : "Bearer "+ token}

payload = {"message" :  "曇りときどき雨"}
opt_files = {"imageFile": open("cloudyandsunny.png", "rb")} #バイナリで画像ファイルを開きます。対応している形式はPNG/JPEGです。

r = requests.post(url_notifyapi ,headers = auth_headers, params=payload, files=opt_files)

payload["message"] = "Temp@pi is " + str(temp) + "deg."
r = requests.post(url_notifyapi ,headers = auth_headers, params=payload)

