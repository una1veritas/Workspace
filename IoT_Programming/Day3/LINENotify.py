#coding: UTF-8
import requests

url_notifyapi = "https://notify-api.line.me/api/notify"
# IoT_Dev_Prog
# C8ehzq5NuFNre6HwwD31A3mIxhN7iElDsAahP3nNHO6
token = "C8ehzq5NuFNre6HwwD31A3mIxhN7iElDsAahP3nNHO6"
auth_headers = {"Authorization" : "Bearer "+ token}

payload = {"message" :  "曇りのち晴れ"}
opt_files = {"imageFile": open("cloudyandsunny.png", "rb")} #バイナリで画像ファイルを開きます。対応している形式はPNG/JPEGです。

r = requests.post(url_notifyapi ,headers = auth_headers, params=payload, files=opt_files)
