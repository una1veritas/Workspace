# -*- coding: utf-8 -*-
import requests
import json
import datetime
import sys

params = dict()
params['API_KEY'] = "3efcd2e4fecf61bd823bcc55b23b0aab"
params['CITY_NAME']= "Fukuoka-shi"

if ( len(sys.argv) > 1 ) :
    params['CITY_NAME']= sys.argv[1]

api = ("http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}&units=metric")
url = api.format(city = params['CITY_NAME'], key = params['API_KEY'])
resp = requests.get(url)
data_dict = json.loads(resp.text)
print(data_dict)

print()
print(u"天気 "+data_dict["weather"][0]["main"])
print(u"最高気温 {0} °C, 最低気温 {1} °C".format(data_dict["main"]["temp_max"], 
                                         data_dict["main"]["temp_min"]))
utimestamp = datetime.datetime.fromtimestamp(data_dict["sys"]["sunset"])
sunset = utimestamp.strftime("%Y/%m/%d %H:%M:%S")
print(u"日没 " + sunset)
