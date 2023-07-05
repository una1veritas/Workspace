import requests
import json

parameters = {
    "lat" : 33.65268,
    "lon" : 130.67200,
    "key" : "3efcd2e4fecf61bd823bcc55b23b0aab"
}

api = "https://api.openweathermap.org/data/2.5/onecall?lat={0}&lon={1}&appid={2}"

url = api.format(parameters["lat"], parameters["lon"], parameters["key"])
forcast_data = requests.request("GET", url)
forcast_json = json.loads(forcast_data.text)

#forcast_json は python の dict 型．
#forcast_json["hourly"] はリスト型
#見づらいので一つプリント
print(len(forcast_json["hourly"]))
print(forcast_json["hourly"][3])

