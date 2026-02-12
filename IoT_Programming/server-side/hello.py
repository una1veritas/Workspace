#!/usr/bin/env python3

import json

# Specify the file name
file_name = "data.json"

# Read the JSON file and load it into a dictionary
with open(file_name, "r") as json_file:
    data_dict = json.load(json_file)


print("Content-Type: text/html")  # HTTP header
print()  # Blank line to end headers
print("<html><body>")
print("<h1>Hello, Python CGI!</h1>")
if 'temperature' in data_dict :
    print(f"<p>temperature = {data_dict['temperature']}</p>")
if 'humidity' in data_dict :
    print(f"<p>humidity = {data_dict['humidity']}</p>")
print("</body></html>")
