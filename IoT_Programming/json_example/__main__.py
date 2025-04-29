'''
Created on 2025/04/29

@author: sin
'''
import json

# Example dictionary
data = {
    "name": "John Doe",
    "age": 30,
    "is_employee": True,
    "skills": ["Python", "JavaScript", "SQL"],
    "address": {
        "city": "New York",
        "state": "NY",
        "zip": "10001"
    }
}

# Specify the file name
file_name = "data.json"

# Write the dictionary to the JSON file
with open(file_name, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"The dictionary has been written to {file_name}")

# if __name__ == '__main__':
#     pass

# Specify the file name
#file_name = "output.json"

# Read the JSON file and load it into a dictionary
with open(file_name, "r") as json_file:
    another = json.load(json_file)

# Print the dictionary
print("The content of the JSON file loaded in another is:")
print(another)