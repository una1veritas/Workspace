'''
Created on 2025/04/05

@author: sin
'''

from openai import OpenAI

# Make sure you have your API key stored in an environment variable or replace 'YOUR_API_KEY' with your actual API key.
MODEL="gpt-4o"
client = OpenAI(api_key='api key here')

completion = client.chat.completions.create(  
    model=MODEL,  
    messages=[
        {"role": "system", "content": "You are a helpful assistant that helps me with my math homework!"},
        {"role": "user", "content": "Hello! Could you solve 20 x 5?"}
        ]
    )
print("Assistant: " + completion.choices[0].message.content)
