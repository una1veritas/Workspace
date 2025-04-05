'''
Created on 2025/04/04

@author: sin
'''

import asyncio
import websockets
import pyttsx3

async def handler(websocket):
    print(f"New connection from {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            text_to_speech(message, 'com.apple.eloquence.en-GB.Shelley')
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("Server started on ws://0.0.0.0:8765")
    await server.wait_closed()

def text_to_speech(text, voice_id : str = 'com.apple.eloquence.ja-JP.Eddy'):
    if len(text) == 0 :
        return
    # Initialize the TTS engine
    #engine = pyttsx3.init()
    
    if voice_id != None :
        # Set the voice by ID
        engine.setProperty('voice', voice_id)  # Change index to select different voices
    
    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Add text to be spoken
    engine.say(text)

    # Run the speech engine
    engine.runAndWait()

engine = pyttsx3.init()
asyncio.run(main())
