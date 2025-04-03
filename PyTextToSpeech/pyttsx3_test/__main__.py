'''
Created on 2025/03/11

@author: sin

created by github copilot with query 
voice synthesis library in python program
'''
import pyttsx3

def text_to_speech(text, voice_id : str = None):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    if voice_id != None and voice_id in ('help', 'list'):
        voices = engine.getProperty('voices')
        for voice in voices:
            print(f"ID: {voice.id}, Name: {voice.name}")
    elif voice_id != None :
        # Set the voice by ID
        engine.setProperty('voice', voice_id)  # Change index to select different voices
    
    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Add text to be spoken
    engine.say(text)

    # Run the speech engine
    engine.runAndWait()

if __name__ == "__main__":
    text = "Hello, this is a text to speech conversion example using pyttsx3 in Python."
    text_to_speech(text, 'com.apple.eloquence.en-GB.Shelley') 
    text = "こんにちは。わたしは、とおりがかりのただのつうこうにんです。なかのひとは、いません。"
    text_to_speech(text, 'com.apple.eloquence.ja-JP.Eddy') 
    text = "Wenn ist das Nunstück git und Slotermeyer? Ja! Beiherhund das Oder die Flipperwaldt gersput!"
    text_to_speech(text, 'com.apple.eloquence.de-DE.Sandy') 
    