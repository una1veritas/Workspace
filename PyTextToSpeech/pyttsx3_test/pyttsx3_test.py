'''
Created on 2025/03/12

@author: sin
'''

import pyttsx3

def text_to_speech(text, voice = None):
    engine = pyttsx3.init()
    if voice != None :
        engine.setProperty('voice', voice)
    engine.say(text)
    engine.runAndWait()

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        print(f"ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}, Gender: {voice.gender}, Age: {voice.age}")

if __name__ == "__main__":
    voice = 'com.apple.eloquence.ja-JP.Eddy'
    text = "なかのひとなど、いない！"
    text_to_speech(text, voice)
    voice = 'com.apple.eloquence.en-GB.Grandpa'
    text = "This royal throne of kings, this sceptred isle!" 
    text_to_speech(text, voice)
    voice = 'com.apple.voice.compact.it-IT.Alice'
    text = "Lasciate ogne speranza, voi ch'intrate."
    text_to_speech(text, voice)
    #list_voices()