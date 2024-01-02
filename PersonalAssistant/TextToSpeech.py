import os

# The text that you want to convert to speech
mytext = input("Input a word or sentence you would like to hear: ")

# Use osascript to make the Mac speak the text
os.system(f'say "{mytext}"')
