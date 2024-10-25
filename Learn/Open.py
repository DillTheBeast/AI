import os
files = os.listdir(path = "/Users/dillonmaltese/Documents/git/AI/Learn")

for file in files:
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        print(file)