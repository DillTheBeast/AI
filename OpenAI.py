import openai as ai
import time
import sys


#1
# # Set your OpenAI GPT-3 API key
# api_key = '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'
# ai.api_key = api_key  # Set the API key

# # User input
# user_message = "Write 2 sentences about dogs"
# print(f"User: {user_message}")

# # Initial message
# messages = [{"role": "user", "content": user_message}]
# print(f"Sending initial message to GPT-3: {messages}")

# # Make API call
# complete = ai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=messages
# )

# # Simulate typing and print the content of the completed message
# response_content = complete.choices[0].message['content']
# print("GPT-3 is typing: ", end="", flush=True)
# time.sleep(1)  # Initial delay
# for char in response_content:
#     print(char, end="", flush=True)
#     time.sleep(0.05)  # Adjust the delay to control typing speed
# print()  # Move to the next line after completion

#2
ai.api_key = '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'
messages = []

systemMSG = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": systemMSG})

print("Your assistant is ready")

while input != "quit()":
    message = input("")
    if message == "quit":
        sys.exit(1)
    messages.append({"role": "user", "content": message})
    response = ai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "user", "content": reply})
    print("\n" + reply + "\n")