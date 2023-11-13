import openai as ai

ai.api_key = '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'

def chat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = ai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message

def chat_with_gpt():
    print("Chatbot: Hello! Ask me anything or type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        prompt = f"You: {user_input}\nChatbot:"
        response = chat_with_chatgpt(prompt)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat_with_gpt()
