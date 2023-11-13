import openai

# Set your OpenAI GPT-3 API key
api_key = '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'
openai.api_key = api_key

def generate_response(prompt):
    # Specify the engine (text-davinci-003 is recommended for GPT-3.5)
    engine = "text-davinci-003"

    # Set the parameters for the completion
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
    )

    # Extract and return the generated text
    return response['choices'][0]['text']

# Example usage
user_input = input("You: ")

while user_input.lower() != 'exit':
    # You can customize the prompt based on your application needs
    prompt = f"You: {user_input}\nBot:"
    response = generate_response(prompt)
    
    print(f"Bot: {response}")

    user_input = input("You: ")
