import openai
import gradio
import time

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a guy who is really good at getting girls"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    
    # Simulate typing by displaying a "Typing..." message
    demo.result = "Typing..."
    
    # Wait for a short duration to simulate typing
    time.sleep(1)
    
    # Get the actual model response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )["choices"][0]["message"]["content"]
    
    # Update the display with the actual response
    demo.result = response
    
    return response

demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", live=True, title="How to Rizz")
demo.launch(share=True)
