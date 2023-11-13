import os 
os.environ["OPENAI_API_KEY"] = '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'

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('/Users/tarakram/Documents/Chatbot/data').load_data()
print(documents)

index = GPTVectorStoreIndex.from_documents(documents)