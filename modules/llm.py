
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

def get_llm():
    #return  OllamaLLM(model="llama3")
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
    request_timeout=30  

  )
    return llm



