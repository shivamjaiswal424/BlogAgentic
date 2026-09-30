import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


class GroqLLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        try:
            os.environ["GROQ_API_KEY"]=self.groq_api_key=os.getenv("GROQ_API_KEY")
            llm=ChatGroq(api_key=self.groq_api_key,model="openai/gpt-oss-20b")
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception : {e}")

    def get_research_llm(self):
        try:
            os.environ["GROQ_API_KEY"]=self.groq_api_key=os.getenv("GROQ_API_KEY")
            llm=ChatGroq(api_key=self.groq_api_key,model="groq/compound-mini")
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception : {e}")

