import os
import uvicorn
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
   [ 
      ("system", system_template),
      ("user", "{text}")
   ] 
)

chain = prompt_template | llm | parser

app = FastAPI(
   title="Simple Translator Demo",
   version="1.0",
   description="A simple API server using LangChain's Runnable Interfaces"
)

add_routes(
   app,
   chain,
   path="/chain"
)

if __name__ == "__main__":
   uvicorn.run(app, host="localhost", port=8000)