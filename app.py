import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import psycopg2
#from langchain.llms import OpenAI
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#from langchain import LangChain
import logging

class NaturalLanguageQueryRequest(BaseModel):
    natural_language_query: str


cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

security = HTTPBearer()

async def firebase_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        # Verify the token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #"http://localhost:3000"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

DATABASE_URL = os.getenv('DATABASE_URL')
logging.info(DATABASE_URL)
db = SQLDatabase.from_uri(DATABASE_URL)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

@app.get("/health")
async def health_check(user_data: dict = Depends(firebase_auth)):
    logging.info("HEALTH endpoint hit!")
    return {"message": "This is a protected route", "user": user_data}

@app.post("/query")
async def query_db(data: NaturalLanguageQueryRequest, user_data: dict = Depends(firebase_auth)):
    # Initialize LangChain and PostgreSQL connection
    try:
        logging.info("Query endpoint hit")
        if not result:
            raise HTTPException(status_code=404, detail="No results found for the query.")
            
        result = agent_executor.invoke(data.natural_language_query)
        return {"result": result}
    except Exception as e:
        logging.error(f"Error during query execution: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
