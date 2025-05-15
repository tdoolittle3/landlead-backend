from operator import itemgetter
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, auth
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from langchain import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
import os
import gc

import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Customize the log message format
)

SQL_PREFIX = """You are an agent designed to interact with a Postgres SQL database.
Given an input question, create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware (Adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins or restrict as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Step 1: Initialize the LLM and ensure the environment variable is set
def initialize_llm():
    try:
        # Make sure the OpenAI model is correctly initialized
        logging.info("Initializing ChatOpenAI model")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Change to gpt-4 or another model if needed
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {str(e)}")
        raise

# Step 2: Initialize the database connection with better error handling
def initialize_db_connection():
    try:
        # Ensure that the DATABASE_URL environment variable is set
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set")

        # Create a connection to the PostgreSQL database with connection pooling
        logging.info("Connecting to the database")
        engine = create_engine(
            db_url,
            pool_size=10,          # Set pool size for performance
            max_overflow=20,       # Handle overflow connections
        )
        db = SQLDatabase(engine)
        return db
    except Exception as e:
        logging.error(f"Failed to connect to the database: {str(e)}")
        raise

# Step 3: Create and cache usable table names and metadata
def get_table_names(db):
    try:
        logging.info("Fetching usable table names")
        table_names = db.get_usable_table_names()
        if not table_names:
            raise ValueError("No usable tables found in the database")
        logging.info(f"Usable tables: {table_names}")
        return table_names
    except Exception as e:
        logging.error(f"Failed to fetch table names: {str(e)}")
        raise


def strip_markdown_formatting(sql_query: str) -> str:
    # Strip out the markdown SQL block
    return sql_query.strip("```sql").strip("```").strip()

def create_postgis_prompt_template(example_geospatial_functions=None):
    # Default list of PostGIS functions if none provided
    if example_geospatial_functions is None:
        example_geospatial_functions = ["ST_Intersects", "ST_Distance", "ST_Within", "ST_Buffer"]

    # Construct the template
    custom_template = (
        "You are an AI assistant specialized in spatial databases (PostGIS) with access to the following tables: {table_info}. "
        "Each table may contain spatial data such as geometries (e.g., points, polygons). Use appropriate geospatial functions, "
        f"like those in the PostGIS extension, when needed, such as: {', '.join(example_geospatial_functions)}. "
        "\n\nYou should consider spatial relationships, distances, and intersections if the query implies spatial data."
        "\n\nBased on the user's natural language request, generate an efficient SQL query that handles both traditional "
        "and spatial data, if relevant. Here's the request: {input}."
        "\n\nEnsure the query uses correct PostGIS functions when dealing with spatial data and includes conditions that are optimized "
        "for geospatial indexing, where applicable. Return the SQL only, exactly what is intended for the user to run, NOTHING ELSE VALID SQL ONLY.  Top k is {top_k}."
    )

    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["input", "top_k", "table_info"],
        template=custom_template
    )
    return prompt_template

# Answer Prompt Template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# OpenAI LLM and SQL database initialization
llm = initialize_llm()
db = initialize_db_connection()  # SQLDatabase(engine) - PostgreSQL URI from environment variables
print("DB Dialect is " + db.dialect)

# Create PostGIS-specific prompt template
postgis_prompt_template = create_postgis_prompt_template()

# SQL Query generation and execution tools
write_query = create_sql_query_chain(llm, db, prompt=postgis_prompt_template)
execute_query = QuerySQLDataBaseTool(db=db)

# Custom logging class for different stages
class LogStep(Runnable):
    def invoke(self, input_data):
        logging.info(f"Logging step data: {input_data}")
        return input_data

# Chain with logging at each step
db_chain = (
    RunnablePassthrough.assign(query=write_query)
    .assign(
        question=lambda input_data: input_data["input"],  # Extracting user's natural language query as 'question'
        result=itemgetter("query") | llm
    | LogStep()  # Logs the LLM response
    | StrOutputParser()
    )
)
# Data model for the query request
class NaturalLanguageQueryRequest(BaseModel):
    natural_language_query: str


def process_query(natural_language_query):
    try:
        logging.info(f"Processing query: {natural_language_query}")

        # Prepare the necessary information
        table_names = db.get_usable_table_names()  # Ensure this is efficient
        example_geospatial_functions = "ST_Intersects, ST_Distance, ST_Within, ST_Buffer"

        # Optimize call to write_query
        query_result = write_query.invoke({"question": natural_language_query})
        logging.info(f"Got query response: {query_result[:500]}...")  # Log only part of the response if it's large

        # Optimize call to execute_query
        execute_result = execute_query.invoke({"query": strip_markdown_formatting(query_result)})
        logging.info(f"Got execute response: {execute_result[:500]}...")  # Log only part of the response if it's large


        prompt_input = {"question": "User question", "query": query_result, "result": execute_result}
        formatted_prompt = answer_prompt.format(**prompt_input)
        logging.info(f"Generated prompt: {formatted_prompt[:500]}...")  # Log only part of the prompt if it's large
        
        del query_result

        del execute_result

        # Optimize call to llm
        llm_response = llm.invoke(formatted_prompt)
        #logging.info(f"LLM response: {llm_response}...")  # Log only part of the response if it's large

        del formatted_prompt

        # Parse the output
        parsed_output = StrOutputParser().invoke(llm_response)
        logging.info(f"Parsed final output: {parsed_output[:500]}...")  # Log only part of the parsed output if it's large

        return parsed_output

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return "An error occurred while processing your request."

    finally:
        # Optionally force garbage collection
        gc.collect()

# Firebase Authentication Dependency
security = HTTPBearer()
async def firebase_auth_dep(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        # Verify the token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

# Helper function to track and log using Langsmith (optional)
async def run_query_with_logging(natural_language_query: str):
    try:
        # Run the SQL query generation process
        sql_result = process_query(natural_language_query) #db_chain.invoke({"question" : natural_language_query })

        return sql_result

    except Exception as e:
        # Log errors (both locally and to Langsmith if enabled)
        logging.error(f"Error processing query: {e}")

        raise e

# POST route to process natural language queries
@app.post("/query")
async def query_db(data: NaturalLanguageQueryRequest, user_data: dict = Depends(firebase_auth_dep)):
    logging.info(f"Received query: {data.natural_language_query}")
    
    try:
        # Process the natural language query into SQL and execute
        result = await run_query_with_logging(data.natural_language_query)
        return {"result": result}
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

# Health check endpoint
@app.get("/health")
async def health_check():
    logging.info("health check hit")
    return {"status": "ok"}

