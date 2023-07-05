from flask import Flask, request, jsonify
import time
import os
import textwrap
import openai
import markdown
import jwt
import vecs
import atlassian_jwt_auth 
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine, text, select, delete, Table, MetaData
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from supabase import create_client, Client
from llama_index import ListIndex, SimpleWebPageReader, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import SupabaseVectorStore
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts  import Prompt
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import StructuredOutputParser
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL, DEFAULT_REFINE_PROMPT_TMPL
from llama_index.output_parsers import LangchainOutputParser
from llama_index import ServiceContext 

from init_env import load_env_vars
from auth import authenticate, generate_atlassian_jwt 
from utils import linkify 
from atlassian_requests import process_requests, fetch_tasks

load_env_vars()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DB_CONNECTION_STRING = os.environ.get('DB_CONNECTION_STRING')
CONFLUENCE_EMAIL = os.environ.get('CONFLUENCE_EMAIL')
CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN')
PROJECT_PILOT_API_KEY = os.environ.get("PROJECT_PILOT_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

app = Flask(__name__)
parser = SimpleNodeParser()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

jobstores = {
    'default': SQLAlchemyJobStore(DB_CONNECTION_STRING)
}

def job():
    print('run job')
    # your job here; for instance, calling an endpoint on your Flask server
    # if the job involves making an HTTP request, you can use requests.get('http://localhost:5000/your-endpoint')

scheduler = BackgroundScheduler(daemon=True, jobstores=jobstores)
# Run job every day at a specific time
scheduler.add_job(job, trigger='cron', hour=21, minute=43)
scheduler.start()


@app.route('/initialize', methods=['POST'])
@authenticate
def create_embeddings():
    data = request.json
    key = data['key']
    client_key = data['clientKey']
    base_url = data['baseUrl']
    org_name = base_url.split('https://')[1].split('.')[0]

    response = supabase.table('companies').select('shared_secret').eq('client_key', client_key).execute()
    shared_secret = response.data[0]['shared_secret']

    pages = process_requests("page", org_name, "/wiki/api/v2/pages?body-format=storage", client_key, shared_secret, key)
    blog_posts = process_requests("blog post", org_name, "/wiki/api/v2/blogposts?body-format=storage", client_key, shared_secret, key)
    tasks = fetch_tasks("task", org_name, "/wiki/api/v2/tasks?body-format=storage", client_key, shared_secret, key)
    # spaces = process_requests("space", org_name, "/wiki/api/v2/spaces?body-format=storage", client_key, shared_secret, key)

    documents = pages + blog_posts + tasks

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name=org_name
    )

    engine = create_engine(DB_CONNECTION_STRING)

    # Initialize a metadata object
    metadata = MetaData()

    # Define the table
    table = Table(org_name, metadata, autoload_with=engine, schema='vecs')

    ids_to_delete = []

    # Start a new session
    with engine.connect() as connection:
        # Execute a query
        result = connection.execute(select(table.c.id))

        # Fetch all rows
        ids = result.fetchall()

        for id in ids:
            ids_to_delete.append(id[0])

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    vx = vecs.create_client(DB_CONNECTION_STRING)
    docs = vx.get_collection(name=org_name)
    docs.create_index(measure=vecs.IndexMeasure.cosine_distance)

    vx.disconnect()

    # Cleanup
    stmt = delete(table).where(table.c.id.in_(ids_to_delete))

    with engine.connect() as connection:
        connection.execute(stmt)
        connection.commit()

    return 'Successfully initialized Confluence data.'


@app.route('/ask', methods=['POST'])
@authenticate
def ask_question():
    data = request.json
    query = data['query']
    messages = data['messages']
    client_key = data['clientKey']
    base_url = data['baseUrl']
    org_name = base_url.split('https://')[1].split('.')[0]

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name=org_name
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    llm_predictor = StructuredLLMPredictor()

    create_schema = ResponseSchema(
        name="create",
        description="Use the tool to answer the questions: Does the user want to create something? Answer True if yes or False if not or unknown."
    )

    answer_schema = ResponseSchema(
        name="answer",
        description="Use the tool to give a response in Markdown format."
    )

    # define output schema
    response_schemas = [
        create_schema,
        answer_schema
    ]

    # define output parser
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)

    # format each prompt with output parser instructions
    fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

    # query index
    query_engine = index.as_query_engine(
        service_context=ServiceContext.from_defaults(
            llm_predictor=llm_predictor
        ),
        text_qa_template=qa_prompt, 
        refine_template=refine_prompt, 
    )

    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name="confluence_tool",
        description="useful for querying confluence documentation",
        tool_kwargs={"return_direct": True}
    )

    toolkit = LlamaToolkit(
        index_configs=[tool_config]
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    for message in messages:
        if message["type"] == "user":
            memory.chat_memory.add_user_message(message['value'])
        else:
            memory.chat_memory.add_ai_message(message['value'])

    llm=OpenAI(temperature=0)

    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        query_engine=query_engine,
        verbose=True
    )

    enhanced_query = f"Use the tool to answer: {query}."

    response = agent_chain.run(input=enhanced_query)

    data = {
      "answer": markdown.markdown(linkify(response))
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8080)
