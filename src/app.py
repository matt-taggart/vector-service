from flask import Flask, request, jsonify
import requests
import time
from requests.auth import HTTPBasicAuth
import json
from html2text import html2text
import os
import textwrap
import openai
import markdown
import jwt
from hashlib import sha256
import urllib.parse
from urllib.parse import urlparse, parse_qs, quote
import marshal
from sqlalchemy import create_engine, text, select, delete, Table, MetaData
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from supabase import create_client, Client
import atlassian_jwt_auth 

import vecs
from llama_index import ListIndex, SimpleWebPageReader, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import SupabaseVectorStore
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts  import Prompt
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain import OpenAI

from init_env import load_env_vars
from auth import authenticate 
from utils import linkify 

load_env_vars()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DB_CONNECTION_STRING = os.environ.get('DB_CONNECTION_STRING')
CONFLUENCE_EMAIL = os.environ.get('CONFLUENCE_EMAIL')
CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN')
PROJECT_PILOT_API_KEY = os.environ.get("PROJECT_PILOT_API_KEY")

app = Flask(__name__)
parser = SimpleNodeParser()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

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

def create_query_string_hash(method, url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract and URL-encode the path
    path = quote(parsed_url.path)

    # Extract and URL-encode the query parameters
    query_params = parse_qs(parsed_url.query)
    canonical_query_string = '&'.join([
        f'{quote(param)}={quote(value[0])}'
        for param, value in sorted(query_params.items())
    ])

    # Create the canonical request string
    canonical_request = f'{method}&{path}&{canonical_query_string}'

    # Compute the SHA-256 hash of the canonical request string
    return sha256(canonical_request.encode('utf-8')).hexdigest()

def generate_atlassian_jwt(client_info_key, client_key, method, url, shared_secret):
    # Generate the QSH
    qsh = create_query_string_hash(method, url)

    # Get the current time
    now = int(time.time())

    # Create the JWT payload
    payload = {
        'iss': client_info_key,
        'iat': now,
        'exp': now + 180,
        'qsh': qsh,
        'aud': client_key,
    }

    # Encode the JWT
    encoded_jwt = jwt.encode(payload, shared_secret, algorithm='HS256')

    return encoded_jwt


def fetchPages(org_name, fragment, client_key, shared_secret, key, documents = []):
    base_url = f"https://{org_name}.atlassian.net"
    url = f"{base_url}{fragment}"
    method = 'GET'
    index = url.find("/api")
    formatted_fragment = url[index:]

    encoded_jwt = generate_atlassian_jwt(key, client_key, method, formatted_fragment, shared_secret)

    headers = {
      "Accept": "application/json",
      "Authorization": f"JWT {encoded_jwt}"
    }

    payload = json.dumps({
      "spaceId": "<string>",
      "status": "current",
      "title": "<string>",
      "parentId": "<string>",
      "_links": {
          "webui": "<string>",
          "links": "<string>",
      },
      "body": {
        "representation": "storage",
        "value": "<string>"
      }
    })

    response = requests.request(
       "GET",
       url,
       data=payload,
       headers=headers
    )

    pages = json.loads(response.text)

    next_link = pages.get("_links", {}).get("next", None)
    pages_results = pages['results']

    for page in pages_results:
        id = page['id']
        title = page['title']
        url_fragment = page['_links']['webui']
        full_url = base_url + url_fragment
        html_content = page['body']['storage']['value']
        content = f"""
         The url for the content below is: {full_url}.

         The content of this document is {html_content}.
        """
        formatted_content = html2text(content)

        documents.append(Document(
            text=formatted_content,
            doc_id=id,
            extra_info={'title': title, 'url': full_url }
        ))

    if next_link:
        return fetchPages(org_name, next_link, client_key, shared_secret, key, documents)

    return documents


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

    documents = fetchPages(org_name, "/wiki/api/v2/pages?body-format=storage&limit=2", client_key, shared_secret, key)

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

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name=org_name
    )
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
    query_engine = index.as_query_engine()

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

    enhanced_query = f"Use the tool to answer: {query}. Output the answer in Markdown format."

    response = agent_chain.run(input=enhanced_query)

    data = {
      "answer": markdown.markdown(linkify(response))
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8080)
