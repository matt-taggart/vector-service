from flask import Flask, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
import json
from html2text import html2text
import os
import textwrap
import openai
import markdown
import jwt

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

load_env_vars()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DB_CONNECTION_STRING = os.environ.get('DB_CONNECTION_STRING')
CONFLUENCE_EMAIL = os.environ.get('CONFLUENCE_EMAIL')
CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN')
PROJECT_PILOT_API_KEY = os.environ.get("PROJECT_PILOT_API_KEY")

app = Flask(__name__)
parser = SimpleNodeParser()

@app.route('/initialize')
@authenticate
def get_connection():
    url = "https://redelklabs.atlassian.net/wiki/api/v2/pages?body-format=storage"

    auth = HTTPBasicAuth(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)

    headers = {
      "Accept": "application/json"
    }

    payload = json.dumps({
      "spaceId": "<string>",
      "status": "current",
      "title": "<string>",
      "parentId": "<string>",
      "_links": {
          "webui": "<string>"
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
       headers=headers,
       auth=auth
    )

    pages = json.loads(response.text)
    pages_results = pages['results']

    documents = []

    for page in pages_results:
        id = page['id']
        title = page['title']
        base_url = 'https://redelklabs.atlassian.net/wiki' 
        url_fragment = page['_links']['webui']
        full_url = base_url + url_fragment
        html_content = page['body']['storage']['value']
        content = html2text(html_content)

        documents.append(Document(
            text=content,
            doc_id=id,
            extra_info={'title': title, 'url': full_url }
        ))

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name='llm-demo'
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return 'Successfully initialized Confluence data.'

@app.route('/ask', methods=['POST'])
@authenticate
def ask_question():
    data = request.json
    query = data['query']
    messages = data['messages']

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name='llm-demo'
    )

    vx = vecs.create_client(DB_CONNECTION_STRING)
    doc_data = vx.get_collection(name="llm-demo")

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

    print('markdown', markdown.markdown(response));

    data = {
      "answer": markdown.markdown(response)
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8080)
