from flask import Flask, request
import requests
from requests.auth import HTTPBasicAuth
import json
from html2text import html2text
import os
import textwrap
import openai

import vecs
from llama_index import ListIndex, SimpleWebPageReader, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import SupabaseVectorStore
from llama_index.node_parser import SimpleNodeParser

from init_env import load_env_vars

load_env_vars()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DB_CONNECTION_STRING = os.environ.get('DB_CONNECTION_STRING')
CONFLUENCE_EMAIL = os.environ.get('CONFLUENCE_EMAIL')
CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN')

app = Flask(__name__)
parser = SimpleNodeParser()

@app.route('/initialize')
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

    print('got here!')

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
    query_engine = index.as_query_engine()

    response = query_engine.query("What were the goals and OKRs listed for the June 2023 all-hands meeting?")

    return str(response)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data['query']

    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION_STRING, 
        collection_name='llm-demo'
    )

    vx = vecs.create_client(DB_CONNECTION_STRING)
    doc_data = vx.get_collection(name="llm-demo")

    embedding = openai.Embedding.create(input = [query], model="text-embedding-ada-002")['data'][0]['embedding']

    docs = doc_data.query(query_vector=embedding, measure="cosine_distance", include_value=True, include_metadata=True, limit=3)

    parsed_docs = []

    for doc in docs:
        entry = doc[2]
        parsed_docs.append(Document(text=entry['text'], doc_id=entry['doc_id'], extra_info={'title': entry['title'], 'url': entry['url']  }))

    index = ListIndex.from_documents(parsed_docs)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return str(response)


if __name__ == '__main__':
    app.run(port=8080)
