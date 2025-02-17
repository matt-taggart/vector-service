import requests
import json
from llama_index import Document
from html2text import html2text

from auth import generate_atlassian_jwt 

def process_requests(resource, org_name, fragment, client_key, shared_secret, key, documents = None):
    if documents is None:
        documents = []
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

         The content of this {resource} is {html_content}.
        """
        formatted_content = html2text(content)

        documents.append(Document(
            text=formatted_content,
            doc_id=id,
            extra_info={'title': title, 'url': full_url }
        ))

    if next_link:
        return process_requests(resource, org_name, next_link, client_key, shared_secret, key, documents)

    return documents
 
def fetch_tasks(resource, org_name, fragment, client_key, shared_secret, key, documents = None):
    if documents is None:
        documents = []
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
      "id": "<string>",
      "localId": "<string>",
      "spaceId": "<string>",
      "pageId": "<string>",
      "blogPostId": "<string>",
      "status": "<string>",
      "body": {
        "representation": "storage",
        "value": "<string>"
      },
      "_links": {
          "webui": "<string>",
          "links": "<string>",
      },
      "createdBy": "<string>",
      "assignedTo": "<string>",
      "completedBy": "<string>",
      "createdAt": "<string>",
      "updatedAt": "<string>",
      "dueAt": "<string>",
      "completedAt": "<string>"
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
        html_content = page['body']['storage']['value']
        print('html_content', html_content)

        content = f"""
         The content of this {resource} is {html_content}.
        """
        formatted_content = html2text(content)

        documents.append(Document(
            text=formatted_content,
            doc_id=id
        ))

    if next_link:
        return fetch_tasks(resource, org_name, next_link, client_key, shared_secret, key, documents)

    return documents
