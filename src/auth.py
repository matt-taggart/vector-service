import os
import time
from functools import wraps
from urllib.parse import urlparse, parse_qs, quote
from hashlib import sha256
from flask import Flask, request, jsonify
from supabase import create_client, Client
import jwt

from init_env import load_env_vars

load_env_vars()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
PROJECT_PILOT_API_KEY = os.environ.get("PROJECT_PILOT_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def authenticate(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        data = request.json
        client_key = data['clientKey']
        response = supabase.table('companies').select('shared_secret').eq('client_key', client_key).execute()
        shared_secret = response.data[0]['shared_secret']

        token_header = request.headers.get('Authorization')
        token = token_header.split(' ')[1]
        api_key = request.headers.get('x-project-pilot-api-key')

        if not token:
            return jsonify({"message": "Unauthorized request"}), 403

        if not api_key:
            return jsonify({"message": "Unauthorized request"}), 403

        try:
            if api_key != PROJECT_PILOT_API_KEY:
                raise ExceptionType("Error message")
        except ValueError:
            return jsonify({"message": "Unauthorized request"}), 403

        try:
            jwt.decode(token, shared_secret, algorithms=["HS256"], audience=client_key)
            return func(*args, **kwargs)
        except jwt.exceptions.InvalidTokenError as e:
            return jsonify({"message": "Unauthorized request"}), 403

    return decorated_function

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
