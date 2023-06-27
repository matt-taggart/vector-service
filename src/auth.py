import os
from flask import Flask, request, jsonify
from functools import wraps
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
            jwt.decode(token, shared_secret, algorithms=["HS256"])
            return func(*args, **kwargs)
        except jwt.exceptions.InvalidTokenError as e:
            return jsonify({"message": "Unauthorized request"}), 403

    return decorated_function
