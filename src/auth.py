from flask import Flask, request, jsonify

def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the authentication token is present in the request headers
        auth_token = request.headers.get('Authorization')
        if auth_token and auth_token == 'YOUR_AUTH_TOKEN':
            return func(*args, **kwargs)
        else:
            return jsonify({'message': 'Unauthorized'}), 401

    return wrapper
