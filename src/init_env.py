import os
from dotenv import dotenv_values

def load_env_vars():
    # Load environment variables from .env file
    env_vars = dotenv_values('.env')

    # Set the environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

