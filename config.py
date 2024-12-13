import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('openai_api_key', "")