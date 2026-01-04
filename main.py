from importlib import reload
import tools.aquah_run
reload(tools.aquah_run)
from tools.aquah_run import aquah_run
import os
import getpass

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API Key not found in environment variables.")
    api_key = getpass.getpass("Please enter your OpenAI API Key (input will be hidden): ")

if not api_key:
    raise ValueError("No API Key provided. Exiting.")
os.environ["OPENAI_API_KEY"] = api_key
llm_model_name = 'gpt-4o'
# llm_model_name = 'claude-4-sonnet-20250514'
# llm_model_name = 'gemini-2.5-flash-preview-05-20'
# llm_model_name = 'claude-4-opus-20250514'
aquah_run(llm_model_name)