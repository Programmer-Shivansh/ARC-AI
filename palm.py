from langchain_community.llms.google_palm import GooglePalm
from dotenv import load_dotenv
# from langchain.llms import google_palm as GooglePalm
import os
load_dotenv()
api_key = os.getenv('GOOGLEPALM_API_KEY')

llm = GooglePalm(google_api_key=api_key,temperature=0.9)

response = llm("how many times can i use google palm api key tell what are the limits to use freely ")
print(response)