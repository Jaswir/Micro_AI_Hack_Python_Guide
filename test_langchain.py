import os
import json
import pymongo
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import AzureCosmosDBVectorSearch
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_core.messages import SystemMessage

# Load settings for the notebook
load_dotenv()
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
COMPLETIONS_DEPLOYMENT_NAME = "gpt-35-turbo"
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_KEY = os.environ.get("AOAI_KEY")
AOAI_API_VERSION = "2023-09-01-preview"


# Establish Azure OpenAI connectivity
# llm = AzureChatOpenAI(            
#         temperature = 0,
#         openai_api_version = AOAI_API_VERSION,
#         azure_endpoint = AOAI_ENDPOINT,
#         openai_api_key = AOAI_KEY,         
#         azure_deployment = COMPLETIONS_DEPLOYMENT_NAME
# )

embedding_model = AzureOpenAIEmbeddings(
    openai_api_version = AOAI_API_VERSION,
    azure_endpoint = AOAI_ENDPOINT,
    openai_api_key = AOAI_KEY,   
    azure_deployment = EMBEDDINGS_DEPLOYMENT_NAME,
    chunk_size=10
)


# llm.invoke("hello")
query = "Hello?"
embedding_vector = embedding_model.embed_query(query)
print(embedding_vector)