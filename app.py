import os
import streamlit as st
from typing import List, Tuple
import json
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import Tool
from langserve import add_routes
from langchain.prompts import PromptTemplate
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.agents import Tool, Agent, AgentType
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
embeddings = OpenAIEmbeddings()
llm_1 = AzureChatOpenAI(openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt4chat"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://gpt-4-trails.openai.azure.com/"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"))

llm = ChatOpenAI(temperature=0.2,
                 model="gpt-3.5-turbo-0125",
                 streaming=True,
                 callbacks=[FinalStreamingStdOutCallbackHandler()]).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM"))
assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Define the API call function for Ares API

def api_call(text):
  url = "https://api-ares.traversaal.ai/live/predict"

  payload = { "query": [text]}
  headers = {
    "x-api-key": "ares_a0866ad7d71d2e895c5e05dce656704a9e29ad37860912ad6a45a4e3e6c399b5",
    "content-type": "application/json"
  }

  response = requests.post(url, json=payload, headers=headers)

  # here we will use the llm to summarize the response received from the ares api
  response_data = response.json()
  #print(response_data)
  try:
    response_text = response_data['data']['response_text']
    web_urls = response_data['data']['web_url']
    # Continue processing the data...
  except KeyError:
    print("Error: Unexpected response from the API. Please try again or contact the api owner.")
    # Optionally, you can log the error or perform other error handling actions.
  

  if len(response_text) > 10000:
    response_text = response_text[:8000]
    prompt = f"Summarize the following text in 500-100 0 words and jsut summarize what you see and do not add anythhing else: {response_text}"
    summary = llm_1.invoke(prompt)
    print(summary)
  else:
    summary = response_text

  result = "{} My list is: {}".format(response_text, web_urls)

# Convert the result to a string
  result_str = str(result)

  return result_str



def metadata_func(record: str, metadata: dict) -> dict:
    lines = record.split('\n')
    locality_line = lines[10]
    price_range_line = lines[12]
    locality = locality_line.split(': ')[1]
    price_range = price_range_line.split(': ')[1]
    metadata["location"] = locality
    metadata["price_range"] = price_range

    return metadata

# Instantiate the JSONLoader with the metadata_func
jq_schema = '.parser[] | to_entries | map("\(.key): \(.value)") | join("\n")'
loader = JSONLoader(
    jq_schema=jq_schema,
    file_path='data.json',
    metadata_func=metadata_func,
)

# Load the JSON file and extract metadata
documents = loader.load()

    
from langchain.vectorstores import FAISS
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Check if the FAISS index file already exists
    if os.path.exists("faiss_index"):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings)
        print("Loaded existing FAISS index.")
    else:
        # Create a new FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        # Save the new FAISS index locally
        vectorstore.save_local("faiss_index")
        print("Created and saved new FAISS index.")
    return vectorstore

#docs = new_db.similarity_search(query)

vector = get_vectorstore(documents)

template = """

context:- I have low budget what is the best hotel in Instanbul?
anser:- The other hotels in instanbul are costly and are not in your budget. so the best hotel in instanbul for you is hotel is xyz."

Don’t give information not mentioned in the CONTEXT INFORMATION. 
The system should take into account various factors such as location, amenities, user reviews, and other relevant criteria to 
generate informative and personalized explanations.
{context} 
Question: {question}
Answer:"""

def search():
    #llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    vector = vector
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    chain_type_kwargs = {"prompt": prompt}
    return RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

# Initialize LangChain tools


api_tool = Tool(name="Ares_API",
                func=api_call,
                description="Integration with Traversaal AI Ares API for real-time internet searches."
               )

chain_rag_tool = Tool(name="RAG_Chain",
                      func=search,
                      description="RAG chain for question answering."
                     )


tools = [chain_rag_tool, api_tool]
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)


def get_response(user_input):
    response = agent_executor.invoke({"input":user_input, "chat_history": _format_chat_history([])})
    return response


# Define the Streamlit app
def main():
    st.title("Chatbot")
    user_input = st.text_input("User Input:", "")

    if st.button("Submit"):
        response = get_response(user_input)
        st.text("Bot Response:")
        st.write(response)

# Function to get response from LangChain model or API
def get_response(user_input):
    # Example: You can choose either to use LangChain model or API
    response = llm.invoke(user_input)
    # response = api_call(user_input)
    return response.content

if __name__ == "__main__":
    main()