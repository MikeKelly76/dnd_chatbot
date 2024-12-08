"""
NAME:           dnd_bot.py
AUTHOR:         Mike Kelly
DATE:           December 12, 2024
DESCRIPITON:    A chatbot program that takes user questions about the 2024 Dungeons & Dragons rules update
                and provides that user with an answer in the form of rules text.
OTHER FILES:    > .env (not included) - File with the OpenAI API key necessary for embedding and execution of the LLM
                > requirements.txt - File with all Python dependencies necessary to run the project.
                        $pip install -r requiements.txt to install in local environment.
                > jsonFiles (not included) - This is a directory that contains the data in JSON format used to
                        create embeddings by the LLM for the vector database. You must create and include your own
                        files. Please only use information that you have permission to use.
                > vector_db - File directory that contain the vector database information necessary for running the 
                        question and answer execution chain.    
"""

import os
from dotenv import load_dotenv
import openai
from langchain.document_loaders import JSONLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.llms import OpenAI
from langchain import LLMChain
import gradio as gr
import warnings

# ignore deprication warnings because they're annoying
warnings.filterwarnings("ignore", category=DeprecationWarning)


# load the documents for parsing and embedding
# Json files
json_load = DirectoryLoader('jsonFiles', glob="**/[!.]*.json", loader_cls=JSONLoader, loader_kwargs={"jq_schema" : ".[]", "text_content" : False})
json_data = json_load.load()
json_data

# split the text for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20,
    length_function = len
)

data = json_data
documents = text_splitter.split_documents(data)


# Load OpenAI API key from .env file
load_dotenv()
openai.api_key= os.environ["OPENAI_API_KEY"]

# Choose embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Persistent directory name for the local vector database returned from OpenAI
persist_directory = "vector_db"

"""
This line is used to create the embeddings and store them locally in a directory named "vector_db"
Leave commented if embeddings have already been created.
"""
#vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)

# Load the embeddings saved in the local directory
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Load the large language model to be used for the chatbot.
llm = ChatOpenAI(temperature = 0, model="gpt-4")

# Set up the document retriever
doc_retriever = vectordb.as_retriever()

# Setup the Retreval QA chain 
dnd_bot = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=doc_retriever)

#agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)

dnd_bot = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=doc_retriever, memory=readonlymemory)

# Define tools for the agent 
tools = [
    Tool(
        name = "Dungeons and Dragons 2024 QA System",
        func=dnd_bot.run,
        description="Useful for when you need to answer questions about the Dungeons and Dragons 2024 rules. Input should be a fully formed question."
    )
]


prefix = """Have a conversation with a human, answering the following questions as best you can. 
You are an expert in the 2024 Dungeons and dragons ruleset.
Retrieve information for your answers from the Dungeons and Dragons 2024 QA system.
Do not alter the wording of the information retrieved from the QA system.
Present the information retrieved for the answer in its entirety.
Output your responses in a list bullet format when applicable.
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

from langchain import OpenAI, LLMChain, PromptTemplate

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True , handle_parsing_errors=True, memory=memory)


def dnd_chat_bot(message, history):
    response = agent_chain.run(message)
    return response

gr.ChatInterface(
    fn=dnd_chat_bot,
    type="messages",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Hello Adventurer. Ask me a question about D&D", container=True, scale=7),
    title="D&D BOT",
    description="Ask D&D BOT a question about the D&D 2024 rules",
    theme='YTheme/Minecraft',
).launch()
