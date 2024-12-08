import os
from dotenv import load_dotenv
import openai
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

# ignore deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create .env file to use your OpenAI API key
load_dotenv()
openai.api_key= os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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