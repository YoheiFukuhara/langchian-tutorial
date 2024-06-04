#!/usr/bin/env python
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.globals import set_debug
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langserve import add_routes

import os
load_dotenv('../', verbose=True)
print(os.environ['OPENAI_API_KEY'])

set_debug(True)

# 1. Create prompt template
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions="You are an helpful agent.")

tools = [DuckDuckGoSearchRun()]

# 2. Create model
model = ChatOpenAI()
_agent = create_openai_tools_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=_agent,
                               tools=tools,
                               verbose=True
                               )

# 3. Create parser
#parser = OpenAIFunctionsAgentOutputParser()

# 4. Create chain
#chain = agent_executor | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

# 5. Adding chain route
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)