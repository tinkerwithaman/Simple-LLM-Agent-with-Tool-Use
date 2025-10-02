# Project: LangChain_Search_Agent
# Demonstrates an AI agent with tool-use (Tavily Search) for up-to-date answers.
# To run this: pip install langchain-openai langchain-community tavily-python python-dotenv
# Requires API Keys set in a .env file or environment variables.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Setup and Tool Definition ---

# Load environment variables (API keys) from a .env file
load_dotenv() 

# 1. Define the Tools the Agent Can Use
# TavilySearchResults is a tool for web browsing
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# 2. Define the LLM
# Change 'gpt-3.5-turbo' to another model (like Groq's 'llama3-70b-8192') if desired
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
except Exception:
    print("Warning: Could not initialize ChatOpenAI. Check OPENAI_API_KEY.")
    print("Falling back to dummy model. Real tool use will require a valid LLM setup.")
    llm = None 
    # Placeholder to allow script to run, though agent functionality will be limited

if llm:
    # 3. Create the Agent Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful and resourceful AI assistant. Use the provided tools to answer questions that require up-to-date or external information."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Create the Agent Executor (The brain that reasons over the prompt and tools)
    print("Initializing LangChain Agent...")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- Agent Execution ---

    # Agent should use the search tool for this up-to-date question
    question_1 = "What is the capital of Australia and what is the latest news about its leader?"
    print(f"\n--- Question 1 (Requires Tool Use) ---\nInput: {question_1}")
    
    result_1 = agent_executor.invoke({"input": question_1, "chat_history": []})
    print("\nAgent Final Answer:")
    print(result_1["output"])

    # Agent should NOT need the search tool for this general knowledge question
    question_2 = "Explain the concept of photosynthesis in one paragraph."
    print(f"\n--- Question 2 (Does NOT Require Tool Use) ---\nInput: {question_2}")
    
    result_2 = agent_executor.invoke({"input": question_2, "chat_history": []})
    print("\nAgent Final Answer:")
    print(result_2["output"])

else:
    print("Agent could not run due to missing LLM configuration.")

# To run this project successfully:
# 1. Create a file named '.env' in the same directory.
# 2. Add your API key for the LLM (e.g., OPENAI_API_KEY=sk-...).
# 3. Get a free Tavily API key and add it: TAVILY_API_KEY=tvly-....
# 4. Run: python LangChain_Search_Agent.py
