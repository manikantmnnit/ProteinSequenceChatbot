from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

def initialize_llm(api_key, temperature=0.7, max_tokens=1000, model="llama3-8b-8192"):
    """Initialize the Groq LLM with given parameters"""
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )

def create_chain(llm):
    """Create the chat chain with system prompt"""
    system_prompt = """As a protein scientist, analyze the sequence and its properties..."""  
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}")
    ])
    return prompt | llm

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults

def get_llm_response(llm, prompt, user_input, chat_history, tools=None):
    """
    Enhanced version supporting both direct LLM queries and tool usage
    
    Args:
        tools: List of tools (e.g., [TavilySearchResults()])
    """
    if not tools:  # Original direct LLM flow
        chain = prompt | llm
        response = chain.invoke({
            "user_input": user_input,
            "chat_history": chat_history
        })
        content = response.content
    else:  # Agent with tools flow
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        content = response["output"]

    # Update history (works for both cases)
    updated_history = chat_history + [
        HumanMessage(content=user_input),
        AIMessage(content=content)
    ]
    
    return content, updated_history
