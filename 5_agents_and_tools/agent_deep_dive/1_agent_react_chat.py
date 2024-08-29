from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool,StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from calc_tool import multiply,substract,add,divide

class TimeArgs(BaseModel):
    pass

class CalcArgs(BaseModel):
    num1: int=Field()
    num2:int = Field()

class DateArgs(BaseModel):
    pass

class WikiPediaArgs(BaseModel):
    query: str = Field()
    pass

# Load environment variables from .env file
load_dotenv()


# Define Tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format.It take NO PARAMETERS"""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def get_current_date(*args, **kwargs):
    """Returns the current date in DD/MM/YYYY format.It take NO PARAMETERS"""
    import datetime
    now = datetime.datetime.now()
    return now.date().strftime("%Y-%m-%d")

def get_current_date_and_time(*args, **kwargs):
    """Returns the current date and time in  DD-MM-YYYY HH:MM format.It take NO PARAMETERS"""
    import datetime
    now = datetime.datetime.now()
    dt=now.strftime("%d-%m-%Y %H:%M")
    return dt


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary
    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2,auto_suggest=False, redirect=True)
    except Exception as e:
        print(e)
        return "I couldn't find any information on that."




# Define the tools that the agent can use
tools = [
    # StructuredTool(
    #     name="Date",
    #     func=get_current_date,
    #     description="Useful for when you need to know the current DATE.IT DOES NOT TAKE ANY PARAMETERS",
    #     args_schema=DateArgs,
    # ),
     StructuredTool(
        name="Date and Time",
        func=get_current_date_and_time,
        description="Useful for when you need to know the current DATE AND TIME.IT DOES NOT TAKE ANY PARAMETERS",
        args_schema=DateArgs,
    ),
    # StructuredTool(
    #     name="Time",
    #     func=get_current_time,
    #     description="Useful for when you need to know the current TIME.IT DOES NOT TAKE ANY PARAMETERS",
    #     args_schema=TimeArgs,
    # ),
    StructuredTool(
        name="Addition",
        func=add,
        description="Useful for when you need to add two numbers.Returns sum given two integers",
        args_schema=CalcArgs
    ),
    StructuredTool(
        name="Substraction",
        func=substract,
        description="Useful for when you need to substract two numbers.Returns difference given two integers",
        args_schema=CalcArgs
    ),
    StructuredTool(
        name="Multiplication",
        func=multiply,
        description="Useful for when you need the product of two numbers.Returns product given two integers",
        args_schema=CalcArgs
    ),
     StructuredTool(
        name="Division",
        func=divide,
        description="Useful for when you need to divide two numbers.Returns quotient given two integers",
        args_schema=CalcArgs
    ),
    StructuredTool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful when you need any kind of information on any topics.Takes in a query and returns sumarized article",
        args_schema=WikiPediaArgs
    )
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: 1)Date and Time for current date and time\n2)Addition,Substraction,Multiplication,Division for mathematics\n3) Wikipedia for information.\nUse math tools for all calculations by forming simple binary equations and do not assume you know math."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
