from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model= ChatOpenAI(model="gpt-3.5-turbo-0125")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert prodict reviewer"),
        ("human","List all the main fetaures of the {product}")
    ]
)

def analyze(str1,features):
    template =ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert product reviewer"),
            ("human","Given these features :{features},list the {str1} of the features")
        ]
    )
    return template.format_prompt(features=features,str1=str1)

def combine(pros,cons):
    return f"Pros:\n {pros} \n Cons:\n {cons}"

pros_branch_chain = (
    RunnableLambda(lambda x:analyze("pros",x)) | model | StrOutputParser())


cons_branch_chain = (
    RunnableLambda(lambda x:analyze("cons",x)) | model | StrOutputParser())


chain = (
    prompt_template 
    | model 
    | RunnableParallel(branches={"pros":pros_branch_chain,"cons":cons_branch_chain})
    | RunnableLambda(lambda x:combine(x["branches"]["pros"],x["branches"]["cons"]))
)

result = chain.invoke({"product":"Iphone 10R"})

print(result)