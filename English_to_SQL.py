from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

HUGGINGFACEHUB_API_TOKEN = ""

hub_llm = HuggingFaceHub(
    repo_id="mrm8488/t5-base-finetuned-wikiSQL",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

user_question = input("Enter your question: ")

template = "Translate English to SQL: {question}"

input_variables = ["question"]

prompt = PromptTemplate(template=template, input_variables=input_variables)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run(user_question))
