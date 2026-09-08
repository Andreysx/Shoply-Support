import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL")

llm = ChatOpenAI(model=MODEL)

parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="Пожалуйста, верни данные в формате JSON.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llm | parser

result = chain.invoke({"question": "Что такое LangChain?"})
print(result)  # вывод: {'question': 'Что такое LangChain?', 'answer': 'LangChain — это ...'}


# современные версии OpenAI API (и, соответственно, LangChain) позволяют включить нативный режим возврата данных строго в формате JSON.
# Для этого при создании модели достаточно «привязать» параметр response_format={"type": "json_object"}

# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# json_llm = ChatOpenAI(model="gpt-5").bind(response_format={"type": "json_object"})# !!!

# responce = json_llm.invoke("верни JSON объект с ключом 'random_ints' и значением из 10 случайных чисел в диапазоне [0-99]")

# print(responce.content) # вывод: '\n{\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\n}'