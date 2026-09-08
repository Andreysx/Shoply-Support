import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
MODEL = os.getenv("OPENAI_API_MODEL")

llm = ChatOpenAI(model=MODEL)
#PromptTemplate -  шаблон для однострочного текстового промпта.
#Чтение промпта из файла при помощи PromptTemplate
#Шаблоны промптов создаются для переиспользования многоразовые шаблоны запросов, меняются лишь данные в плейсхолдерах - которые получают от пользователя.
print("PromptTemplate")
print("Чтение промпта из файла: ")
cur_dir_path = Path(__file__).parent
template_path = cur_dir_path / "prompt_template_demo.txt"
template_1 = PromptTemplate.from_file(template_path, encoding="utf-8")

prompt_1 = template_1.invoke({"name":"Андрей", 
                              "error_details": "Ошибка соединения", 
                              "app_name": "Oracle"})

prompt_2 = template_1.invoke({"name":"Геннадий", 
                              "error_details": "Ошибка аутентификации", 
                              "app_name": "Amazon"})

prompt_3 = template_1.invoke({"name":"Иван", 
                              "error_details": "Ошибка 500 Internal Server Error", 
                              "app_name": "Роскомнадзор"})

print("promt_1 res: ", prompt_1)
print("promt_2 res: ", prompt_2)
print("promt_3 res: ", prompt_3)
print("="*15)

#Шаблон для однострочного текстового промпта при помощи PromptTemplate
print("Шаблон для однострочного текстового промпта: ")
template_2 = "Ты опытный переводчик. Переведи текст с английского на русский:\n\"{input_user_text}\""
template = PromptTemplate.from_template(template_2)
# Вариант 1 — format (возвращает просто строку)
prompt_text = template.format(input_user_text="Hello, world!")
print('format result:', prompt_text)

# Вариант 2 — invoke (возвращает объект типа StringPromptValue)
prompt_strobj = template.invoke({"input_user_text": "Hello, world!"})
print('invoke result:', prompt_strobj)
print(type(prompt_strobj))
print("="*15)
print()

print("ChatPromptTemplate")

# ChatPromptTemplate шаблон для моделей чат-формата, которые работают с последовательностью сообщений (с ролями вроде system, human, ai). 
# С его помощью можно определить шаблоны сразу для нескольких сообщений (например: системное сообщение → пользовательское сообщение → модель → пользовательское).
# Создаём шаблон чата с системным и пользовательским сообщением
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Ты — опытный переводчик."),
    HumanMessagePromptTemplate.from_template("Переведи текст с английского на русский:\n\"{input_text}\"")
])

# Вариант 1 — format_messages (возвращает список сообщений/объектов для модели)
messages_list = chat_template.format_messages(input_text="Hello, world!")
print("format_messages result:", messages_list)

# Вариант 2 — invoke (возвращает объект ChatPromptValue со списком внутри него)
prompt_chatobj = chat_template.invoke({"input_text": "Hello, world!"})
print("invoke result:", prompt_chatobj)
print(type(prompt_chatobj))




#Чейнинг промптов (многошаговость)
prompt1 = PromptTemplate.from_template("Назови 5 популярных достопримечательностей в городе {city}.")
prompt2 = PromptTemplate.from_template("Составь маршрут на 3 дня по городу {city}, включив следующие места: {places_list}.")

chain = (
    {"city": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        places_list=lambda x: (prompt1 | llm | StrOutputParser()).invoke(x)
    )
    | prompt2
    | llm
    | StrOutputParser()
)

# Запуск
# result = chain.invoke({"city": "Париж"})
# print(result)
