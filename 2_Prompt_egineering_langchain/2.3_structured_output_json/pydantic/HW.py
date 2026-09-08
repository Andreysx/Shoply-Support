import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL")

llm = ChatOpenAI(model=MODEL)

# Определяем схему данных - модель Pydantic 
class WeatherInfo(BaseModel):
    city: str = Field(..., max_length=10, description="Название города")
    temperature: float = Field(description="Температура в городе")
    condition: str = Field(description="Общее состояние погоды")


# Создаём парсер
output_parser = PydanticOutputParser(pydantic_object=WeatherInfo)
format_instructions = output_parser.get_format_instructions()

# Шаблон промпта
prompt = PromptTemplate(
    template=(
        "Ответь на вопрос в требуемом формате.\n"
        "{format_instructions}\n"
        "Информация о погоде в {user_city}\nОтвет:"
    ),
    input_variables=["user_city"],
    partial_variables={"format_instructions": format_instructions},
)

# цепочка
chain = prompt | llm | output_parser



user_city = input("Введите название города: ")


try:
    result = (chain.invoke({"user_city": user_city})).model_dump_json()
except Exception as e:
    print({"error": "Invalid response format"})
else:
    print(result)


# Запуск
# result = chain.invoke({"user_city": user_city})
# print(type(result)) # <class '__main__.BookInfo'>
# print(result) # title='1984', author='Джордж Оруэлл', tags=['антиутопия', 'классика']