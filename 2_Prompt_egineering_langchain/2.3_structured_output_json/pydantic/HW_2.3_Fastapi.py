import os
from fastapi import FastAPI, HTTPException, status, Query
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL")

app = FastAPI(title="FastAPI для запросов к llm",
              version="0.1.0",
              description="Простое api на FastAPI")

llm = ChatOpenAI(model=MODEL)

# Определяем схему данных - модель Pydantic 
class WeatherInfo(BaseModel):
    city: str = Field(..., max_length=10, description="Название города")
    temperature: float = Field(description="Температура в городе")
    condition: str = Field(description="Общее состояние погоды")

# #Модель для GET запроса погоды
# class WeatherRequest(BaseModel):
#     city: str = Field(..., max_length=10, description="Название города")


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


# Пример с простым консольным input() 
# user_city = input("Введите название города: ")


# try:
#     result = (chain.invoke({"user_city": user_city})).model_dump_json()
# except Exception as e:
#     print({"error": "Invalid response format"})
# else:
#     print(result)




# Пример с fastapi эндпоинтом
@app.get(path="/weather", status_code=status.HTTP_200_OK)
async def get_city_weather(city_name: Annotated[str, Query(max_length=10, description="Узнать погоду в городе: ")]) -> dict():


        # result = (chain.invoke({"user_city": user_city})).model_dump_json()#json
    result = (chain.invoke({"user_city": city_name})) #pydantic-model
    if not result:
        # print({"error": "Invalid response format"})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "Invalid response format"})
    
    return result

# Запуск - uvicorn HW_2_Fastapi:app --port 8080 --reload
