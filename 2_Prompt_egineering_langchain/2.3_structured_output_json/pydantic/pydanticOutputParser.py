# LangChain предоставляет PydanticOutputParser, который объединяет мощь Pydantic и удобство генерации форматирующих инструкций для модели. Этот парсер:

# генерирует инструкцию format_instructions на основе модели,

# после получения ответа парсит его:
# → JSON-строка → dict → YourModel(**dict)
# (при ошибке — выбрасывает OutputParserException)



import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL")

llm = ChatOpenAI(model=MODEL)

# Определяем схему данных
class BookInfo(BaseModel):
    title: str = Field(..., description="Название книги")
    author: str = Field(..., description="Имя автора")
    tags: list[str] = Field(..., description="Список тегов или жанров")

# Создаём парсер
output_parser = PydanticOutputParser(pydantic_object=BookInfo)
format_instructions = output_parser.get_format_instructions()

# Шаблон промпта
prompt = PromptTemplate(
    template=(
        "Ответь на вопрос в требуемом формате.\n"
        "{format_instructions}\n"
        "Вопрос: {user_question}\nОтвет:"
    ),
    input_variables=["user_question"],
    partial_variables={"format_instructions": format_instructions},
)

# цепочка
chain = prompt | llm | output_parser

# Запуск
# result = chain.invoke({"user_question": "Расскажи о книге '1984' Джорджа Оруэлла."})
# print(type(result)) # <class '__main__.BookInfo'>
# print(result) # title='1984', author='Джордж Оруэлл', tags=['антиутопия', 'классика']



# Метод .with_structured_output()
# Современные версии LangChain позволяют не использовать PydanticOutputParser явно 
# вместо этого можно создать “структурированную” LLM, которая сама возвращает Pydantic-объект:


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")

llm = ChatOpenAI(model=MODEL)

class Profile(BaseModel):
    name: str = Field(description="Имя пользователя")
    age: int = Field(description="Возраст в годах")
    premium: bool = Field(description="Есть премиум-подписка")

structured_llm = llm.with_structured_output(Profile)

result = structured_llm.invoke("Создай профиль русского пользователя средних лет")
print(type(result))   # <class '__main__.Profile'>
print(result.name)    # Алексей Петрович
print(result.age)     # 37
print(result.premium) # False

# Ответ модели будет в формате:

# {
#   "name": "Алексей Петрович",
#   "age": 37,
#   "premium": false
# }

                  
# LangChain автоматически преобразует этот JSON в экземпляр Pydantic-модели Profile