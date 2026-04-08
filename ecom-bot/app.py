import os
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from custom_logger import SessionLogger



# Создаём класс для CLI-бота
class Cli_bot():
    def __init__(self, model_name, system_prompt="Ты бот поддержки."):
        # Создаём модель
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=15
        )

        # Создаём Хранилище истории
        self.store = {} 

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), # Добавим возможность менять системный промпт
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain, # Цепочка с историей
            self.get_session_history, # метод для получения истории
            input_messages_key="question", # ключ для вопроса
            history_messages_key="history", # ключ для истории
        )
        
        self.faq = ["Как оформить возврат?", 
                    "Сколько идёт доставка?", 
                    "Какие способы оплаты?", 
                    "Как изменить адрес после заказа?", 
                    "Как применить промокод?"]

        self.session_loggers = {}


    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    # Метод возвращает или создаёт логгер для сессии
    def get_session_logger(self, session_id: str) -> SessionLogger:
        if session_id not in self.session_loggers:
            self.session_loggers[session_id] = SessionLogger(session_id)
            print(f"Лог сессии: {self.session_loggers[session_id].session_file}")
        return self.session_loggers[session_id]
    
    

    def __call__(self, session_id):
        print("Чат-бот поддержки магазина Shoply запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n - Команда /order order_number показывает статус заказа по его номеру.")

        session_logger = self.get_session_logger(session_id)
        
        # Логируем начало сессии
        session_logger.log_interaction(
            user_message="Начало сессии",
            assistant_message="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata={"event": "session_start"}
        )      
       
       
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                exit_msg = "Бот: Завершение работы."
                print(f"\n{exit_msg}")
                session_logger.log_interaction(
                    user_message="Ctrl + C",
                    assistant_message=exit_msg,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    metadata={"event": "session_end"}
                )
                break
            if not user_text:
                continue


            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                # Логируем завершение сессии
                session_logger.log_interaction(
                    user_message=user_text,
                    assistant_message="До свидания!",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    metadata={"event": "session_end"}
                )
                break
            
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                session_logger.log_interaction(
                    user_message=user_text,
                    assistant_message="Контекст диалога очищен",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    metadata={"event": "context_reset"}
                )
                continue
            
            # Обработка FAQ
            msg_faq = user_text.strip()
            if msg_faq in self.faq:     
                with open("ecom-bot/data/faq.json", "r", encoding='utf-8') as f:
                    data = json.load(f, )
                    for d in data:
                        if d["q"].lower() == msg_faq.lower():
                            info = d["a"]
                            print(f"Бот: {info}")
                            session_logger.log_faq_answer(user_text, info)
                continue
        
            # Обработка запроса по номеру заказа (/order)              
            if user_text.startswith("/order"):
                with open("ecom-bot/data/orders.json", "r", encoding='utf-8') as f:
                    data = json.load(f)   
                    for k in data.keys():
                        tmp = user_text.lower().split()
                        order_number = user_text.lower().split()[1] if len(tmp) > 1 else None
                        if order_number is None or order_number == " ":
                            msg = "Пожалуйста введите номер заказа в формате: /order + (пробел) + номер заказа."
                            print(f"Бот: {msg}")
                            session_logger.log_error(msg, user_text)
                            break
                            
                        if order_number == k:
                            order_info = data[k]
                            status = order_info.get("status")                   
                            print(f"Бот: Текущий статус вашего заказ - {status}")
                            session_logger.log_order_status(k, status)
                            break
                    else: 
                        not_found_msg = "К сожалению не смогли найти заказ по вашему запросу."    
                        print(f"Бот: {not_found_msg}")
                        session_logger.log_error(not_found_msg, user_text)
                        
                continue 
            
            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
                
                if hasattr(response, 'usage_metadata'):
                    usage = {
                        "prompt_tokens": response.usage_metadata.get('input_tokens', 0),
                        "completion_tokens": response.usage_metadata.get('output_tokens', 0),
                        "total_tokens": response.usage_metadata.get('total_tokens', 0)
                    }
            except Exception as e:
                # Логируем и выводим ошибку, продолжаем чат
                print(f"[Ошибка] {e}")
                session_logger.log_error(str(e), user_text)
                continue
            
            # Форматируем и выводим ответ
            bot_reply = response.content.strip()
            session_logger.log_interaction(user_text, bot_reply, usage)
            print(f"Бот: {bot_reply}")


if __name__ == "__main__":
    model = os.getenv("OPENAI_API_MODEL")
    system_prompt = '''Ты вежливый бот поддержки магазина "Shoply". Отвечай кратко и по существу.'''

    bot = Cli_bot(
        model_name=model,
        system_prompt=system_prompt
    )
    bot("user_5")