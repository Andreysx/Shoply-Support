
#!/usr/bin/env python3
import os
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

def main() -> int:
    # 1) Загружаем переменные окружения (.env не обязателен)
    load_dotenv()

    # 2) Считываем аргументы запуска    
    parser = argparse.ArgumentParser(description="Первый запрос к LLM")
    parser.add_argument(
        "-q", "--query",
        default="Привет, бот! Назови столицу Франции.",
        help="Текст запроса к модели (по умолчанию — простой вопрос)"
    )
    args = parser.parse_args()

    # 3) Создаём клиента OpenAI (v1.x)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # Если используете прокси, задайте свой базовый URL
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        # при желании можно задать request_timeout=15 для всех вызовов
    )

    # 4) Делаем минимальный запрос (Responses API)
    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # подставьте свою модель при необходимости
        instructions="Ты дружелюбный ассистент. Отвечай кратко и по делу.",
        input=args.query,
        # temperature=0.0,
        # max_output_tokens=200,
        # timeout=15,  # чтобы не зависать при долгом ответе
    )

    # 5) Печатаем ответ (готовый текст)
    print(resp.output_text.strip())

    # 6) (необязательно) печатаем число токенов, если оно доступно
    usage = getattr(resp, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage is not None else None
    if isinstance(total, int):
        print(f"\n(токены: {total})")

    return 0

if __name__ == "__main__":
    sys.exit(main())
