import os
import json
from datetime import datetime
from typing import Dict, Any


class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logs_dir = "logs"
        self.session_file = self._create_session_file()
        self._ensure_logs_dir()
    
    def _ensure_logs_dir(self):
        """Создаёт директорию logs если её нет"""
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def _create_session_file(self) -> str:
        """Создаёт уникальное имя файла для сессии"""
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"session_{self.session_id}.jsonl"
        return os.path.join(self.logs_dir, filename)
    
    def log_interaction(self, user_message: str, assistant_message: str, 
                       usage: Dict[str, int], metadata: Dict[str, Any] = None):
        """
        Логирует одно взаимодействие в JSONL файл
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        }
        
        if metadata:
            record["metadata"] = metadata
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def log_faq_answer(self, question: str, answer: str):
        """Логирует ответ из FAQ"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": "faq",
            "user_message": question,
            "assistant_message": answer,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def log_order_status(self, order_id: str, status: str):
        """Логирует запрос статуса заказа"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": "order_status",
            "order_id": order_id,
            "status": status,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def log_error(self, error_message: str, user_message: str = None):
        """Логирует ошибки выполнения"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": "error",
            "error": error_message,
            "user_message": user_message,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")