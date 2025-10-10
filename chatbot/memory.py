import json
import os
from datetime import datetime

class MemoryManager:
    def __init__(self, file_path="chat_history.json", max_turns=20):
        self.file_path = file_path
        self.max_turns = max_turns
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.history[-self.max_turns:], f, ensure_ascii=False, indent=2)

    def add_message(self, role, content):
        message = {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(message)
        self.save_history()

    def get_conversation(self):
        return [(msg["role"], msg["content"]) for msg in self.history]

    def clear_history(self):
        self.history = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
