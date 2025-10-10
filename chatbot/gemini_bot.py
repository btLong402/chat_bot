import os

# Suppress noisy gRPC/ALTS logs (must be set before importing google libs)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

import google.generativeai as genai
from dotenv import load_dotenv
from .memory import MemoryManager
from .retriever import RAGRetriever

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiBot:
    def __init__(self, name="GeminiBot", model="gemini-2.5-flash-lite", memory_file="chat_history.json"):
        self.name = name
        self.memory = MemoryManager(memory_file)
        self.retriever = RAGRetriever()
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat(history=[])

    def ask(self, message, use_rag=True):
        context = ""
        if use_rag:
            results = self.retriever.retrieve(message)
            if results:
                context = "\n\n".join(results)
        
        prompt = f"""
        Đây là nội dung người dùng hỏi: "{message}"
        Nếu có, hãy dùng thông tin sau để trả lời chính xác:
        -----------------------
        {context}
        -----------------------
        """
        response = self.chat.send_message(prompt)
        answer = response.text.strip()
        self.memory.add_message("user", message)
        self.memory.add_message("assistant", answer)
        return answer

    def clear_context(self):
        self.memory.clear_history()
        self.chat = self.model.start_chat(history=[])
        return "🧹 Bộ nhớ hội thoại đã được xóa!"

