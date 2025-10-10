import os

# Suppress noisy gRPC/ALTS logs (must be set before importing google libs)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

try:
    # Prefer the public Generative AI SDK. Avoid heavy setup at import time.
    import google.generativeai as _genai_pkg  # type: ignore
    _HAS_GENAI = True
except Exception:
    _genai_pkg = None
    _HAS_GENAI = False
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv is optional; provide a no-op fallback so importing this module
    # doesn't fail in minimal environments.
    def load_dotenv():
        return None
from .memory import MemoryManager
from .retriever import RAGRetriever

load_dotenv()

import warnings

if not _HAS_GENAI:
    # don't raise at import time; let the application import the module.
    warnings.warn("google-genai package not installed. Install with: pip install google-genai", RuntimeWarning)

class GeminiBot:
    def __init__(self, name="GeminiBot", model="gemini-2.5-flash-lite", memory_file="chat_history.json"):
        self.name = name
        self.memory = MemoryManager(memory_file)
        self.retriever = RAGRetriever()
        if not _HAS_GENAI or _genai_pkg is None:
            raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                _genai_pkg.configure(api_key=api_key)
            except Exception as exc:
                raise RuntimeError(f"google-generativeai configure failed: {exc}")
        else:
            warnings.warn("GEMINI_API_KEY not set. Set it via environment or .env file.", RuntimeWarning)
        try:
            self._genai_model = _genai_pkg.GenerativeModel(model)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GenerativeModel '{model}': {exc}")
        # maintain legacy attributes for compatibility with older calling code
        self.model = None
        self.chat = None

    def ask(self, message, use_rag=True, history_turns=10):
        context = ""
        if use_rag:
            results = self.retriever.retrieve(message)
            if results:
                context = "\n\n".join(results)

        history_snippets = []
        if history_turns:
            conversation = self.memory.get_conversation()
            if conversation:
                # Pull the last N entries from the stored conversation.
                for role, content in conversation[-history_turns:]:
                    history_snippets.append(f"{role}: {content}")
        history_block = "\n".join(history_snippets).strip() or "(không có lịch sử phù hợp)"
        context_block = context.strip() or "(không có tài liệu tham chiếu phù hợp)"

        prompt = f"""
        Bạn là trợ lý AI chuyên nghiệp tên {self.name}. Trả lời bằng tiếng Việt rõ ràng, chính xác và đầy đủ.

        DỮ LIỆU ĐẦU VÀO
        1. Câu hỏi hiện tại của người dùng:
           "{message}"
        2. Lịch sử trò chuyện gần đây (vai trò: nội dung):
           {history_block}
        3. Tài liệu/thông tin tham chiếu từ hệ thống:
           {context_block}

        HƯỚNG DẪN TRẢ LỜI
        - Ưu tiên sự chính xác; chỉ sử dụng thông tin có trong lịch sử và tài liệu tham chiếu.
        - Nếu câu trả lời cần suy luận, hãy nêu lập luận ngắn gọn dựa trên dữ liệu.
        - Nếu thiếu thông tin, hãy nêu rõ phần chưa biết và đưa ra gợi ý tiếp theo.
        - Giữ cấu trúc mạch lạc với đoạn văn hoặc gạch đầu dòng khi cần.
        """
        # Use google-generativeai to generate a response from the model.
        try:
            response = self._genai_model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                answer = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                parts = []
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    if content and getattr(content, "parts", None):
                        for part in content.parts:
                            text = getattr(part, "text", None)
                            if text:
                                parts.append(text)
                answer = "\n".join(parts).strip() if parts else str(response).strip()
            else:
                answer = str(response).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to get response from google-generativeai client: {e}")
        self.memory.add_message("user", message)
        self.memory.add_message("assistant", answer)
        return answer

    def clear_context(self):
        self.memory.clear_history()
        # No persistent chat object when using the `google-genai` client
        # in this simplified adapter; just clear conversation memory.
        self.chat = None
        return "🧹 Bộ nhớ hội thoại đã được xóa!"

