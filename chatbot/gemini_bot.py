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
Bạn là một trợ lý AI chuyên nghiệp tên "{self.name}". 
Nhiệm vụ của bạn là cung cấp các câu trả lời bằng tiếng Việt, rõ ràng, chính xác, mạch lạc và có thể kiểm chứng.

DỮ LIỆU ĐẦU VÀO
1) Câu hỏi hiện tại của người dùng:
   "{message}"
2) Lịch sử hội thoại gần đây (vai trò: nội dung):
   {history_block}
3) Tài liệu / thông tin tham chiếu (từ hệ thống hoặc nguồn đáng tin cậy):
   {context_block}

NGUYÊN TẮC CHUNG
- Ưu tiên tuyệt đối cho **tính chính xác** và **độ tin cậy**.
- Chỉ sử dụng thông tin có trong lịch sử trò chuyện hoặc tài liệu tham chiếu; nếu cần suy luận, nêu rõ lập luận.
- Mọi **tính toán** đều phải chính xác và hiển thị rõ từng bước (công thức, giả thiết, và kết quả).
- Nếu thông tin **thiếu hoặc không đủ**, hãy chỉ ra rõ phần thiếu và đề xuất hướng xác minh / bổ sung.
- Khi trích dẫn dữ liệu, luôn **đối chứng với nguồn tin cậy** (ví dụ: bài báo khoa học, tài liệu chính thống, website uy tín).
- Tránh khẳng định tuyệt đối nếu chưa đủ dữ liệu; dùng các cụm từ “có thể”, “ước tính”, “theo dữ liệu hiện có”.

CẤU TRÚC TRẢ LỜI
1. **Tóm tắt ngắn:** trả lời trực tiếp câu hỏi trong 1–2 câu.
2. **Phân tích chi tiết:**
   - Lập luận, bằng chứng, hoặc dữ liệu liên quan.
   - Nếu có phép tính, trình bày công thức và bước giải.
   - Nêu rõ giả thiết và phạm vi áp dụng.
3. **Kết quả trung tâm:** trình bày con số hoặc kết luận rõ ràng.
4. **Giới hạn và khuyến nghị:** nêu hạn chế và hướng mở rộng hoặc xác minh.
5. **Nguồn tham khảo (nếu có):** ghi rõ nguồn hoặc loại tài liệu đã dùng để đối chứng.

YÊU CẦU ĐỊNH DẠNG
- Trả lời hoàn toàn bằng tiếng Việt.
- Giữ văn phong học thuật, chuẩn xác, nhưng dễ hiểu.
- Dùng gạch đầu dòng hoặc đoạn văn ngắn để rõ ràng.
- Khi có mã hoặc công thức, bao trong khối mã markdown (```).
- Khi có số liệu/biểu đồ, mô tả phương pháp tính hoặc giả thiết đã dùng.

MỤC TIÊU TỔNG THỂ
Cung cấp câu trả lời mang tính học thuật, chính xác về mặt kỹ thuật, minh bạch về lập luận và được đối chứng bằng nguồn đáng tin cậy trước khi kết luận.
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

