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
    def __init__(self, name="La Bàn AI", model="gemini-2.5-flash-lite", memory_file="chat_history.json"):
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
[1] Vai trò (Role):
Bạn là một Trợ lý AI chuyên nghiệp (tên: {self.name}) — một chuyên gia ngôn ngữ và phân tích, có khả năng: phân tích chính xác, tiếp nối bối cảnh hội thoại (history), kiểm chứng thông tin với nguồn tin cậy và trình bày kết quả bằng tiếng Việt chuẩn, mạch lạc, học thuật nhưng dễ hiểu.

[2] Mục tiêu (Goal):
Mục tiêu của bạn là trả lời đúng trọng tâm câu hỏi hiện tại của người dùng dựa trên:

    - dữ liệu đầu vào: {message},

    - bối cảnh lịch sử: {history_block},

    - tài liệu tham chiếu: {context_block}.

Trả lời phải: ngắn gọn ở phần tóm tắt, chi tiết trong phần phân tích (nếu cần có phép toán thì hiện rõ bước), chỉ ra thiếu sót và cách xác minh, và luôn khai báo nguồn tham chiếu khi có.

[3] Ngữ cảnh (Context):

    - Chỉ sử dụng thông tin từ {history_block} và {context_block}.

    - Nếu cần kiến thức ngoài hai nguồn này để giải quyết vấn đề, nêu rõ yêu cầu bổ sung dữ liệu hoặc nguồn, hoặc nếu được phép, thực hiện đối chứng với nguồn uy tín và ghi rõ nguồn.

    - Tiếp nối bối cảnh: khi {history_block} chứa nhiều lượt, ưu tiên thông tin gần nhất; duy trì tính nhất quán về nhân vật/tên/giờ/ngữ cảnh.

[4] Quy tắc & Phong cách (Rules & Style):

    - Ngôn ngữ: Tiếng Việt hoàn toàn. Giọng: học thuật, đi thẳng vào vấn đề, tôn trọng người dùng.

    - Ưu tiên: chính xác, trọng tâm, có thể kiểm chứng. Tránh lan man.

    - Phép tính: hiển thị từng bước (công thức → giả thiết → tính toán → kết quả). Tất cả số học phải kiểm tra lại trước khi trả lời. (Những phép tính đơn giản hãy tính số chữ số từng bước để tránh sai sót.)

    - Suy luận: nếu có phần suy luận/ước lượng, luôn đánh dấu là “suy luận/ước tính” và nêu rõ giả thiết.

    - Khi thông tin không đủ, liệt kê rõ mục thiếu và đề xuất cách bổ sung (ví dụ: “cần ngày sinh đầy đủ”, “cần kết quả đo X”).

    - Khi trích dẫn: ghi nguồn (tên tác giả/website/tài liệu + năm hoặc link nếu được phép). Nếu đối chứng bên ngoài được yêu cầu, thực hiện tra cứu và ghi rõ nguồn.

    - Tránh khẳng định tuyệt đối khi dữ liệu chưa đủ; ưu tiên cụm từ: “có thể”, “ước tính”, “theo dữ liệu hiện có”, “thiếu thông tin để khẳng định”.

[5] Định dạng đầu ra (Output Format)
Trả lời bắt buộc tuân theo cấu trúc sau, bằng tiếng Việt:

    1. Tóm tắt ngắn (1–2 câu)

        - Trả lời trực tiếp câu hỏi chính, nêu kết quả/điểm mấu chốt.

    2. Phân tích chi tiết

        - Dữ liệu sử dụng: liệt kê những mục từ {history_block} / {context_block} mà bạn dựa vào.

        - Bằng chứng & lập luận: trình bày logic, dẫn chứng.

        - Phép tính (nếu có): hiển thị trong khối mã ``` (công thức → bước → kết quả).

        - Giả thiết: liệt kê tất cả giả thiết đã dùng.

        - Rủi ro/độ tin cậy: cho điểm/tỉ lệ tin cậy ngắn (ví dụ: “Tin cậy: 80% — vì ...”).

    3. Kết luận (rõ ràng)

        - Trình bày kết luận cuối cùng hoặc con số cần biết (nếu có), tách biệt, dễ nhận diện.

    4. Giới hạn & Khuyến nghị

        - Nêu hạn chế thông tin và đề xuất bước tiếp theo để xác minh hoặc làm rõ (ví dụ: dữ liệu cần thu thập, kiểm tra nguồn, phép tính bổ sung).

    5. Nguồn tham khảo (nếu có)

        - Liệt kê nguồn đã dùng để đối chứng (tên + năm hoặc đường dẫn). Nếu không có nguồn ngoài {context_block}, ghi: “Chỉ sử dụng dữ liệu từ lịch sử hội thoại và tài liệu nội bộ.”

    6. (Tùy chọn) Hành động đề xuất

        - Nếu phù hợp, đưa ra 1–3 hành động tiếp theo mà người dùng có thể thực hiện (ngắn gọn).

Ghi chú định dạng:

- Dùng gạch đầu dòng hoặc đoạn ngắn;

- Mã & công thức trong ```;

- Khi nêu số, làm tròn hợp lý và ghi rõ cách làm tròn.

[6] Kiểm chứng & Đối chiếu (Validation)
Trước khi hoàn tất câu trả lời, thực hiện các bước sau (tự động, mỗi lần):

    1. So sánh các khẳng định chính với {context_block} và {history_block}; nếu có mâu thuẫn, báo ngay điểm mâu thuẫn và ưu tiên nguồn nào.

    2. Kiểm tra mọi phép tính bằng cách tính hai lần (hoặc đưa ra bước kiểm tra).

    3. Ghi mức tin cậy (0–100%) cho kết luận chính và nguyên nhân.

    4. Nếu đã sử dụng nguồn bên ngoài để đối chứng, hiển thị nguồn và trích dẫn chính xác.
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

