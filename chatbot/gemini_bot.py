import os

# Suppress noisy gRPC/ALTS logs (must be set before importing google libs)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

try:
    # prefer the newer official client package; do NOT instantiate a Client
    # at import time (instantiation may fail due to environment TLS/ssl issues).
    import google.genai as _genai_pkg  # type: ignore
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
            raise RuntimeError("google-genai package not installed. Install with: pip install google-genai")
        # Instantiate a Client now (at runtime) so import-time failures are avoided.
        try:
            # client is provided under google.genai.client.Client
            genai_client = _genai_pkg.client.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize google-genai Client: {e}")
        # set API key if available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai_client.api_key = api_key
            except Exception:
                # not all client versions expose api_key attribute; ignore
                pass
        # Using the google.genai client: create a small adapter for the
        # project's existing call sites. We store the client and model
        # name here and call the client's generate/chat endpoints in ask().
        self._genai_client = genai_client
        self._genai_model = model
        # keep placeholders for compatibility with older code
        self.model = None
        self.chat = None

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
        # Use google-genai to generate a response from the model
        # Try the high-level embed/chat API first
        try:
            # some versions expose client.chat.completions.create or client.models.generate
            # We'll try a generic models.generate if available.
            client = self._genai_client
            # The exact call shape can vary; try the standard generate API
            res = None
            try:
                res = client.models.generate(model=self._genai_model, text=prompt)
            except Exception:
                # fallback: try chat completions API
                try:
                    res = client.chat.completions.create(model=self._genai_model, messages=[{"role": "user", "content": prompt}])
                except Exception as e:
                    raise

            # Extract text from probable response shapes
            if res is None:
                raise RuntimeError("No response from google-genai client")

            # Try common attribute names
            if hasattr(res, "text"):
                answer = res.text.strip()
            elif hasattr(res, "output") and isinstance(res.output, list) and len(res.output) > 0:
                # some responses include output list
                answer = getattr(res.output[0], "content", "").strip() or str(res.output[0]).strip()
            else:
                # fallback to string conversion
                answer = str(res).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to get response from google-genai client: {e}")
        self.memory.add_message("user", message)
        self.memory.add_message("assistant", answer)
        return answer

    def clear_context(self):
        self.memory.clear_history()
        # No persistent chat object when using the `google-genai` client
        # in this simplified adapter; just clear conversation memory.
        self.chat = None
        return "🧹 Bộ nhớ hội thoại đã được xóa!"

