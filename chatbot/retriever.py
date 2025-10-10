import os
import pickle
import logging
# heavy/optional imports (faiss, numpy, langchain) are imported lazily inside
# the methods that need them so importing this module doesn't fail in
# minimal environments.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RAGRetriever:
    def __init__(self, vector_store_path="data/vector_store.pkl", embed_batch_size=16):
        self.vector_store_path = vector_store_path
        self.index = None
        self.docs = []
        self.embed_batch_size = embed_batch_size

        # embedding client placeholders
        self._embed_client = None
        self._embed_type = None
        self._embed_model = None
        self._init_embed_client()

        self.load_index()

    def _init_embed_client(self):
        """
        Detect and initialize an embedding client once.
        Supports:
          - google.genai (new client) -> client.models.embed_content(...)
          - google.generativeai (older wrapper) -> gga.embeddings.create(...)
        Sets self._embed_type to 'genai' or 'gga' and stores client object.
        """
        # Try new google.genai
        try:
            from google.genai import client as genai_client
            from google.genai import types
            client = genai_client.Client()
            # prefer Gemini embedding model if available
            self._embed_client = client
            self._embed_type = "genai"
            self._embed_model = "gemini-embedding-001"
            logger.info("Using google.genai embedding client")
            return
        except Exception:
            pass

        # No older-wrapper fallback: only support the official `google-genai` client.
        # If the new client wasn't found above, mark the embed client as not available
        # and provide a clear install message for the user.
        self._embed_client = None
        self._embed_type = None
        logger.warning("No embedding client initialized. Install `google-genai` (pip install google-genai)")

    def load_index(self):
        if os.path.exists(self.vector_store_path):
            with open(self.vector_store_path, "rb") as f:
                self.index, self.docs = pickle.load(f)
            logger.info(f"Loaded vector store: {len(self.docs)} docs")
        else:
            self.index = None
            self.docs = []

    def save_index(self):
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        with open(self.vector_store_path, "wb") as f:
            pickle.dump((self.index, self.docs), f)
        logger.info("Vector store saved")

    def load_pdf(self, pdf_path):
        # import PDF reader lazily and give clear instruction if missing
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader
            except Exception:
                raise RuntimeError(
                    "PDF parser not installed. Install with:\n"
                    "  pip install pypdf\n"
                )

        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def _batch(self, iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def _embed_batch(self, texts):
        """
        Embed a list of strings and return numpy array shape (n, dim) dtype float32.
        Uses the detected client and batches requests.
        """
        if not texts:
            # Lazy import numpy only when needed
            try:
                import numpy as np
            except Exception:
                raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")
            return np.zeros((0, 0), dtype="float32")

        if self._embed_type is None or self._embed_client is None:
            raise RuntimeError("Embedding client not initialized. Install and configure API key.")

        vectors = []
        for batch in self._batch(texts, self.embed_batch_size):
            if self._embed_type == "genai":
                # google.genai client
                client = self._embed_client
                from google.genai import types  # local import to avoid module import errors earlier
                cfg = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                res = client.models.embed_content(model=self._embed_model, contents=batch, config=cfg)
                try:
                    import numpy as np
                except Exception:
                    raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")
                for emb in res.embeddings:
                    vectors.append(np.array(emb.values, dtype="float32"))
            elif self._embed_type == "gga":
                # google.generativeai wrapper (no longer recommended)
                gga = self._embed_client
                try:
                    import numpy as np
                except Exception:
                    raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")
                for text in batch:
                    resp = gga.embeddings.create(model=self._embed_model, input=text)
                    vectors.append(np.array(resp.data[0].embedding, dtype="float32"))
            else:
                raise RuntimeError("Unsupported embed client type")
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")
        return np.vstack(vectors)

    def embed_texts(self, texts):
        """
        Public wrapper: accepts single string or list of strings.
        """
        contents = texts if isinstance(texts, list) else [texts]
        return self._embed_batch(contents)

    def add_documents(self, file_path):
        # Đọc PDF
        text = self.load_pdf(file_path)
        if not text:
            logger.warning("No text extracted from PDF")
            return

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except Exception:
            raise RuntimeError("langchain is required for text splitting; install with: pip install 'langchain'")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks produced from document")
            return

        # Tạo vector embedding (batched)
        vectors = self.embed_texts(chunks)
        if vectors.size == 0:
            raise RuntimeError("Embedding produced no vectors")

        # Ensure vectors are C-contiguous float32
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for embeddings; install with: pip install numpy")

        vectors = np.ascontiguousarray(vectors, dtype="float32")

        # Create new index if needed or if dimension mismatch
        dim = vectors.shape[1]
        if self.index is None:
            try:
                import faiss
            except Exception:
                raise RuntimeError("faiss is required for vector indexing; install with: pip install faiss-cpu")
            self.index = faiss.IndexFlatL2(dim)
            logger.info(f"Created new FAISS index dim={dim}")
        else:
            # try to infer existing index dim
            try:
                existing_dim = self.index.d
            except Exception:
                # fallback: rebuild
                existing_dim = None

            if existing_dim != dim:
                logger.info(f"Index dimension mismatch (existing={existing_dim}, new={dim}), rebuilding index")
                # rebuild from stored docs if any (clear index and re-add)
                self.index = faiss.IndexFlatL2(dim)

        # add to index and docs
        self.index.add(vectors)
        self.docs.extend(chunks)
        self.save_index()
        logger.info(f"Added {len(chunks)} chunks to index")

    def retrieve(self, query, top_k=3):
        if self.index is None or len(self.docs) == 0:
            return []
        q_vecs = self.embed_texts(query)
        if q_vecs.size == 0:
            return []
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("numpy is required for retrieval; install with: pip install numpy")

        q_vec = np.ascontiguousarray(q_vecs[0].reshape(1, -1), dtype="float32")
        top_k = min(top_k, len(self.docs))
        D, I = self.index.search(q_vec, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs) and idx >= 0:
                results.append(self.docs[idx])
        return results
