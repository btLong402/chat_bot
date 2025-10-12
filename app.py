import hashlib
import json
import os
from pathlib import Path
from typing import List

import streamlit as st
from streamlit.components.v1 import html as components_html

from chatbot.gemini_bot import GeminiBot

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .status-card {
        background-color: #f5f7ff;
        border: 1px solid #eef0ff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(15, 17, 30, 0.05);
    }
    .status-card h3 {
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .status-card p {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }
    .uploaded-list {
        border: 1px solid #e6e6ef;
        border-radius: 8px;
        padding: 8px 12px;
        background-color: #fafaff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 La Bàn AI")
st.markdown("Chatbot này có thể nhớ context **và** tìm thông tin trong tài liệu PDF bạn tải lên.")

history_dir = "data/history"
docs_dir = Path("data/docs")
default_memory_file = os.path.join(history_dir, "chat_history.json")


def list_uploaded_pdfs(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted([file.name for file in directory.glob("*.pdf")])


def render_copy_button(content: str) -> None:
    counter = st.session_state.setdefault("_copy_btn_counter", 0)
    st.session_state["_copy_btn_counter"] += 1
    unique_id = f"copy-btn-{counter}-{hashlib.md5(content.encode('utf-8')).hexdigest()}"
    safe_content = json.dumps(content)
    components_html(
        f"""
        <style>
        .copy-wrapper {{
            display: flex;
            justify-content: flex-end;
            margin-top: 4px;
        }}
        .copy-btn {{
            background-color: #eef0ff;
            border: 1px solid #d4d8ff;
            border-radius: 6px;
            padding: 4px 10px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: background-color 0.2s ease;
        }}
        .copy-btn:hover {{
            background-color: #dfe3ff;
        }}
        </style>
        <div class='copy-wrapper'>
            <button class='copy-btn' id='{unique_id}'>📋 Copy</button>
        </div>
        <script>
        (function() {{
            const btn = document.getElementById('{unique_id}');
            if (!btn) return;
            const text = {safe_content};
            btn.addEventListener('click', async () => {{
                const original = btn.innerText;
                const writeClipboard = async (value) => {{
                    if (navigator.clipboard && window.isSecureContext) {{
                        await navigator.clipboard.writeText(value);
                        return;
                    }}
                    const textarea = document.createElement('textarea');
                    textarea.value = value;
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                }};
                try {{
                    await writeClipboard(text);
                    btn.innerText = '✅ Đã copy';
                }} catch (err) {{
                    console.error(err);
                    btn.innerText = '❌ Lỗi';
                }}
                setTimeout(() => (btn.innerText = original), 2000);
            }});
        }})();
        </script>
        """,
        height=70,
    )


with st.sidebar:
    st.header("⚙️ Tuỳ chỉnh")
    user_id = st.text_input(
        "👤 Mã người dùng",
        value=st.session_state.get("user_id", ""),
        help="Mỗi người dùng có một lịch sử hội thoại riêng.",
    )
    st.session_state.user_id = user_id

    if st.button("🧹 Xoá bộ nhớ hội thoại", use_container_width=True):
        bot = st.session_state.get("bot")
        if bot:
            bot.clear_context()
            st.success("Đã xoá context!")

    with st.expander("📘 Hướng dẫn nhanh", expanded=False):
        st.markdown(
            """
            - Nhập mã người dùng để tải đúng lịch sử hội thoại.
            - Tải lên tài liệu PDF để kích hoạt chế độ RAG.
            - Đặt câu hỏi cụ thể để nhận câu trả lời sát nội dung tài liệu.
            - Dùng nút xoá bộ nhớ khi muốn trò chuyện từ đầu.
            """
        )

    uploaded_files = st.file_uploader("📄 Tải lên file PDF tri thức", type=["pdf"], accept_multiple_files=True)

    existing_docs = list_uploaded_pdfs(docs_dir)
    if existing_docs:
        with st.expander("📂 Bộ tài liệu hiện có", expanded=False):
            st.markdown("<div class='uploaded-list'>" + "<br>".join(existing_docs) + "</div>", unsafe_allow_html=True)
    else:
        st.caption("Chưa có tài liệu nào được tải lên.")


memory_file = (
    os.path.join(history_dir, f"chat_history_{user_id}.json")
    if user_id
    else default_memory_file
)

if "bot" not in st.session_state or st.session_state.get("bot_memory_file") != memory_file:
    st.session_state.bot = GeminiBot(memory_file=memory_file)
    st.session_state.bot_memory_file = memory_file

bot = st.session_state.bot

if uploaded_files:
    processed_files = []
    os.makedirs(docs_dir, exist_ok=True)
    with st.spinner("Đang xử lý tài liệu..."):
        for uploaded in uploaded_files:
            temp_path = docs_dir / uploaded.name
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            bot.retriever.add_documents(str(temp_path))
            processed_files.append(uploaded.name)
    if processed_files:
        st.sidebar.success("✅ Đã nạp file: " + ", ".join(processed_files))


total_turns = len(bot.memory.history)
total_pairs = total_turns // 2
doc_chunks = len(getattr(bot.retriever, "docs", []))
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='status-card'><h3>📥 Lượt tương tác</h3><p>{}</p></div>".format(total_turns), unsafe_allow_html=True)
with col2:
    st.markdown("<div class='status-card'><h3>🗂️ Số lượt hỏi đáp</h3><p>{}</p></div>".format(total_pairs), unsafe_allow_html=True)
with col3:
    st.markdown(
        "<div class='status-card'><h3>📑 Số đoạn tri thức</h3><p>{}</p></div>".format(doc_chunks),
        unsafe_allow_html=True,
    )

st.divider()

chat_container = st.container()

with chat_container:
    for msg in bot.memory.history:
        with st.chat_message(msg["role"]):
            timestamp = msg.get("time")
            if timestamp:
                st.caption(timestamp)
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_copy_button(msg["content"])


user_input = st.chat_input("Hỏi La Bàn về tài liệu hoặc bất cứ điều gì...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    reply = bot.ask(user_input, use_rag=True)
    with st.chat_message("assistant"):
        st.markdown(reply)
        render_copy_button(reply)
