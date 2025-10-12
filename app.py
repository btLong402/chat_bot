import os
from pathlib import Path
from typing import List

import streamlit as st

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

st.title("📚 Gemini RAG Chatbot")
st.markdown("Chatbot này có thể nhớ context **và** tìm thông tin trong tài liệu PDF bạn tải lên.")

history_dir = "data/history"
docs_dir = Path("data/docs")
default_memory_file = os.path.join(history_dir, "chat_history.json")


def list_uploaded_pdfs(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted([file.name for file in directory.glob("*.pdf")])


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

    uploaded = st.file_uploader("📄 Tải lên file PDF tri thức", type=["pdf"])

    existing_docs = list_uploaded_pdfs(docs_dir)
    if existing_docs:
        st.markdown("**📂 Bộ tài liệu hiện có**")
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

if uploaded:
    temp_path = docs_dir / uploaded.name
    os.makedirs(temp_path.parent, exist_ok=True)
    with st.spinner("Đang xử lý tài liệu..."):
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        bot.retriever.add_documents(str(temp_path))
    st.sidebar.success(f"✅ Đã nạp file: {uploaded.name}")


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


user_input = st.chat_input("Hỏi Gemini về tài liệu hoặc bất cứ điều gì...")
if user_input:
    st.chat_message("user").markdown(user_input)
    reply = bot.ask(user_input, use_rag=True)
    st.chat_message("assistant").markdown(reply)
