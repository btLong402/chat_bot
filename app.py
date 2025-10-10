import streamlit as st
from chatbot.gemini_bot import GeminiBot

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="📚")

st.title("📚 Gemini RAG Chatbot")
st.markdown("Chatbot này có thể nhớ context **và** tìm thông tin trong tài liệu PDF bạn tải lên.")

bot = st.session_state.get("bot", GeminiBot())
st.session_state.bot = bot

# --- Upload file ---
uploaded = st.sidebar.file_uploader("📄 Tải lên file PDF tri thức", type=["pdf"])
if uploaded:
    temp_path = f"data/docs/{uploaded.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())
    bot.retriever.add_documents(temp_path)
    st.sidebar.success(f"✅ Đã nạp file: {uploaded.name}")

if st.sidebar.button("🧹 Xoá bộ nhớ hội thoại"):
    bot.clear_context()
    st.sidebar.success("Đã xoá context!")

# --- Hiển thị hội thoại ---
for msg in bot.memory.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Hỏi Gemini về tài liệu hoặc bất cứ điều gì...")
if user_input:
    st.chat_message("user").markdown(user_input)
    reply = bot.ask(user_input, use_rag=True)
    st.chat_message("assistant").markdown(reply)
