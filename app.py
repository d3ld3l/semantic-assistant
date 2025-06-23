# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Семантический помощник", layout="wide")

st.title("🧠 Семантический ассистент")
st.markdown("Введите запрос, и помощник подберёт соответствующие фразы и темы.")

# Загрузка данных
try:
    df = load_all_excels()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

query = st.text_input("🔎 Введите запрос:")

if query:
    with st.spinner("Ищем совпадения..."):
        results = semantic_search(query, df)

    if not results:
        st.warning("Ничего не найдено.")
    else:
        st.markdown("### 📋 Результаты:")

        for score, phrase, topics in results:
            highlight = "💡" if score >= 0.8 else ""
            color = "#d1ffd6" if score >= 0.8 else "#f0f0f0" if score > 0 else "#e0e0ff"
            score_label = f"**Точность:** {round(score, 3)}" if score > 0 else "*Совпадение по слову*"
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px;'>"
                f"<b>{highlight} Фраза:</b> {phrase}<br>"
                f"<b>Темы:</b> {', '.join(topics)}<br>"
                f"{score_label}</div>",
                unsafe_allow_html=True
            )
