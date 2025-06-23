# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🔍 Результаты поиска:")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено.")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
