# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_keyword_search

st.set_page_config(page_title="🔍 Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()

        # Семантический поиск
        semantic_results = semantic_search(query, df)

        # Точный поиск по короткому слову с учётом синонимов
        exact_results = exact_keyword_search(query, df)

        # Вывод результатов семантического поиска
        if semantic_results:
            st.markdown("### 🧠 Семантический поиск:")
            for score, phrase, topics in semantic_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.info("❗ Семантических совпадений не найдено.")

        # Вывод результатов точного поиска
        if exact_results:
            st.markdown("### 🎯 Точный поиск по ключевому слову:")
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        elif len(query.strip()) <= 5:
            st.info("❗ Точных совпадений по короткому слову не найдено.")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
