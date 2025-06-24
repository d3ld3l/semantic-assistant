# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_match_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        results = semantic_search(query, df)

        # Добавляем точный поиск по коротким словам
        if len(query.strip()) <= 5:
            exact_matches = exact_match_search(query, df)
            # Добавляем точные совпадения в начало
            results = exact_matches + [r for r in results if r not in exact_matches]

        if results:
            st.markdown("### 🔍 Результаты поиска:")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено.")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
