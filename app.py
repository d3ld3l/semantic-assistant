import streamlit as st
from utils import load_all_excels, semantic_search, exact_word_match, load_synonyms

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        synonyms = load_synonyms("synonyms.txt")
        results = semantic_search(query, df, synonyms)
        exacts = exact_word_match(query, df)

        if results:
            st.markdown("### 🔍 Семантический поиск (с учетом синонимов):")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено в семантическом поиске.")

        if exacts:
            st.markdown("---")
            st.markdown("### 🎯 Точный поиск по короткому слову:")
            for phrase, topics in exacts:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
    except Exception as e:
        st.error(f"Ошибка: {e}")
