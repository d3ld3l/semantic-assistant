import streamlit as st
from utils import load_all_excels, semantic_search, exact_word_match, load_synonym_groups

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        synonym_groups = load_synonym_groups()

        sem_results = semantic_search(query, df, synonym_groups)
        exact_results = exact_word_match(query, df, synonym_groups)

        if sem_results:
            st.markdown("### 🔎 Семантические результаты:")
            for score, phrase, topics in sem_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.info("Семантических совпадений не найдено.")

        if exact_results:
            st.markdown("---")
            st.markdown("### 🎯 Точный поиск по ключевым словам:")
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
    except Exception as e:
        st.error(f"Ошибка: {e}")
