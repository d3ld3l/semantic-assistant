import streamlit as st
from utils import load_all_excels, semantic_search, exact_keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        results = semantic_search(query, df)

        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено.")

        # Ключевой поиск
        st.markdown("---")
        st.markdown("### 🧷 Результаты точного поиска по ключевому слову:")
        key_results = exact_keyword_search(query, df)
        if key_results:
            for phrase, topics in key_results:
                st.markdown(f"- {phrase} → {', '.join(topics)}")
        else:
            st.info("Ключевые слова не дали результатов.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
