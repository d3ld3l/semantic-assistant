import streamlit as st
from utils import load_all_excels, semantic_search, exact_keyword_search, build_keyword_index

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        keyword_index = build_keyword_index(df)

        results = semantic_search(query, df)
        exact_results = exact_keyword_search(query, keyword_index)

        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено.")

        st.markdown("---")
        st.markdown("### 🖋 Результаты точного поиска по ключевому слову:")
        if exact_results:
            for phrase, topics in exact_results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)}")
        else:
            st.info("Ключевые слова не дали результатов.")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
