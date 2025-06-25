import streamlit as st
from utils import load_all_excels, semantic_search, exact_keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()

        # Семантический поиск
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🔍 Результаты семантического поиска:")
            for score, phrase, topics in results:
                st.markdown(f"- **{phrase}** → {', '.join(topics)} (_{score:.2f}_)")
        else:
            st.warning("Совпадений не найдено семантически.")

        # Точный поиск по ключевому слову
        if len(query.strip()) <= 5:
            st.markdown("### 🎯 Результаты точного поиска по ключевому слову:")
            exact_matches = exact_keyword_search(query, df)
            if exact_matches:
                for phrase, topics in exact_matches:
                    st.markdown(f"- **{phrase}** → {', '.join(topics)}")
            else:
                st.warning("Нет точных совпадений по ключевому слову.")
        else:
            st.markdown("_Для точного поиска введите короткое ключевое слово (до 5 символов)._")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
