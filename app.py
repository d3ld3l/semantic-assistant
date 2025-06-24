
# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_match_results

st.set_page_config(page_title="Semantic Assistant")

st.title("Semantic Assistant")
st.write("Введите фразу, чтобы найти релевантные темы.")

@st.cache_data(show_spinner="Загружаю данные...")
def load_data():
    return load_all_excels()

try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

query = st.text_input("Введите фразу:")

if query:
    results = semantic_search(query, df)
    exacts = exact_match_results(query, df)

    if not results and not exacts:
        st.info("Ничего не найдено.")
    else:
        st.subheader("Результаты поиска:")

        for score, phrase, topics in results:
            st.markdown(f"**{phrase}** — {', '.join(topics)}")

        if exacts:
            st.markdown("---")
            st.subheader("Дополнительные совпадения по короткому слову:")
            for phrase, topics in exacts:
                st.markdown(f"`{phrase}` — {', '.join(topics)}`")
