# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_match_results
import time

st.set_page_config(page_title="🔍 Semantic Assistant", layout="wide")

st.title("🔍 Semantic Assistant")
st.markdown("Введите запрос, и помощник найдет релевантные темы из Excel-файлов.")

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
    with st.spinner("🔎 Ищу похожие фразы..."):
        time.sleep(0.3)  # задержка для UX
        results = semantic_search(query, df)
        exact_matches = exact_match_results(query, df)

    if not results and not exact_matches:
        st.warning("Ничего не найдено.")
    else:
        st.markdown("### 🔹 Результаты семантического поиска:")
        for score, phrase, topics in results:
            phrase_html = f"<b style='color: #005bbb;'>{phrase}</b>"
            topics_html = ", ".join(topics)
            st.markdown(f"• {phrase_html} — <span style='color:gray'>{topics_html}</span>", unsafe_allow_html=True)

        if exact_matches:
            st.markdown("### 🔸 Дополнительные точные совпадения по короткому слову:")
            for phrase, topics in exact_matches:
                phrase_html = f"<b style='color: #d93f0b;'>{phrase}</b>"
                topics_html = ", ".join(topics)
                st.markdown(f"• {phrase_html} — <span style='color:gray'>{topics_html}</span>", unsafe_allow_html=True)
