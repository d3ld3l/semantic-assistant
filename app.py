# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_match_results

st.set_page_config(page_title="Семантический помощник", layout="wide")

st.title("💬 Семантический помощник")
st.markdown("Введите запрос, и помощник найдёт подходящие фразы и темы.")

# Загрузка данных
@st.cache_data(show_spinner=True)
def load_data():
    return load_all_excels()

try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

# Интерфейс запроса
query = st.text_input("Ваш запрос:")

if query:
    st.markdown("### 🔎 Семантический поиск")
    results = semantic_search(query, df)

    if not results:
        st.warning("Ничего не найдено по смыслу.")
    else:
        for score, phrase, topics in results:
            st.markdown(f"✅ **{phrase}** — `{', '.join(topics)}` (сходство: **{score:.2f}**)")

    st.markdown("---")
    st.markdown("### 🎯 Точные вхождения по ключевым словам (дополнительно)")
    exact_matches = exact_match_results(query, df)

    if not exact_matches:
        st.info("Нет точных совпадений по словам.")
    else:
        for phrase, topics in exact_matches:
            st.markdown(f"🔸 <span style='color:#3366cc;font-weight:bold'>{phrase}</span> — `{', '.join(topics)}`", unsafe_allow_html=True)
