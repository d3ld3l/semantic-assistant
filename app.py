# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Семантический помощник", layout="wide")
st.title("🤖 Семантический ассистент")

# Загрузка базы данных
@st.cache_data
def load_data():
    return load_all_excels()

df = load_data()

# Ввод запроса
query = st.text_input("Введите запрос:")

if query:
    with st.spinner("Анализируем..."):
        results = semantic_search(query, df)

    if not results:
        st.warning("Ничего не найдено 😕")
    else:
        st.success("Найдено совпадений: {}".format(len(results)))
        for i, (score, phrase, topics) in enumerate(results):
            color = "#d1e7dd" if i == 0 else "#f8f9fa"
            st.markdown(
                f"""
                <div style="background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px;">
                <strong>Фраза:</strong> {phrase}<br>
                <strong>Совпадение:</strong> {score:.2f}<br>
                <strong>Темы:</strong> {", ".join(topics)}
                </div>
                """,
                unsafe_allow_html=True
            )
