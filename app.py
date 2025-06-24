# app.py
import streamlit as st
from utils import load_data, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="wide")
st.title("🧠 Semantic Assistant")

uploaded_file = st.file_uploader("Загрузите Excel-файл с базой", type=[".xlsx"])

if uploaded_file:
    df, embeddings = load_data(uploaded_file)

    query = st.text_input("Введите ваш запрос:")

    if query:
        semantic_results, keyword_matches = semantic_search(query, df, embeddings)

        if semantic_results:
            st.markdown("### 🔍 Семантические результаты:")
            for item in semantic_results:
                st.markdown(f"- **{item['text']}** → {item['label']} (_{item['code']}_) ({item['score']:.2f})")

        if keyword_matches:
            st.markdown("### 🧩 Ключевые совпадения (точные слова):")
            for item in keyword_matches:
                st.markdown(f"- **{item['text']}** → {item['label']} (_{item['code']}_)")
else:
    st.info("Пожалуйста, загрузите файл Excel.")
