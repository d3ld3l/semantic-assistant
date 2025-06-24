import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        df = load_all_excels()
        results = semantic_search(query, df)

        if results:
            st.markdown("### 🔍 Результаты поиска:")
            for score, phrase, topics in results:
                st.markdown(
                    f'<div style="padding: 6px; background-color: #f4f4f4; border-radius: 10px; margin-bottom: 8px;">'
                    f'<b style="color: #2a7bde">{phrase}</b><br>'
                    f'<small>Темы: {", ".join(topics)} | Совпадение: {score:.2f}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("Совпадений не найдено.")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
