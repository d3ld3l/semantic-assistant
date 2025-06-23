# utils.py
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # Более точная и быстрая модель

@st.cache_data
def load_excel_from_github(url):
    response = requests.get(url)
    df = pd.read_excel(BytesIO(response.content))
    df = df.rename(columns=lambda x: x.strip().lower())
    df = df[['phrase', 'topics1', 'topics2', 'topics3', 'topics4', 'topics5', 'topics6']]
    df = df.fillna("")
    df['topics'] = df[[f'topics{i}' for i in range(1, 7)]].agg(", ".join, axis=1).str.strip(', ')
    df['phrase_clean'] = df['phrase'].str.lower().str.strip()
    return df[['phrase', 'topics', 'phrase_clean']]

@st.cache_data
def load_all_excels(model):
    urls = [
        "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
        "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
        "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
    ]
    dfs = [load_excel_from_github(url) for url in urls]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['embedding'] = full_df['phrase_clean'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return full_df

def semantic_search(query, df, model, top_k=5):
    query_embedding = model.encode(query.lower(), convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, list(df['embedding']))[0]
    top_indices = np.argsort(-scores.cpu().numpy())[:top_k]
    return df.iloc[top_indices].copy()
