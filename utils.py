# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import torch

# Обновленная, более умная модель
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Нормализация фразы
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Загрузка одного Excel-файла и предобработка
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")

    df = pd.read_excel(BytesIO(response.content))
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df = df[['phrase'] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    df['embedding'] = df['phrase_proc'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df[['phrase', 'phrase_proc', 'topics', 'embedding']]

# Загрузка всех файлов и объединение
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# Поиск
def semantic_search(query, df, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)

    scores = []
    for idx, row in df.iterrows():
        sim = util.pytorch_cos_sim(query_emb, row['embedding'])[0][0].item()
        if sim >= threshold:
            scores.append((sim, row['phrase'], row['topics']))

    scores.sort(key=lambda x: x[0], reverse=True)
