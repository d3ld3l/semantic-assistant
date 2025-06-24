# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Загружаем модель
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Ссылки на файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Словарь синонимов (ручной)
SYNONYM_DICT = {
    "симка": "симкарта",
    "кредитка": "кредитная карта",
    "пэй": "pay",
    "теле2": "tele2",
}

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    for short, full in SYNONYM_DICT.items():
        text = text.replace(short, full)
    return text

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
    return df[['phrase', 'phrase_proc', 'topics']]

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

def semantic_search(query, df, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = []

    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    # 🔍 Дополнительно — точные вхождения короткого слова (до 8 символов)
    if len(query.strip().split()) == 1 and len(query.strip()) <= 8:
        exact_matches = df[df['phrase_proc'].str.contains(rf'\b{re.escape(query_proc)}\b', regex=True)]
        for _, row in exact_matches.iterrows():
            if row['phrase'] not in [r[1] for r in top_results]:
                top_results.append((0.0, row['phrase'], row['topics']))  # 0.0 — чтобы показать, что это не семантический

    return top_results
