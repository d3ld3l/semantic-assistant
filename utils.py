# utils.py
import pandas as pd
import requests
import re
import json
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

def load_synonyms(file_path="synonyms.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_synonyms = json.load(f)
    norm_map = {}
    for group in raw_synonyms:
        root = group[0]
        for word in group:
            norm_map[word.lower()] = root.lower()
    return norm_map

def normalize_text(text, synonyms_map):
    words = re.sub(r"[^\w\s-]", "", text.lower()).split()
    return " ".join([synonyms_map.get(word, word) for word in words])

def preprocess(text, synonyms_map):
    return normalize_text(text.strip(), synonyms_map)

def load_excel(url, synonyms_map):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df = df[["phrase"] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(lambda x: preprocess(x, synonyms_map))
    return df[['phrase', 'phrase_proc', 'topics']]

def load_all_excels(synonyms_map):
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url, synonyms_map)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

def exact_word_match(query, df, synonyms_map, max_len=5):
    norm_query = normalize_text(query, synonyms_map)
    if len(norm_query) > max_len:
        return []
    exacts = []
    for _, row in df.iterrows():
        if norm_query in row['phrase_proc']:
            exacts.append((0.99, row['phrase'], row['topics']))
    return exacts

def semantic_search(query, df, synonyms_map, top_k=5, threshold=0.5):
    norm_query = normalize_text(query, synonyms_map)
    query_emb = model.encode(norm_query, convert_to_tensor=True)
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
    results = results[:top_k]

    # Добавляем точные совпадения
    exacts = exact_word_match(query, df, synonyms_map)
    return results + exacts
