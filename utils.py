import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
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

def load_synonyms(path):
    synonyms = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = [preprocess(p) for p in line.strip().split(",") if p.strip()]
            for word in parts:
                synonyms[word] = parts[0]
    return synonyms

def apply_synonyms(text, synonyms):
    words = text.split()
    replaced = [synonyms.get(w, w) for w in words]
    return " ".join(replaced)

def semantic_search(query, df, synonyms, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_syn = apply_synonyms(query_proc, synonyms)

    query_emb = model.encode(query_syn, convert_to_tensor=True)
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
    return results[:top_k]

def exact_word_match(query, df):
    query_proc = preprocess(query)
    if len(query_proc) > 5:
        return []
    
    matches = []
    for _, row in df.iterrows():
        words = row['phrase_proc'].split()
        if query_proc in words:
            matches.append((row['phrase'], row['topics']))
    return matches
