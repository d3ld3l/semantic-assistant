import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

SYNONYMS = {
    "симка": "симкарта", "симки": "симкарта", "сим-карта": "симкарта", "сим-карты": "симкарта", "сим": "симка",
    "кредитка": "кредитная карта",
    "пэй": "пей":
    "заявление": "претензия",
    "несанкционированное списание": "несанкционированное снятие" ,
}

def apply_synonyms(text):
    for key, value in SYNONYMS.items():
        pattern = re.compile(rf'\b{re.escape(key)}\b', re.IGNORECASE)
        text = pattern.sub(value, text)
    return text

def preprocess(text):
    text = str(text).lower()
    text = apply_synonyms(text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

def semantic_search(query, df, top_k=5, threshold=0.45):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = []

    used_indices = set()
    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))
            used_indices.add(idx)

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    # ➕ Дополнительный точный поиск для коротких слов
    if len(query.strip()) <= 5:
        exact_matches = []
        for idx, row in df.iterrows():
            if idx in used_indices:
                continue
            if re.search(rf'\b{re.escape(query.lower())}\b', row['phrase'].lower()):
                exact_matches.append((0.999, row['phrase'], row['topics']))  # score = почти 1
        top_results.extend(exact_matches)

    return top_results
