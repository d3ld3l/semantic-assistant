import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

def load_synonym_groups(filepath="synonyms.txt"):
    groups = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = [preprocess(w) for w in line.strip().split(",") if w]
                if words:
                    groups.append(set(words))
    except Exception as e:
        print(f"⚠️ Ошибка загрузки синонимов: {e}")
    return groups

def expand_with_synonyms(text, synonym_groups):
    tokens = preprocess(text).split()
    expanded = set(tokens)
    for group in synonym_groups:
        if any(t in group for t in tokens):
            expanded.update(group)
    return expanded

def semantic_search(query, df, synonym_groups, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    synonym_expanded = " ".join(expand_with_synonyms(query_proc, synonym_groups))
    query_emb = model.encode(synonym_expanded, convert_to_tensor=True)
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

def exact_word_match(query, df, synonym_groups):
    query_tokens = expand_with_synonyms(query, synonym_groups)
    if len(query.strip()) > 5:
        return []
    matches = []
    for _, row in df.iterrows():
        phrase_tokens = set(row['phrase_proc'].split())
        if any(token in phrase_tokens for token in query_tokens):
            matches.append((row['phrase'], row['topics']))
    return matches
