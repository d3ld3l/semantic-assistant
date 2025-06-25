import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from nltk.stem.snowball import SnowballStemmer

# Синонимы (группы)
SYNONYM_GROUPS = [
    ["сим", "симка", "сим-карта", "симкарта", "симку", "симки", "симке", "симкарте"],
    ["кредитка", "кредитная карта", "кредитную карту", "кредитной картой"],
    ["наличка", "наличные", "наличными", "наличные деньги"]
]

def build_synonym_dict(groups):
    synonym_dict = {}
    for group in groups:
        for word in group:
            synonym_dict[word] = set(group)
    return synonym_dict

SYNONYM_DICT = build_synonym_dict(SYNONYM_GROUPS)

# Модель для семантики
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Ссылки на файлы Excel
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

stemmer = SnowballStemmer("russian")

def normalize_synonyms(text):
    words = text.split()
    normalized = []
    for word in words:
        stem = stemmer.stem(word)
        for key, synonyms in SYNONYM_DICT.items():
            if stem in map(stemmer.stem, synonyms):
                normalized.append(list(synonyms)[0])  # взять первое как канон
                break
        else:
            normalized.append(word)
    return ' '.join(normalized)

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    text = normalize_synonyms(text)
    return text

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    # Разбивка по /, если есть
    expanded_rows = []
    for _, row in df.iterrows():
        phrases = str(row['phrase']).split("/")
        for phrase in phrases:
            entry = {"phrase": phrase.strip()}
            for col in topic_cols:
                entry[col] = row[col]
            expanded_rows.append(entry)
    df_expanded = pd.DataFrame(expanded_rows)

    df_expanded['topics'] = df_expanded[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df_expanded['phrase_proc'] = df_expanded['phrase'].apply(preprocess)
    return df_expanded[['phrase', 'phrase_proc', 'topics']]

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
    return results[:top_k]

def keyword_search(query, df):
    query_proc = preprocess(query)
    if len(query_proc) > 5:
        return []

    matches = []
    for _, row in df.iterrows():
        words = row['phrase_proc'].split()
        if any(stemmer.stem(query_proc) == stemmer.stem(w) for w in words):
            matches.append((row['phrase'], row['topics']))
    return matches
