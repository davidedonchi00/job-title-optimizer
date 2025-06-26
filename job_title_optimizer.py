
"""Job Title Optimizer – Streamlit app
======================================

INPUT:
    • Job Title (come lo scrive l'azienda)
    • Job Description completa

OUTPUT:
    • Elenco di job title alternativi con:
        - Volume medio ricerche Google (Google Trends, scala 0‑100)
        - Numero annunci Indeed e LinkedIn (via JSearch API, RapidAPI)
        - Pertinenza semantica rispetto alla JD
        - Score finale (0‑1) calcolato 50% Google + 25% Indeed + 15% LinkedIn + 10% pertinenza

Dipendenze (requirements.txt):
    streamlit
    pandas
    pytrends
    spacy
    scikit-learn
    requests

Per usare spaCy:
    python -m spacy download it_core_news_sm

Per usare Indeed/LinkedIn:
    • crea un account gratuito su RapidAPI
    • sottoscrivi l'API "JSearch"
    • imposta la chiave in un file .streamlit/secrets.toml, es:

        [api_keys]
        rapidapi = "LA_TUA_CHIAVE"

    oppure come variabile d'ambiente RAPIDAPI_KEY
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
import spacy
import streamlit as st
from pytrends.request import TrendReq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# CONFIG STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Job Title Optimizer", page_icon="🔍", layout="wide")

# ------------------------------------------------------------------------------
# CARICA spaCy una sola volta (cache)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_spacy():
    return spacy.load("it_core_news_sm")

nlp = load_spacy()

# ------------------------------------------------------------------------------
# Dizionario di sinonimi di base (puoi estenderlo caricando un CSV)
# ------------------------------------------------------------------------------
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "operatore": ["addetto", "tecnico", "manovale"],
    "carrello": ["muletto", "forklift"],
    "magazzino": ["logistica", "deposito"],
    "macchine": ["impianti", "linea"],
    "saldatore": ["saldobrasatore", "tubista"],
    "impianto": ["plant", "stabilimento"],
}

# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------
def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^a-zà-ù0-9 ]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def extract_keywords(jd: str, top_n: int = 10) -> List[str]:
    doc = nlp(jd)
    tokens = [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "PROPN"} and len(t) > 2]
    freq = pd.Series(tokens).value_counts()
    return list(freq.head(top_n).index)

def generate_titles(original: str, keywords: List[str]) -> List[str]:
    words = original.lower().split()
    candidates: set[str] = set()

    # sostituzione sinonimi
    for i, w in enumerate(words):
        if w in DEFAULT_SYNONYMS:
            for syn in DEFAULT_SYNONYMS[w]:
                new_words = words.copy()
                new_words[i] = syn
                candidates.add(" ".join(new_words))

    # combinazioni con keyword JD
    for kw in keywords:
        candidates.add(f"{words[0]} {kw}")
        candidates.add(f"{kw} {words[-1]}")
    for base in ["operatore", "addetto", "tecnico", "manutentore"]:
        for kw in keywords[:5]:
            candidates.add(f"{base} {kw}")

    # pulizia
    return sorted({clean_text(c).title() for c in candidates if 2 <= len(c.split()) <= 4})

# ------------------------------------------------------------------------------
# Google Trends
# ------------------------------------------------------------------------------
def google_trends_scores(titles: List[str]) -> Dict[str, int]:
    trend = TrendReq(hl="it-IT", tz=360)
    result: Dict[str, int] = {}
    # batch di 5 per evitare limitazioni API
    for i in range(0, len(titles), 5):
        batch = titles[i:i+5]
        try:
            trend.build_payload(batch, timeframe="today 12-m", geo="IT")
            df = trend.interest_over_time()
            if not df.empty:
                for t in batch:
                    result[t] = int(df[t].mean())
        except Exception:
            for t in batch:
                result[t] = 0
    return result

# ------------------------------------------------------------------------------
# Indeed & LinkedIn via JSearch API
# ------------------------------------------------------------------------------
def job_board_counts(titles: List[str]) -> Dict[str, Dict[str, int]]:
    api_key = st.secrets.get("api_keys", {}).get("rapidapi") or os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return {t: {"indeed": 0, "linkedin": 0} for t in titles}

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    base_url = "https://jsearch.p.rapidapi.com/search"

    out: Dict[str, Dict[str, int]] = {}
    for title in titles:
        params = {"query": title, "page": "1", "num_pages": "1", "date_posted": "all", "country": "it"}
        try:
            r = requests.get(base_url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            indeed = sum(1 for d in data if d.get("source") == "Indeed")
            linkedin = sum(1 for d in data if d.get("source") == "LinkedIn")
            out[title] = {"indeed": indeed, "linkedin": linkedin}
        except Exception:
            out[title] = {"indeed": 0, "linkedin": 0}
    return out

# ------------------------------------------------------------------------------
# Pertinenza semantica
# ------------------------------------------------------------------------------
def jd_similarity(title: str, jd: str) -> float:
    vect = TfidfVectorizer().fit([jd, title])
    tfidf = vect.transform([jd, title])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

# ------------------------------------------------------------------------------
# INTERFACCIA STREAMLIT
# ------------------------------------------------------------------------------
st.title("🔍 Job Title Optimizer (Blue‑Collar)")

with st.form("form"):
    st.subheader("Inserisci i dati di partenza:")
    job_title_input = st.text_input("Job title attuale", placeholder="Es. Operatore CNC")
    jd_input = st.text_area("Job description", height=250, placeholder="Incolla qui la JD completa…")
    submitted = st.form_submit_button("Genera suggerimenti")

if submitted:
    if not job_title_input or not jd_input:
        st.error("⚠️ Inserisci sia il job title che la job description!")
        st.stop()

    with st.spinner("Analisi in corso…"):
        jd_clean = clean_text(jd_input)
        kw = extract_keywords(jd_clean)
        cand = generate_titles(job_title_input, kw)

        g_scores = google_trends_scores(cand)
        jb_scores = job_board_counts(cand)

        rows = []
        for ct in cand:
            google = g_scores.get(ct, 0)
            indeed = jb_scores.get(ct, {}).get("indeed", 0)
            linkedin = jb_scores.get(ct, {}).get("linkedin", 0)
            sim = jd_similarity(ct, jd_clean)
            score = (google/100)*0.5 + min(indeed,10)/10*0.25 + min(linkedin,10)/10*0.15 + sim*0.10
            rows.append({
                "Titolo suggerito": ct,
                "Google Trends (0‑100)": google,
                "Annunci Indeed": indeed,
                "Annunci LinkedIn": linkedin,
                "Pertinenza JD (0‑1)": round(sim,2),
                "Score finale (0‑1)": round(score,2)
            })

        df = pd.DataFrame(rows).sort_values("Score finale (0‑1)", ascending=False).reset_index(drop=True)

    st.success("Ecco i titoli ottimizzati!")
    st.dataframe(df, use_container_width=True)

    st.markdown("""**Legenda Score**  
    • 50 % Google Trends (normalizzato 0‑1)  
    • 25 % Indeed (max 10 annunci)  
    • 15 % LinkedIn (max 10 annunci)  
    • 10 % Pertinenza JD  
    Puoi modificare i pesi nella funzione `score`.""")
