
import math
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer



st.set_page_config(
    page_title="HealthSearch - Busca Hibrida",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;}
        .subtitle {color: #52606d; margin-bottom: 1rem;}
        .result-card {
            border: 1px solid #d9e2ec;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            background: #f8fafc;
        }
        .small-note {font-size: 0.9rem; color: #627d98;}
    </style>
    """,
    unsafe_allow_html=True,
)


DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "Doc 1",
        "title": "Protocolo Emergência ECG",
        "content": (
            "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
            "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "title": "Guia de Farmacologia Cardíaca",
        "content": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
            "reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "title": "Diretriz de Hipertensão Arterial",
        "content": (
            "A crise hipertensiva severa requer administração de anti-hipertensivos "
            "venosos e monitoramento contínuo da pressão arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "title": "Manual de AVC Isquêmico",
        "content": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
            "trombolíticos venosos em até quatro horas e meia do início dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "title": "Protocolo de Reanimação RCR",
        "content": (
            "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas "
            "de alta qualidade e desfibrilação precoce no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "title": "Procedimentos de UTI Geral",
        "content": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
            "recomenda-se a monitorização cardíaca contínua por telemetria."
        ),
    },
]


PORTUGUESE_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "as", "ate",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "do",
    "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "essa",
    "essas", "esse", "esses", "esta", "estas", "este", "estes", "foi", "ha",
    "isso", "isto", "ja", "mais", "mas", "me", "mesmo", "na", "nas", "nao",
    "nem", "no", "nos", "nossa", "nosso", "num", "numa", "o", "os", "ou",
    "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que",
    "quem", "se", "sem", "ser", "seu", "seus", "sua", "suas", "tambem", "tem",
    "uma", "umas", "um", "uns",
}


def clean_text(text: str) -> str:
 
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9à-ÿ\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:

    return [
        token
        for token in clean_text(text).split()
        if token not in PORTUGUESE_STOPWORDS and len(token) > 1
    ]


def searchable_text(doc: Dict[str, str]) -> str:
    
    return f"{doc['title']}. {doc['content']}"



EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


@st.cache_data(show_spinner=False)
def build_document_embeddings(_model: SentenceTransformer) -> np.ndarray:
    texts = [searchable_text(doc) for doc in DOCUMENTS]
    # normalize_embeddings=True permite calcular similaridade de cosseno por produto escalar.
    return _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def run_bm25(query: str, k1: float, b: float) -> pd.DataFrame:
    corpus_tokens = [tokenize(searchable_text(doc)) for doc in DOCUMENTS]
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(DOCUMENTS))

    rows = []
    for idx, doc in enumerate(DOCUMENTS):
        rows.append(
            {
                "doc_index": idx,
                "ID": doc["id"],
                "Titulo": doc["title"],
                "Trecho": doc["content"],
                "Score_BM25": float(scores[idx]),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["Score_BM25", "ID"], ascending=[False, True]
    ).reset_index(drop=True)
    df["Rank_BM25"] = np.arange(1, len(df) + 1)
    return df



def run_semantic(query: str, model: SentenceTransformer, doc_embeddings: np.ndarray) -> pd.DataFrame:
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    cosine_scores = doc_embeddings @ query_embedding

    rows = []
    for idx, doc in enumerate(DOCUMENTS):
        rows.append(
            {
                "doc_index": idx,
                "ID": doc["id"],
                "Titulo": doc["title"],
                "Trecho": doc["content"],
                "Similaridade_Cosseno": float(cosine_scores[idx]),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["Similaridade_Cosseno", "ID"], ascending=[False, True]
    ).reset_index(drop=True)
    df["Rank_Semantico"] = np.arange(1, len(df) + 1)
    return df



def run_rrf(df_bm25: pd.DataFrame, df_sem: pd.DataFrame, alpha: float) -> pd.DataFrame:
    k_rrf = 60

    bm25_map = df_bm25.set_index("doc_index")[["Score_BM25", "Rank_BM25"]].to_dict("index")
    sem_map = df_sem.set_index("doc_index")[["Similaridade_Cosseno", "Rank_Semantico"]].to_dict("index")

    rows = []
    for idx, doc in enumerate(DOCUMENTS):
        rank_bm25 = int(bm25_map[idx]["Rank_BM25"])
        rank_sem = int(sem_map[idx]["Rank_Semantico"])
        rrf_score = (
            alpha * (1.0 / (k_rrf + rank_bm25))
            + (1.0 - alpha) * (1.0 / (k_rrf + rank_sem))
        )

        rows.append(
            {
                "doc_index": idx,
                "ID": doc["id"],
                "Titulo": doc["title"],
                "Trecho": doc["content"],
                "Rank_BM25": rank_bm25,
                "Rank_Semantico": rank_sem,
                "Score_BM25": float(bm25_map[idx]["Score_BM25"]),
                "Similaridade_Cosseno": float(sem_map[idx]["Similaridade_Cosseno"]),
                "Score_RRF": float(rrf_score),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["Score_RRF", "ID"], ascending=[False, True]
    ).reset_index(drop=True)
    df["Rank_RRF"] = np.arange(1, len(df) + 1)
    return df



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def run_cross_encoder(query: str, df_rrf: pd.DataFrame) -> pd.DataFrame:
    top3 = df_rrf.head(3).copy()
    model = load_cross_encoder()
    pairs: List[Tuple[str, str]] = [
        (query, f"{row['Titulo']}. {row['Trecho']}")
        for _, row in top3.iterrows()
    ]
    raw_scores = model.predict(pairs)

    top3["Score_CrossEncoder_Raw"] = [float(s) for s in raw_scores]
    top3["Relevancia_CrossEncoder"] = [sigmoid(s) for s in raw_scores]
    top3["Rank_RRF_Antes"] = top3["Rank_RRF"]
    top3 = top3.sort_values(
        by="Score_CrossEncoder_Raw", ascending=False
    ).reset_index(drop=True)
    top3["Rank_CrossEncoder"] = np.arange(1, len(top3) + 1)
    top3["Variacao_Rank"] = top3["Rank_RRF_Antes"] - top3["Rank_CrossEncoder"]
    return top3



st.sidebar.title("⚙️ Calibracao do HealthSearch")
st.sidebar.subheader("BM25")
k1 = st.sidebar.slider(
    "k1 - saturacao da frequencia",
    min_value=0.0,
    max_value=3.0,
    value=1.2,
    step=0.1,
)
b = st.sidebar.slider(
    "b - normalizacao por comprimento",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
)

st.sidebar.subheader("Fusao RRF")
alpha = st.sidebar.slider(
    "α - peso do BM25",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="α=1 usa apenas a contribuicao do rank BM25; α=0 usa apenas a contribuicao do rank semantico.",
)
st.sidebar.caption("Constante RRF fixa: k = 60")

st.sidebar.subheader("Bonus")
use_cross_encoder = st.sidebar.checkbox(
    "Ativar Cross-Encoder nos Top-3 do RRF", value=False
)


st.markdown('<div class="main-title">🩺 HealthSearch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Motor de busca hibrido para protocolos e diretrizes medicas: BM25 + Embeddings + RRF.</div>',
    unsafe_allow_html=True,
)

example_query = st.selectbox(
    "Consulta de exemplo",
    [
        "ataque cardíaco",
        "CÓD-ECG-12D",
        "pressão muito alta na UTI",
        "derrame isquêmico tratamento",
        "parada cardíaca desfibrilação",
    ],
    index=0,
)
query = st.text_input("Digite a consulta:", value=example_query)

if not query.strip():
    st.info("Digite uma consulta para executar os tres motores de busca.")
    st.stop()



try:
    with st.spinner("Carregando modelo semantico e processando a consulta..."):
        start = time.perf_counter()
        embedding_model = load_embedding_model()
        document_embeddings = build_document_embeddings(embedding_model)

        df_bm25 = run_bm25(query, k1=k1, b=b)
        df_sem = run_semantic(query, embedding_model, document_embeddings)
        df_rrf = run_rrf(df_bm25, df_sem, alpha=alpha)
        elapsed_ms = (time.perf_counter() - start) * 1000
except Exception as exc:
    st.error(
        "Nao foi possivel inicializar a busca semantica. Verifique a instalacao de "
        "sentence-transformers e a conexao necessaria para baixar o modelo na primeira execucao."
    )
    st.exception(exc)
    st.stop()

best_bm25 = df_bm25.iloc[0]
best_sem = df_sem.iloc[0]
best_rrf = df_rrf.iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Top BM25", f"{best_bm25['ID']}")
m2.metric("Top Semantico", f"{best_sem['ID']}")
m3.metric("Top Hibrido RRF", f"{best_rrf['ID']}")
m4.metric("Tempo da consulta", f"{elapsed_ms:.1f} ms")



tab_lex, tab_sem, tab_rrf, tab_matrix, tab_bonus = st.tabs(
    [
        "🔤 Lexico - BM25",
        "🧠 Semantico",
        "🔀 Hibrido - RRF",
        "📊 Matriz Comparativa",
        "🎯 Bonus - Cross-Encoder",
    ]
)

with tab_lex:
    st.subheader(f"Ranking BM25 - k1={k1:.2f}, b={b:.2f}")
    st.caption(
        "O BM25 privilegia correspondencia lexical, sendo especialmente util para codigos, siglas e termos exatos."
    )
    view = df_bm25[["Rank_BM25", "ID", "Titulo", "Score_BM25", "Trecho"]].copy()
    view["Score_BM25"] = view["Score_BM25"].round(4)
    st.dataframe(view, use_container_width=True, hide_index=True)

with tab_sem:
    st.subheader("Ranking por Similaridade de Cosseno")
    st.caption(
        f"Embeddings densos gerados pelo modelo {EMBEDDING_MODEL_NAME}. A busca semantica tenta capturar sinonimos e contexto."
    )
    view = df_sem[["Rank_Semantico", "ID", "Titulo", "Similaridade_Cosseno", "Trecho"]].copy()
    view["Similaridade_Cosseno"] = view["Similaridade_Cosseno"].round(4)
    st.dataframe(view, use_container_width=True, hide_index=True)

with tab_rrf:
    st.subheader(f"Ranking Hibrido por Reciprocal Rank Fusion - α={alpha:.2f}, k=60")
    st.latex(
        r"Score_{RRF}(D)=\alpha\left(\frac{1}{60+Rank_{BM25}}\right)+(1-\alpha)\left(\frac{1}{60+Rank_{Semantico}}\right)"
    )
    view = df_rrf[
        [
            "Rank_RRF",
            "ID",
            "Titulo",
            "Rank_BM25",
            "Rank_Semantico",
            "Score_RRF",
            "Trecho",
        ]
    ].copy()
    view["Score_RRF"] = view["Score_RRF"].round(6)
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.markdown("### Resultado mais relevante")
    for pos, (_, row) in enumerate(df_rrf.head(3).iterrows(), start=1):
        st.markdown(
            f"""
            <div class="result-card">
                <strong>#{pos} - {row['ID']} | {row['Titulo']}</strong><br>
                <span>{row['Trecho']}</span><br>
                <span class="small-note">Rank BM25: {int(row['Rank_BM25'])} | Rank Semantico: {int(row['Rank_Semantico'])} | Score RRF: {row['Score_RRF']:.6f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_matrix:
    st.subheader("Matriz Comparativa de Posicoes")
    matrix = df_rrf[
        ["ID", "Titulo", "Rank_BM25", "Rank_Semantico", "Rank_RRF"]
    ].sort_values("ID")
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.markdown("#### Visualizacao comparativa")
    chart = matrix.set_index("ID")[["Rank_BM25", "Rank_Semantico", "Rank_RRF"]]
    st.line_chart(chart)
    st.caption("Em rankings, posicoes menores representam maior relevancia.")

with tab_bonus:
    st.subheader("Cross-Encoder Re-Ranking - Bonus")
    if not use_cross_encoder:
        st.info("Ative o checkbox na barra lateral para reordenar os Top-3 candidatos do RRF.")
    else:
        try:
            with st.spinner("Aplicando Cross-Encoder nos Top-3 candidatos..."):
                df_cross = run_cross_encoder(query, df_rrf)

            st.caption(
                "O Cross-Encoder avalia diretamente cada par consulta-documento e pode alterar a ordem dos tres melhores resultados recuperados pelo RRF."
            )
            view = df_cross[
                [
                    "Rank_RRF_Antes",
                    "Rank_CrossEncoder",
                    "Variacao_Rank",
                    "ID",
                    "Titulo",
                    "Score_CrossEncoder_Raw",
                    "Relevancia_CrossEncoder",
                ]
            ].copy()
            view["Score_CrossEncoder_Raw"] = view["Score_CrossEncoder_Raw"].round(4)
            view["Relevancia_CrossEncoder"] = view["Relevancia_CrossEncoder"].round(4)
            st.dataframe(view, use_container_width=True, hide_index=True)

            for pos, (_, row) in enumerate(df_cross.iterrows(), start=1):
                st.markdown(
                    f"""
                    <div class="result-card">
                        <strong>#{pos} - {row['ID']} | {row['Titulo']}</strong><br>
                        <span>Rank RRF antes: {int(row['Rank_RRF_Antes'])} → Rank Cross-Encoder: {int(row['Rank_CrossEncoder'])}</span><br>
                        <span class="small-note">Relevancia normalizada: {row['Relevancia_CrossEncoder']:.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(
                "Nao foi possivel carregar o Cross-Encoder. Na primeira execucao, o modelo precisa ser baixado."
            )
            st.exception(exc)

st.markdown("---")
st.caption(
    "HealthSearch - prototipo academico de recuperacao de informacao. O corpus e fixo e foi definido pelo enunciado do laboratorio."
)
