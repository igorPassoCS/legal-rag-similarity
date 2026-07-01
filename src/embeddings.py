"""
Fase 3 - Embeddings com cache local.

Recebe os chunks da Fase 2 ({doc_id, tipo, texto}) e anexa a cada um o seu
vetor de embedding (text-embedding-3-small da OpenAI), retornando:
    { "doc_id": str, "tipo": str, "texto": str, "vetor": np.ndarray }

Cache:
    Os textos sao estaticos e cada embedding custa uma chamada de API, entao
    guardamos os vetores em disco indexados por sha256 do TEXTO do chunk.
    Antes de chamar a API, so pedimos o que ainda nao esta em cache.

Normalizacao:
    Aqui os vetores sao guardados/retornados CRUS (como vem da API). A
    normalizacao L2 acontece na Fase 4, no momento do cosseno. Ver explicacao
    no final do arquivo.
"""

import hashlib
import json
import os

import numpy as np

MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # dimensao do text-embedding-3-small

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "data", "embeddings_cache.json"
)


def _hash(texto):
    """Chave de cache: sha256 do texto exato do chunk (bytes UTF-8).

    E o TEXTO que identifica o embedding, nao o doc_id. Assim, dois chunks de
    documentos diferentes mas com texto identico (ex.: a secao DOS PEDIDOS de
    Maria e de Joao) compartilham a mesma entrada e gastam uma unica chamada.
    """
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _fetch_embeddings(textos, model):
    """Chama a API da OpenAI em lote. So e invocada quando ha texto faltando."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY nao definida no ambiente, e ha textos sem embedding "
            "em cache. Defina a chave para gerar os embeddings que faltam."
        )
    from openai import OpenAI

    client = OpenAI()  # le OPENAI_API_KEY do ambiente; nada e hardcoded
    vetores = []
    LOTE = 100  # a API aceita varios textos por chamada
    for i in range(0, len(textos), LOTE):
        resp = client.embeddings.create(model=model, input=textos[i : i + LOTE])
        vetores.extend(d.embedding for d in resp.data)
    return vetores


def embed_chunks(chunks, model=MODEL, cache_path=None, verbose=True):
    """Anexa o campo 'vetor' (np.ndarray) a cada chunk, usando cache em disco.

    Retorna uma nova lista; nao muta a de entrada.
    """
    cache_path = cache_path or _DEFAULT_CACHE
    cache = _load_cache(cache_path)

    # Quais textos UNICOS ainda nao tem embedding? (dedup por hash dentro do run)
    faltantes = {}
    for c in chunks:
        h = _hash(c["texto"])
        if h not in cache:
            faltantes[h] = c["texto"]

    if faltantes:
        hashes = list(faltantes.keys())
        textos = [faltantes[h] for h in hashes]
        novos = _fetch_embeddings(textos, model)
        for h, v in zip(hashes, novos):
            cache[h] = v
        _save_cache(cache, cache_path)

    if verbose:
        chamadas = len(faltantes)
        reaproveitados = len(chunks) - sum(
            1 for c in chunks if _hash(c["texto"]) in faltantes
        )
        print(
            f"embeddings: {len(chunks)} chunks | "
            f"{chamadas} textos novos via API | "
            f"{reaproveitados} reaproveitados do cache"
        )

    return [
        {**c, "vetor": np.array(cache[_hash(c["texto"])], dtype=np.float32)}
        for c in chunks
    ]


if __name__ == "__main__":
    from chunking import chunk_file

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(docs_dir) if f.endswith(".txt")
    )

    # 1) Chunking + embeddings de todos os 16 documentos.
    todos_chunks = []
    for nome in nomes:
        todos_chunks.extend(chunk_file(os.path.join(docs_dir, nome + ".txt")))

    chunks = embed_chunks(todos_chunks)

    dims = {c["vetor"].shape[0] for c in chunks}
    print(f"docs: {len(nomes)} | chunks: {len(chunks)} | dimensoes distintas: {dims}")

    # 2) Sanity check de cosseno (normalizando aqui, como sera feito na Fase 4).
    def cosseno(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def fundamentacao(doc_id):
        for c in chunks:
            if c["doc_id"] == doc_id and c["tipo"] == "fundamentacao":
                return c["vetor"]
        raise KeyError(doc_id)

    sim_parente = cosseno(fundamentacao("maria"), fundamentacao("pedro"))
    sim_impostor = cosseno(fundamentacao("maria"), fundamentacao("ana"))
    print("\nSanity check (cosseno entre fundamentacoes):")
    print(f"  maria x pedro (parente de tese): {sim_parente:.3f}  <- deve ser ALTO")
    print(f"  maria x ana   (impostor):        {sim_impostor:.3f}  <- deve ser BAIXO")
    print(f"  diferenca: {sim_parente - sim_impostor:+.3f}")
