"""
Fase 4 - Motor de matching trecho a trecho.

Dado um documento-query e um documento-candidato, produz a MATRIZ DE
CORRESPONDENCIAS: para cada chunk de CONTEUDO da query, qual chunk do candidato
melhor casa (cosseno) e com que score. Essa matriz e a EVIDENCIA que a Fase 5
usa para decidir o rotulo - ela mostra o FORMATO da similaridade (quais secoes
casam com quais), nao um numero unico.

Contrato de saida por par (query, candidato):
    [ { "tipo_query": str, "tipo_candidato": str, "score": float }, ... ]

Filtro anti-jargao:
    A "abertura" (endereçamento padrao) e EXCLUIDA dos dois lados. Ela infla a
    similaridade por motivo errado: duas peticoes sem relacao real ja se parecem
    na abertura. Ver justificativa no final do arquivo.

Normalizacao:
    A Fase 3 guardou os vetores CRUS de proposito. A normalizacao L2 acontece
    AQUI, em um unico lugar: dentro de cosine_sim().
"""

import numpy as np

# Tipos que NAO entram na matriz (de nenhum dos lados). Tudo que nao estiver
# aqui e tratado como "conteudo". E o filtro anti-jargao.
EXCLUIDOS = {"abertura"}

# Corte provisorio para o sinal agregado (quando uma secao "casou"). Serve so
# para CONTRASTAR perfil x numero unico nesta fase; a calibracao formal de
# limiares e responsabilidade da Fase 5.
CORTE_SECAO = 0.75


def _l2_normalize(v):
    norma = np.linalg.norm(v)
    return v / norma if norma else v


def cosine_sim(a, b):
    """Similaridade de cosseno = produto interno dos vetores normalizados L2.

    UNICO ponto onde a normalizacao acontece (os vetores chegam crus da Fase 3).
    """
    return float(_l2_normalize(a) @ _l2_normalize(b))


def _chunks_de_conteudo(chunks):
    return [c for c in chunks if c["tipo"] not in EXCLUIDOS]


def match_pair(query_chunks, candidate_chunks):
    """Matriz de correspondencias query -> candidato.

    Para cada chunk de conteudo da query, encontra o chunk de conteudo do
    candidato com maior cosseno e registra a correspondencia.
    """
    q_conteudo = _chunks_de_conteudo(query_chunks)
    c_conteudo = _chunks_de_conteudo(candidate_chunks)

    correspondencias = []
    for q in q_conteudo:
        melhor_tipo, melhor_score = None, -1.0
        for cand in c_conteudo:
            s = cosine_sim(q["vetor"], cand["vetor"])
            if s > melhor_score:
                melhor_tipo, melhor_score = cand["tipo"], s
        if melhor_tipo is not None:  # candidato pode nao ter conteudo
            correspondencias.append(
                {
                    "tipo_query": q["tipo"],
                    "tipo_candidato": melhor_tipo,
                    "score": melhor_score,
                }
            )
    return correspondencias


def perfil_agregado(correspondencias, corte=CORTE_SECAO):
    """Sinal agregado que resume o PERFIL por secao de um par.

    Nao e um veredito - e um resumo para contrastar com o numero unico:
      - n_secoes: quantas secoes de conteudo da query foram comparadas
      - secoes_acima_corte: quantas casaram com score >= corte
      - score_medio: media dos melhores scores das secoes de conteudo
    """
    scores = [m["score"] for m in correspondencias]
    return {
        "n_secoes": len(scores),
        "secoes_acima_corte": sum(1 for s in scores if s >= corte),
        "score_medio": float(np.mean(scores)) if scores else 0.0,
        "corte": corte,
    }


def match_query_against_acervo(query_id, chunks_por_doc, corte=CORTE_SECAO):
    """Forca bruta: compara a query contra TODOS os outros documentos do acervo.

    chunks_por_doc: dict doc_id -> lista de chunks-com-vetor (formato da Fase 3).
    Retorna uma lista (uma entrada por candidato), cada uma com a matriz de
    correspondencias e o perfil agregado. E o que a Fase 5 consome.
    """
    query_chunks = chunks_por_doc[query_id]
    resultados = []
    for cand_id, cand_chunks in chunks_por_doc.items():
        if cand_id == query_id:
            continue
        corr = match_pair(query_chunks, cand_chunks)
        resultados.append(
            {
                "query": query_id,
                "candidato": cand_id,
                "correspondencias": corr,
                "agregado": perfil_agregado(corr, corte),
            }
        )
    return resultados


if __name__ == "__main__":
    import os

    from chunking import chunk_file
    from embeddings import embed_chunks

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(docs_dir) if f.endswith(".txt")
    )

    todos = []
    for nome in nomes:
        todos.extend(chunk_file(os.path.join(docs_dir, nome + ".txt")))
    todos = embed_chunks(todos, verbose=False)  # vem do cache; sem API

    chunks_por_doc = {}
    for c in todos:
        chunks_por_doc.setdefault(c["doc_id"], []).append(c)

    # Cosseno da fundamentacao ISOLADA = o "numero unico" que alguem poderia
    # tentar usar como limiar. Vamos contrasta-lo com o perfil por secao.
    def fundamentacao_cos(a, b):
        va = next(c["vetor"] for c in chunks_por_doc[a] if c["tipo"] == "fundamentacao")
        vb = next(c["vetor"] for c in chunks_por_doc[b] if c["tipo"] == "fundamentacao")
        return cosine_sim(va, vb)

    casos = [("joao", "copia"), ("pedro", "parente_de_tese"), ("ana", "diferente")]

    print("=" * 64)
    print("MATRIZ DE CORRESPONDENCIAS  -  query = maria")
    print("(abertura excluida dos dois lados pelo filtro anti-jargao)")
    print("=" * 64)
    for cand, rotulo in casos:
        corr = match_pair(chunks_por_doc["maria"], chunks_por_doc[cand])
        agg = perfil_agregado(corr)
        print(f"\nmaria  x  {cand}   (esperado: {rotulo})")
        for m in corr:
            print(
                f"    {m['tipo_query']:>13} -> {m['tipo_candidato']:<13} "
                f"cos = {m['score']:.3f}"
            )
        print(
            f"    agregado: media={agg['score_medio']:.3f} | "
            f"secoes>= {agg['corte']}: {agg['secoes_acima_corte']}/{agg['n_secoes']}"
        )

    print("\n" + "=" * 64)
    print("CONTRASTE: numero unico  x  perfil por secao")
    print("=" * 64)
    print(f"{'candidato':<10}{'rotulo':<16}{'fund. isolada':<15}{'fatos':<9}{'media':<8}{'secoes>=corte'}")
    for cand, rotulo in casos:
        corr = match_pair(chunks_por_doc["maria"], chunks_por_doc[cand])
        agg = perfil_agregado(corr)
        fund_iso = fundamentacao_cos("maria", cand)
        fatos = next((m["score"] for m in corr if m["tipo_query"] == "fatos"), float("nan"))
        print(
            f"{cand:<10}{rotulo:<16}{fund_iso:<15.3f}{fatos:<9.3f}"
            f"{agg['score_medio']:<8.3f}{agg['secoes_acima_corte']}/{agg['n_secoes']}"
        )
