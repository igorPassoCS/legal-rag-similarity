"""
Fase 5 - Camada de decisao.

Recebe a matriz de correspondencias de um par (query, candidato) - saida da
Fase 4 - e atribui um rotulo: copia, versao, parente_de_tese ou diferente.

Principio: o rotulo NAO sai de um numero unico, e sim do PERFIL por secao.
As secoes tem pesos diferentes pela sua FUNCAO JURIDICA, nao por ajuste:
  - fundamentacao -> carrega a tese (o que se reaproveita). EIXO PRINCIPAL.
  - fatos         -> separam "mesmo caso" de "tese parecida em caso diferente".
  - pedidos/abertura -> forma processual padronizada. NAO sao discriminantes
                        (o pedidos do impostor casou em 0.898, mais alto que o
                        do parente em 0.824). Por isso nem entram nas regras.

Contrato de saida:
  { "rotulo": str, "scores_por_secao": {...}, "motivo": str }
"""

import difflib

from matching import match_query_against_acervo

# ---------------------------------------------------------------------------
# CORTES (calibrados contra data/ground_truth.json).
#
# HONESTIDADE SOBRE A CALIBRACAO: estes numeros foram ajustados a mao para que
# os 12 pares de um dataset PEQUENO e SINTETICO (16 docs) saiam corretos. NAO e
# aprendizado: em producao, viriam de um conjunto real e rotulado, via validacao
# cruzada. Sao constantes nomeadas e isoladas de proposito - para serem
# substituiveis e auditaveis, nao escondidas no meio da logica.
#
# Evidencia da calibracao (scores observados na Fase 4, todos os 4 casos-base):
#   fundamentacao -> parentes  : [0.669 .. 0.791]
#                    impostores : [0.547 .. 0.638]   => corte na janela (0.638, 0.669)
#   fatos         -> copias    : [0.949 .. 1.000]
#                    parentes  : [0.643 .. 0.724]    => corte na janela (0.724, 0.949)
#   difflib       -> copias    : [0.930 .. 0.962]
# ---------------------------------------------------------------------------

# Eixo principal: ha tese juridica em comum? (separa parente_de_tese de diferente)
# Posto deliberadamente no PISO da janela valida (logo acima do teto dos
# impostores, 0.638), e nao no meio dela. Isso materializa o VIES PRO-COBERTURA:
# na duvida entre parente e diferente, a fronteira pende para INCLUIR como
# parente. Um corte pro-precisao estaria mais alto (~0.72).
FUND_ALTA = 0.65

# Segundo eixo: os fatos sao os mesmos? (separa "mesmo caso" de "mesma tese").
# Fica no meio da janela larga (0.724..0.949), longe das duas pontas - decisao
# robusta, sem disputa.
FATOS_ALTA = 0.85

# Desempate copia x versao por sobreposicao LITERAL (difflib sobre texto bruto).
# Calibrado em 0.90, e NAO no 0.95 ilustrativo do enunciado: as copias do dataset
# nao sao byte-identicas (mudam nome, CPF, e a concordancia de genero:
# autora->autor etc.), ficando em difflib 0.93-0.96. Com 0.95, duas das quatro
# copias virariam "versao" por engano. RESSALVA: o dataset NAO tem exemplo de
# "versao", entao a fronteira inferior de copia/versao nao pode ser calibrada
# por dados - 0.90 apenas garante que as copias conhecidas caiam como copia.
DIFFLIB_COPIA = 0.90


def _score(correspondencias, tipo):
    """Melhor score de casamento da secao `tipo` da query (None se ausente)."""
    for m in correspondencias:
        if m["tipo_query"] == tipo:
            return m["score"]
    return None


def classificar(correspondencias, texto_query=None, texto_candidato=None):
    """Classifica um par a partir da sua matriz de correspondencias.

    texto_query / texto_candidato sao necessarios APENAS no desempate
    copia x versao (ramo "mesmo caso"), onde se usa difflib sobre o texto bruto.
    """
    fund = _score(correspondencias, "fundamentacao") or 0.0
    fatos = _score(correspondencias, "fatos") or 0.0

    # Evidencia: todos os scores de secao que entraram (inclusive pedidos, para
    # transparencia - mostra que ele foi VISTO mas nao pesou na regra).
    scores_por_secao = {
        m["tipo_query"]: round(m["score"], 3) for m in correspondencias
    }

    # --- Eixo principal: sem tese em comum, nada a reaproveitar. ---
    if fund < FUND_ALTA:
        return {
            "rotulo": "diferente",
            "scores_por_secao": scores_por_secao,
            "motivo": (
                f"fundamentacao baixa (cos={fund:.3f} < {FUND_ALTA}) "
                f"-> sem tese juridica em comum"
            ),
        }

    # A partir daqui ha tese em comum (fundamentacao alta).
    # --- Segundo eixo: os fatos tambem casam? Entao e o MESMO caso. ---
    if fatos >= FATOS_ALTA:
        if texto_query is None or texto_candidato is None:
            raise ValueError(
                "texto bruto dos dois documentos e necessario para o "
                "desempate copia x versao"
            )
        ratio = difflib.SequenceMatcher(None, texto_query, texto_candidato).ratio()
        scores_por_secao["difflib"] = round(ratio, 3)
        if ratio >= DIFFLIB_COPIA:
            rotulo = "copia"
            motivo = (
                f"fundamentacao e fatos altos (cos {fund:.3f}/{fatos:.3f}) e texto "
                f"quase identico (difflib={ratio:.3f} >= {DIFFLIB_COPIA}) -> copia"
            )
        else:
            rotulo = "versao"
            motivo = (
                f"fundamentacao e fatos altos (mesmo caso) mas texto editado "
                f"(difflib={ratio:.3f} < {DIFFLIB_COPIA}) -> versao"
            )
        return {
            "rotulo": rotulo,
            "scores_por_secao": scores_por_secao,
            "motivo": motivo,
        }

    # --- Tese em comum, mas fatos diferentes: o rotulo mais valioso. ---
    return {
        "rotulo": "parente_de_tese",
        "scores_por_secao": scores_por_secao,
        "motivo": (
            f"fundamentacao alta (cos={fund:.3f} >= {FUND_ALTA}), fatos baixos "
            f"(cos={fatos:.3f} < {FATOS_ALTA}) -> mesma tese, caso diferente"
        ),
    }


# ---------------------------------------------------------------------------
# Fase 6 - saida estruturada (o "produto" consumido pela geracao de defesa).
# ---------------------------------------------------------------------------

# Rotulos que geram acao (entram na saida). "diferente" nao gera acao e por
# isso NAO aparece - ver justificativa no final do arquivo.
ROTULOS_RELACIONADOS = ("copia", "versao", "parente_de_tese")

# Prioridade de ordenacao: primeiro "mesmo caso" (copia/versao), depois parente.
_PRIORIDADE = {"copia": 0, "versao": 0, "parente_de_tese": 1}


def _recomendacao(rotulo, scores):
    """Traduz o rotulo em uma acao curta, coerente com a arquitetura."""
    if rotulo == "copia":
        return "Documento duplicado. Vincular ao original; confirmacao humana."
    if rotulo == "versao":
        return "Nova versao do caso. Encadear com historico; destacar o que mudou."
    if rotulo == "parente_de_tese":
        fatos = scores.get("fatos") or 0.0
        return (
            "Mesma tese, fatos diferentes. Reaproveitar a fundamentacao como base "
            "para a defesa; NAO copiar os fatos. "
            f"Divergencia nos fatos: cos={fatos:.3f}."
        )
    return ""


def analisar_query(query_id, chunks_por_doc, textos_por_doc):
    """Produz a saida estruturada da query contra o acervo (o produto do sistema).

    Para cada candidato relacionado (rotulo != "diferente"), um registro com o
    rotulo, os scores por secao que o justificam, o motivo rastreavel e a
    recomendacao acionavel. Candidatos "diferente" nao entram (nao ha acao).

    Ordenado por relevancia: mesmo-caso (copia/versao) antes de parente; dentro
    de cada grupo, por fundamentacao decrescente.
    """
    registros = []
    for r in match_query_against_acervo(query_id, chunks_por_doc):
        cand = r["candidato"]
        dec = classificar(
            r["correspondencias"], textos_por_doc[query_id], textos_por_doc[cand]
        )
        if dec["rotulo"] == "diferente":
            continue
        s = dec["scores_por_secao"]
        registros.append(
            {
                "documento": query_id,
                "candidato": cand,
                "rotulo": dec["rotulo"],
                "score_por_secao": {
                    "fatos": s.get("fatos"),
                    "fundamentacao": s.get("fundamentacao"),
                    "pedidos": s.get("pedidos"),
                },
                "motivo": dec["motivo"],
                "recomendacao": _recomendacao(dec["rotulo"], s),
            }
        )

    registros.sort(
        key=lambda x: (
            _PRIORIDADE[x["rotulo"]],
            -(x["score_por_secao"]["fundamentacao"] or 0.0),
        )
    )
    return registros


if __name__ == "__main__":
    import json
    import os

    from chunking import chunk_file
    from embeddings import embed_chunks
    from matching import match_pair

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(docs_dir) if f.endswith(".txt")
    )
    todos = []
    for nome in nomes:
        todos.extend(chunk_file(os.path.join(docs_dir, nome + ".txt")))
    todos = embed_chunks(todos, verbose=False)
    por_doc = {}
    for c in todos:
        por_doc.setdefault(c["doc_id"], []).append(c)
    raw = {
        n: open(os.path.join(docs_dir, n + ".txt"), encoding="utf-8").read()
        for n in nomes
    }

    print("=" * 70)
    print("DECISAO  -  query = maria")
    print("=" * 70)
    for cand in ["joao", "pedro", "ana"]:
        corr = match_pair(por_doc["maria"], por_doc[cand])
        r = classificar(corr, raw["maria"], raw[cand])
        print(f"\nmaria x {cand}  ->  {r['rotulo'].upper()}")
        print(f"    scores: {r['scores_por_secao']}")
        print(f"    motivo: {r['motivo']}")

    # Conferencia da calibracao: todos os 12 pares do gabarito.
    gt = json.load(open(os.path.join(docs_dir, "..", "ground_truth.json"), encoding="utf-8"))
    print("\n" + "=" * 70)
    print(f"CONFERENCIA DA CALIBRACAO ({len(gt['pares_esperados'])} pares do ground_truth)")
    print("=" * 70)
    acertos = 0
    for p in gt["pares_esperados"]:
        q, a = p["query"], p["alvo"]
        corr = match_pair(por_doc[q], por_doc[a])
        r = classificar(corr, raw[q], raw[a])
        ok = r["rotulo"] == p["rotulo_esperado"]
        acertos += ok
        marca = "OK " if ok else "ERRO"
        print(f"  [{marca}] {q:<9} x {a:<9} esperado={p['rotulo_esperado']:<16} obtido={r['rotulo']}")
    print(f"\n  {acertos}/{len(gt['pares_esperados'])} corretos")
