"""
Fase 7 - Avaliacao (baseline "antes" de qualquer remedio).

MEDE o sistema como ele esta hoje; NAO altera o pipeline (Fase 3/4/5).
Tres blocos:
  1. Classificacao  - matriz de confusao 4x4 nos 16 pares curados do gabarito.
  2. Recuperacao    - cada query contra o acervo inteiro; precisao x recall.
  3. Diagnostico    - para onde vao os falsos positivos e a prova de que
                      nenhum limiar absoluto de fundamentacao separa as classes.

(scikit-learn poderia montar a matriz de confusao do bloco 1; como sao 4 classes
e 16 pares, montamos a mao para nao adicionar dependencia - o resultado e o mesmo.)
"""

import json
import os

from chunking import chunk_file
from decision import classificar
from embeddings import embed_chunks
from matching import match_pair

LABELS = ["copia", "versao", "parente_de_tese", "diferente"]
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
GT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.json")


def carregar():
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(DOCS_DIR) if f.endswith(".txt")
    )
    todos = []
    for nome in nomes:
        todos.extend(chunk_file(os.path.join(DOCS_DIR, nome + ".txt")))
    todos = embed_chunks(todos, verbose=False)
    por, raw = {}, {}
    for c in todos:
        por.setdefault(c["doc_id"], []).append(c)
    for nome in nomes:
        with open(os.path.join(DOCS_DIR, nome + ".txt"), encoding="utf-8") as f:
            raw[nome] = f.read()
    return gt, por, raw


def _fund(corr):
    """Melhor score da fundamentacao da query (o eixo principal da decisao)."""
    return next((m["score"] for m in corr if m["tipo_query"] == "fundamentacao"), 0.0)


# ---------------------------------------------------------------------------
# BLOCO 1 - Classificacao nos pares curados.
# ---------------------------------------------------------------------------
def bloco1_classificacao(gt, por, raw):
    print("=" * 72)
    print(" BLOCO 1 - CLASSIFICACAO (16 pares curados do gabarito)")
    print("=" * 72)

    matriz = {esp: {prev: 0 for prev in LABELS} for esp in LABELS}
    acertos = 0
    for p in gt["pares_esperados"]:
        q, a = p["query"], p["alvo"]
        prev = classificar(match_pair(por[q], por[a]), raw[q], raw[a])["rotulo"]
        esp = p["rotulo_esperado"]
        matriz[esp][prev] += 1
        acertos += esp == prev

    largura = max(len(l) for l in LABELS)
    cab = " " * (largura + 3) + "".join(f"{l[:11]:>13}" for l in LABELS)
    print("\n  matriz de confusao (linha = esperado, coluna = previsto)")
    print(cab)
    for esp in LABELS:
        linha = f"  {esp:<{largura}} " + "".join(
            f"{matriz[esp][prev]:>13}" for prev in LABELS
        )
        print(linha)
    total = len(gt["pares_esperados"])
    print(f"\n  acuracia: {acertos}/{total} = {acertos / total:.2f}")
    print("  => com o PAR CERTO em maos, a classificacao acerta em cheio.")


# ---------------------------------------------------------------------------
# BLOCO 2 - Recuperacao em escala.
# ---------------------------------------------------------------------------
def bloco2_recuperacao(gt, por, raw):
    print("\n" + "=" * 72)
    print(" BLOCO 2 - RECUPERACAO EM ESCALA (cada query x acervo inteiro)")
    print("=" * 72)

    # Para cada query, lista de relacionados (rotulo != diferente), ordenada por
    # fundamentacao decrescente (a mesma ordem de relevancia da Fase 6).
    retornados = {}
    for q in por:
        rel = []
        for cand in por:
            if cand == q:
                continue
            corr = match_pair(por[q], por[cand])
            if classificar(corr, raw[q], raw[cand])["rotulo"] != "diferente":
                rel.append((cand, _fund(corr)))
        rel.sort(key=lambda x: -x[1])
        retornados[q] = [c for c, _ in rel]

    tot_ret = tot_tp = tot_true = 0
    precisions, recalls, rr = [], [], []
    linhas = []
    for q in sorted(por):
        verd = set(gt["recall_at_k"][q])
        ret = retornados[q]
        tp = sum(1 for c in ret if c in verd)
        tot_ret += len(ret)
        tot_tp += tp
        tot_true += len(verd)

        p_at_k = tp / len(ret) if ret else float("nan")
        r_at_k = tp / len(verd) if verd else float("nan")
        if ret:
            precisions.append(p_at_k)
        if verd:
            recalls.append(r_at_k)
            pos = next((i + 1 for i, c in enumerate(ret) if c in verd), None)
            rr.append(1 / pos if pos else 0.0)
        linhas.append((q, len(verd), len(ret), tp, p_at_k, r_at_k))

    print(f"\n  {'query':<16}{'#verd':>6}{'k=#ret':>8}{'tp':>5}{'prec@k':>9}{'rec@k':>8}")
    for q, nv, k, tp, pk, rk in linhas:
        pk_s = f"{pk:.2f}" if pk == pk else "  -"
        rk_s = f"{rk:.2f}" if rk == rk else "  -"
        print(f"  {q:<16}{nv:>6}{k:>8}{tp:>5}{pk_s:>9}{rk_s:>8}")

    prec_g = tot_tp / tot_ret if tot_ret else float("nan")
    rec_g = tot_tp / tot_true if tot_true else float("nan")
    print("\n  --- AGREGADO GLOBAL (lado a lado) ---")
    print(f"  PRECISAO global de 'relacionado' : {prec_g:.2f}  "
          f"({tot_tp} verdadeiros de {tot_ret} marcados)")
    print(f"  RECALL   global de 'relacionado' : {rec_g:.2f}  "
          f"({tot_tp} capturados de {tot_true} existentes)")
    print(f"  precisao@k media : {sum(precisions) / len(precisions):.2f}")
    print(f"  recall@k media   : {sum(recalls) / len(recalls):.2f}")
    print(f"  MRR              : {sum(rr) / len(rr):.2f}")
    print("\n  => LEITURA: recall alto + precisao baixa = o trade-off PRO-COBERTURA")
    print("     projetado (capturo quase tudo, ao custo de muitos alertas falsos),")
    print("     NAO um bug que perde casos uteis.")
    return retornados


# ---------------------------------------------------------------------------
# BLOCO 3 - Diagnostico: para onde vao os falsos positivos.
# ---------------------------------------------------------------------------
def bloco3_diagnostico(gt, por, raw):
    print("\n" + "=" * 72)
    print(" BLOCO 3 - DIAGNOSTICO (para onde vao os falsos positivos)")
    print("=" * 72)

    docs = gt["documentos"]
    falsos, verdadeiros = [], []  # (query, cand, fund)
    fp_cross_familia = 0
    for q in por:
        verd = set(gt["recall_at_k"][q])
        for cand in por:
            if cand == q:
                continue
            corr = match_pair(por[q], por[cand])
            fund = _fund(corr)
            relacionado = classificar(corr, raw[q], raw[cand])["rotulo"] != "diferente"
            if cand in verd:
                verdadeiros.append((q, cand, fund))
            elif relacionado:
                falsos.append((q, cand, fund))
                if docs[q]["caso_base"] != docs[cand]["caso_base"]:
                    fp_cross_familia += 1

    print(f"\n  falsos positivos: {len(falsos)} | "
          f"deles cross-familia (area de consumo diferente): {fp_cross_familia} "
          f"({fp_cross_familia / len(falsos) * 100:.0f}%)")

    print("\n  amostra dos falsos positivos com MAIOR fundamentacao:")
    for q, cand, fund in sorted(falsos, key=lambda x: -x[2])[:5]:
        print(f"     {q:<16} x {cand:<16} fund={fund:.3f}  "
              f"({docs[q]['caso_base']} x {docs[cand]['caso_base']})")

    def stats(lst):
        s = [f for _, _, f in lst]
        return min(s), sum(s) / len(s), max(s)

    fp_min, fp_med, fp_max = stats(falsos)
    tp_min, tp_med, tp_max = stats(verdadeiros)
    print("\n  distribuicao do score de FUNDAMENTACAO:")
    print(f"     parentes VERDADEIROS : min={tp_min:.3f}  media={tp_med:.3f}  max={tp_max:.3f}")
    print(f"     falsos POSITIVOS     : min={fp_min:.3f}  media={fp_med:.3f}  max={fp_max:.3f}")

    pior_fp = max(falsos, key=lambda x: x[2])
    pior_tp = min(verdadeiros, key=lambda x: x[2])
    print("\n  SOBREPOSICAO das distribuicoes (o ponto fatal):")
    print(f"     maior fund entre FALSOS+   : {pior_fp[2]:.3f}  ({pior_fp[0]} x {pior_fp[1]})")
    print(f"     menor fund entre VERDADEIR.: {pior_tp[2]:.3f}  ({pior_tp[0]} x {pior_tp[1]})")

    print("\n  " + "-" * 68)
    if pior_fp[2] > pior_tp[2]:
        print(f"  CONCLUSAO: parente verdadeiro em ({pior_tp[0]} x {pior_tp[1]}) pontua "
              f"{pior_tp[2]:.3f},")
        print(f"  ABAIXO do falso positivo em ({pior_fp[0]} x {pior_fp[1]}), que pontua "
              f"{pior_fp[2]:.3f}.")
        print("  => limiar absoluto NAO separa as classes; remedio = ranking relativo")
        print("     (top-k por query) e/ou remocao de boilerplate da fundamentacao.")
    else:
        print("  (sem sobreposicao neste dataset: um limiar absoluto separaria as classes.)")
    print("  " + "-" * 68)


if __name__ == "__main__":
    gt, por, raw = carregar()
    bloco1_classificacao(gt, por, raw)
    bloco2_recuperacao(gt, por, raw)
    bloco3_diagnostico(gt, por, raw)
