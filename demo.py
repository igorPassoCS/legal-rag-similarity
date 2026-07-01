"""
demo.py - Fase 6.

Roda a query da Maria contra o acervo e imprime, como narrativa, o que o
sistema encontrou: o rotulo de cada candidato, a evidencia (scores por secao),
o motivo rastreavel e a recomendacao acionavel.

ESCOPO: a POC calcula similaridade, rotula e produz a evidencia/recomendacao.
A GERACAO da defesa em si e downstream e esta FORA do escopo desta POC.
"""

import json
import os
import sys

# stdout em UTF-8 para os acentos saírem corretos no terminal do Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chunking import chunk_file
from decision import analisar_query, classificar
from embeddings import embed_chunks
from matching import match_pair

QUERY = "maria"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")


def carregar_acervo():
    """Chunking + embeddings (do cache) de todos os documentos, agrupados por doc."""
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(DOCS_DIR) if f.endswith(".txt")
    )
    todos = []
    for nome in nomes:
        todos.extend(chunk_file(os.path.join(DOCS_DIR, nome + ".txt")))
    todos = embed_chunks(todos, verbose=False)

    chunks_por_doc, textos = {}, {}
    for c in todos:
        chunks_por_doc.setdefault(c["doc_id"], []).append(c)
    for nome in nomes:
        with open(os.path.join(DOCS_DIR, nome + ".txt"), encoding="utf-8") as f:
            textos[nome] = f.read()
    return chunks_por_doc, textos


def main():
    chunks_por_doc, textos = carregar_acervo()
    candidatos = [d for d in chunks_por_doc if d != QUERY]

    print("=" * 70)
    print(f'  ENTROU A PETIÇÃO: "{QUERY}"  (ação por voo cancelado, ré LATAM)')
    print(f"  O sistema compara contra o acervo ({len(candidatos)} documentos),")
    print("  seção a seção, e decide pelo FORMATO da evidência.")
    print("=" * 70)

    # --- Decisao contra todo o acervo (para a narrativa e a nota da Ana). ---
    decisoes = {
        cand: classificar(
            match_pair(chunks_por_doc[QUERY], chunks_por_doc[cand]),
            textos[QUERY],
            textos[cand],
        )
        for cand in candidatos
    }
    relacionados = [c for c in candidatos if decisoes[c]["rotulo"] != "diferente"]
    descartados = [c for c in candidatos if decisoes[c]["rotulo"] == "diferente"]

    # --- 1) O que foi encontrado. ---
    print(f"\n1) ENCONTRADOS {len(relacionados)} candidatos relacionados "
          f"(e {len(descartados)} descartados como 'diferente'):")
    registros = analisar_query(QUERY, chunks_por_doc, textos)  # ja ordenado
    for reg in registros:
        print(f"     - {reg['candidato']:<16} -> {reg['rotulo']}")

    # --- 2) A saida estruturada: o produto entregue para a geracao de defesa. ---
    print("\n2) SAÍDA ESTRUTURADA (o produto consumido pela geração de defesa):")
    print(json.dumps(registros, ensure_ascii=False, indent=2))

    # --- 3) Narrativa dirigida aos quatro casos-chave. ---
    print("\n3) LEITURA CASO A CASO:")

    def bloco(cand, titulo):
        dec = decisoes[cand]
        s = dec["scores_por_secao"]
        print(f"\n  [{cand}] {titulo}")
        print(f"     rótulo : {dec['rotulo']}")
        print(f"     scores : fatos={s.get('fatos')}  "
              f"fundamentacao={s.get('fundamentacao')}  pedidos={s.get('pedidos')}"
              + (f"  difflib={s.get('difflib')}" if "difflib" in s else ""))
        print(f"     motivo : {dec['motivo']}")

    bloco("joao", "cópia — mesmo caso, texto praticamente idêntico")
    print("     => tudo casa alto (fatos, fundamentação, pedidos) e o texto bruto")
    print("        é quase idêntico: é duplicata, não material novo.")

    bloco("maria_versao", "versão — mesmo caso, texto editado")
    print("     => fatos e fundamentação altíssimos (é o MESMO caso), mas o texto")
    print("        foi reescrito (difflib abaixo do corte de cópia): é uma versão.")

    bloco("pedro", "parente de tese — a assinatura-chave da POC")
    print("     => fundamentação ALTA + fatos BAIXA. Mesma tese jurídica, caso")
    print("        diferente. É o rótulo mais valioso: a fundamentação se reaproveita.")

    bloco("ana", "impostor — vista e DESCARTADA")
    print("     => 'pedidos' casou ALTO (jargão de petição de consumo), mas a")
    print("        fundamentação ficou BAIXA e barrou a decisão. Por ser 'diferente',")
    print("        NÃO entrou na saída estruturada — nenhuma ação, corretamente.")
    print(f"     (confirmação: 'ana' está na saída? "
          f"{'sim' if any(r['candidato'] == 'ana' for r in registros) else 'não'})")


if __name__ == "__main__":
    main()
