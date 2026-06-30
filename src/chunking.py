"""
Fase 2 - Chunking por secao.

Le uma peticao e a divide em trechos (chunks), um por secao canonica,
anexando a cada trecho o seu tipo: abertura, fatos, fundamentacao ou pedidos.

Se nenhum titulo canonico for encontrado, cai num fallback por paragrafos
(tipo "indefinido"), para nao quebrar diante de peticoes mal formatadas.

Contrato de saida (consumido pela Fase 3 - embeddings):
    [ { "doc_id": str, "tipo": str, "texto": str }, ... ]
"""

import os
import re

# Tamanho-alvo de cada chunk no fallback, em palavras. Ver justificativa no
# final do arquivo; e proposital que case com a ordem de grandeza de uma secao.
TAMANHO_ALVO_FALLBACK = 200

# Titulos canonicos de secao -> tipo.
# A "abertura" NAO aparece aqui: ela nao tem titulo proprio; e definida como
# tudo o que vem antes do primeiro titulo detectado.
# Note a secao "fundamentacao": ela aceita DUAS variantes de titulo, separadas
# por "|" no mesmo padrao. Assim "DO DIREITO" e "DA FUNDAMENTACAO JURIDICA"
# caem no mesmo tipo - foi por isso que o dataset usa as duas, para exercitar
# este ramo.
SECTION_PATTERNS = [
    ("fatos",         r"DOS\s+FATOS"),
    ("fundamentacao", r"DO\s+DIREITO|DA\s+FUNDAMENTAÇÃO\s+JURÍDICA"),
    ("pedidos",       r"DOS\s+PEDIDOS"),
]

# Um titulo e uma LINHA INTEIRA em caixa alta que bate com um dos padroes acima.
#   ^...$ + re.MULTILINE  => a linha toda, nao um trecho no meio de um paragrafo.
#   [ \t]* nas pontas     => tolera espacos/identacao sobrando.
# Cada padrao individual e embrulhado em (?:...) para que o "|" interno da
# fundamentacao nao "vaze" e se misture com os outros padroes.
_HEADER_RE = re.compile(
    r"^[ \t]*(?P<titulo>"
    + "|".join("(?:%s)" % p for _, p in SECTION_PATTERNS)
    + r")[ \t]*$",
    re.MULTILINE,
)

# Para classificar um titulo ja casado de volta no seu tipo.
_TIPO_POR_PADRAO = [(re.compile(p), tipo) for tipo, p in SECTION_PATTERNS]


def _classifica_titulo(titulo):
    """Dado o texto de um titulo casado, devolve o tipo de secao correspondente."""
    titulo = titulo.strip()
    for regex, tipo in _TIPO_POR_PADRAO:
        if regex.fullmatch(titulo):
            return tipo
    return "indefinido"  # nao deve ocorrer: o titulo veio do proprio _HEADER_RE


def _chunk(doc_id, tipo, texto):
    """Monta um chunk no formato do contrato de saida."""
    return {"doc_id": doc_id, "tipo": tipo, "texto": texto}


def chunk_text(doc_id, texto):
    """Divide o texto de uma peticao em chunks por secao.

    Retorna lista de { "doc_id", "tipo", "texto" }. O texto de cada chunk e o
    CORPO da secao (sem a linha do titulo, que e puro jargao estrutural).
    """
    titulos = list(_HEADER_RE.finditer(texto))

    if not titulos:
        return _fallback_por_paragrafos(doc_id, texto)

    chunks = []

    # Abertura = tudo antes do primeiro titulo (endereçamento, qualificacao etc.).
    abertura = texto[: titulos[0].start()].strip()
    if abertura:
        chunks.append(_chunk(doc_id, "abertura", abertura))

    # Cada titulo abre uma secao que vai ate o inicio do proximo titulo
    # (ou ate o fim do documento, no ultimo).
    for i, m in enumerate(titulos):
        tipo = _classifica_titulo(m.group("titulo"))
        inicio = m.end()
        fim = titulos[i + 1].start() if i + 1 < len(titulos) else len(texto)
        corpo = texto[inicio:fim].strip()
        if corpo:
            chunks.append(_chunk(doc_id, tipo, corpo))

    return chunks


def _fallback_por_paragrafos(doc_id, texto):
    """Fallback para documentos sem titulos canonicos.

    Quebra por paragrafos (linhas em branco) e agrupa ate ~TAMANHO_ALVO_FALLBACK
    palavras por chunk. Todos saem com tipo "indefinido".
    """
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]

    chunks = []
    buffer = []
    palavras = 0
    for p in paragrafos:
        buffer.append(p)
        palavras += len(p.split())
        if palavras >= TAMANHO_ALVO_FALLBACK:
            chunks.append(_chunk(doc_id, "indefinido", "\n\n".join(buffer)))
            buffer, palavras = [], 0
    if buffer:  # sobra do ultimo grupo
        chunks.append(_chunk(doc_id, "indefinido", "\n\n".join(buffer)))

    return chunks


def chunk_file(path, doc_id=None):
    """Conveniencia: le um .txt e aplica chunk_text.

    Se doc_id nao for dado, usa o nome do arquivo sem extensao (ex.: maria.txt -> maria).
    """
    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        return chunk_text(doc_id, f.read())


if __name__ == "__main__":
    import sys

    def _mostra(chunks):
        for c in chunks:
            preview = c["texto"].replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:70] + "..."
            print(f"  [{c['tipo']:>13}] ({len(c['texto'].split()):>3} palavras) {preview}")

    if len(sys.argv) > 1:
        # Uso: python chunking.py caminho1.txt caminho2.txt ...
        for path in sys.argv[1:]:
            chunks = chunk_file(path)
            print(f"\n{path}  ->  {len(chunks)} chunks")
            _mostra(chunks)
    else:
        # Demo padrao: roda sobre alguns docs e forca o fallback.
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
        for nome in ["maria", "pedro", "fernando", "lucas", "camila", "renata"]:
            path = os.path.join(docs_dir, nome + ".txt")
            chunks = chunk_file(path)
            print(f"\n{nome}  ->  {len(chunks)} chunks  (tipos: {[c['tipo'] for c in chunks]})")
            _mostra(chunks)

        print("\n--- Fallback (texto sem titulos canonicos) ---")
        texto_sem_titulos = (
            "Trata-se de peticao sem qualquer titulo em caixa alta.\n\n"
            "Aqui vai um segundo paragrafo, tambem sem cabecalho de secao, "
            "apenas para simular um documento mal formatado.\n\n"
            "E um terceiro paragrafo encerrando o texto."
        )
        chunks = chunk_text("doc_mal_formatado", texto_sem_titulos)
        print(f"doc_mal_formatado  ->  {len(chunks)} chunks  (tipos: {[c['tipo'] for c in chunks]})")
        _mostra(chunks)
