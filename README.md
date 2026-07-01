# POC — Similaridade entre Documentos Jurídicos

> Prova de conceito. **Não** é um produto pronto: é um experimento controlado, com dataset
> pequeno e sintético, feito para validar um insight e encontrar os próprios limites.

---

## 1. O que é e o insight central

O sistema recebe uma petição inicial nova e a compara com um acervo de petições existentes,
classificando a **relação** entre cada par em um de quatro rótulos:

| Rótulo | Significado |
|---|---|
| `copia` | Mesmo caso, texto praticamente idêntico (só mudam dados pessoais). |
| `versao` | Mesmo caso, texto editado/reescrito. |
| `parente_de_tese` | **Mesma tese jurídica, fatos diferentes.** |
| `diferente` | Sem relação reaproveitável. |

O rótulo mais valioso é **`parente_de_tese`**: é ele que permite **reaproveitar a defesa** de um
caso antigo num caso novo — mesma fundamentação jurídica aplicada a fatos distintos.

O insight central da POC é este: **a decisão não sai de um número único de similaridade, e sim
do _formato_ da evidência** — de _quais seções_ de um documento se parecem com _quais_ do outro.

- Uma **cópia** casa em tudo: fatos, fundamentação e pedidos.
- Um **parente de tese** casa só na **fundamentação** (a tese) e diverge nos **fatos** (é outro caso).
- Um documento **diferente** coincide apenas no texto processual padrão (endereçamento, citações de lei genéricas).

A fundamentação carrega a tese reaproveitável; os fatos distinguem "mesmo caso" de "tese parecida".
Decidir pela **forma da evidência**, e não por um escalar, é o coração do projeto.

---

## 2. Arquitetura em duas camadas

O sistema separa **medir** de **decidir** — deliberadamente, para que a decisão seja explicável.

### Camada de medição
- **Embeddings por trecho:** cada seção da petição vira um vetor (`text-embedding-3-small`, 1536 dims).
- **Matriz de correspondências seção×seção:** para cada seção de conteúdo da query, encontra-se a
  seção mais parecida do candidato por **similaridade de cosseno**. O resultado não é um número, é
  um **perfil**: `fatos→fatos: 0.70`, `fundamentacao→fundamentacao: 0.79`, etc.
- **Força bruta, em memória:** o acervo é pequeno (20 documentos), então cada query é comparada
  contra todos os outros sem banco vetorial nem índice aproximado. Simples, exato e explicável —
  o funil de recuperação em escala é problema de produção, não desta POC.

### Camada de decisão
- Regras sobre o **perfil por seção** produzem o rótulo. Os eixos têm peso diferente por **função
  jurídica**, não por ajuste:
  - `fundamentacao` **alta** + `fatos` **alta** → mesmo caso (`copia` ou `versao`).
  - `fundamentacao` **alta** + `fatos` **baixa** → `parente_de_tese`.
  - `fundamentacao` **baixa** → `diferente`.
  - Desempate `copia` × `versao`: sobreposição **literal** do texto via `difflib` (não semântica).

### O filtro anti-jargão
Nem toda seção que se parece indica relação real. A **abertura** ("EXCELENTÍSSIMO SENHOR DOUTOR
JUIZ...") é texto padrão que aparece em quase toda petição — duas petições sem qualquer relação já
pontuam alto nela. Por isso **a abertura é excluída** da matriz. E os **pedidos** são
**despriorizados**: também são forma processual padronizada (toda ação de consumo pede indenização
do mesmo jeito). A evidência disso apareceu nos dados — o `pedidos` de um impostor casou em **0.898**,
_mais alto_ que o de um parente verdadeiro (**0.824**). Uma seção que pontua na direção errada é
ruído, não sinal, e não pode entrar como discriminante. Por isso as regras tocam apenas
`fundamentacao` e `fatos`.

### Pipeline por fases
```
dataset estático (.txt + ground_truth.json)   data/
   └─ chunking por seção (regex de títulos)    src/chunking.py
        └─ embeddings com cache local          src/embeddings.py
             └─ matching seção×seção (cosseno)  src/matching.py
                  └─ decisão (perfil → rótulo)  src/decision.py
                       └─ saída estruturada      src/decision.py + demo.py
                            └─ avaliação          src/evaluate.py
```

O dataset é **estático e versionado** (não gerado por LLM em runtime), para dar controle e
reprodutibilidade. Os embeddings são **cacheados em disco por hash do texto**, então só há chamada
de API para textos novos — execuções repetidas não gastam nada.

---

## 3. A jornada — em três atos

Esta é a parte que importa: a POC não foi só construída, foi **testada até revelar os próprios limites**.

### Ato 1 — O design
A aposta foi decidir pelo **perfil por seção**, e não por um número único. Uma escolha de
arquitetura acompanhou essa aposta: **priorizar recall sobre precision**. O raciocínio é de
negócio — o custo de _esconder_ um caso reaproveitável (uma defesa que poderia ter sido reusada e
não foi) é maior que o de um _alerta falso_, que o revisor humano descarta em segundos. Então, na
dúvida entre `parente_de_tese` e `diferente`, o sistema pende para incluir.

### Ato 2 — Funcionou nos pares curados
Sobre os 16 pares curados do gabarito (cada variação contra o seu próprio original), a
classificação acertou **16/16**: a matriz de confusão 4×4 saiu **perfeitamente diagonal**. Com o
**par certo em mãos**, os quatro rótulos são distinguidos sem erro. O `parente_de_tese` foi
reconhecido mesmo quando o parente foi escrito com vocabulário deliberadamente diferente do
original — sinal de que o sistema captura **parentesco semântico**, não sobreposição de palavras.

### Ato 3 — O acervo inteiro revelou o custo
Aí veio o teste realista: rodar **cada query contra o acervo inteiro**, todas as famílias
misturadas. A precisão de "relacionado" despencou para **0.28** — mas o **recall ficou em 1.00**.

Esses dois números lado a lado contam a história. Recall 1.00 significa que o sistema **não perdeu
nenhum caso útil** (capturou 48 de 48 parentes verdadeiros); a precisão baixa vem de _incluir
demais_, não de _esconder_. Isso **não é um bug** — é o trade-off pró-cobertura do Ato 1
**funcionando como projetado**.

O achado decisivo veio ao diagnosticar _para onde_ iam os 122 falsos positivos: **100% eram
cross-família**, e todos petições consumeristas. A causa é jargão **dentro da própria
fundamentação** — "relação de consumo / CDC / dano moral" aparece em toda petição de consumo e cria
um **piso de similaridade cross-família de ~0.70**, acima do corte de decisão. O filtro anti-jargão
tinha removido a abertura, mas o jargão também mora _dentro_ da seção que decide.

A prova de que o problema é estrutural, e não de ajuste de corte:

> Um **parente verdadeiro** (`patricia × sandra_versao`) pontua **0.667** na fundamentação —
> **abaixo** de um **falso positivo** (`bruno × joao`), que pontua **0.751**.

As distribuições se **sobrepõem**. Nenhum limiar global absoluto consegue, sozinho, deixar os
verdadeiros de um lado e os falsos do outro.

---

## 4. O diagnóstico e o remédio (provado pelos dados)

O mesmo experimento que expôs o problema também mostrou o caminho da solução. Duas métricas:

- **MRR = 1.00**
- **recall@k = 1.00** (com k = tamanho da lista devolvida por query)

Elas dizem que, **em cada query individual, o parente verdadeiro já está no TOPO do ranking** — o
candidato de maior fundamentação é sempre um verdadeiro; os falsos vêm logo _depois_. A sobreposição
que quebra o sistema é um problema do **limiar global absoluto**, não da **ordenação por query**.

Daí o remédio, decidido com os dados na mão:

- **Ranking relativo por query (top-k):** em vez de "toda fundamentação ≥ 0.65 é parente", pegar
  os _k_ candidatos mais bem ranqueados de cada query. Como os verdadeiros já estão em cima
  (MRR 1.00), o top-k **recuperaria a precisão praticamente sem custo de recall**.
- **Remoção do boilerplate da fundamentação:** tirar as frases-jargão antes de embeddar, para
  baixar o piso cross-família.

**Este remédio não foi implementado** (restrição de tempo, e a POC foi deliberadamente congelada no
estado da Fase 7 para servir de baseline "antes"). Mas os dados — especialmente o MRR 1.00 — já
demonstram que ele funcionaria.

Vale um fechamento: essa conclusão **reconverge com a arquitetura de funil de recuperação** que o
desenho original previa. A POC, ao ser testada de verdade, **redescobriu empiricamente** a
necessidade daquele funil — o ranking relativo por query _é_ a primeira etapa dele.

---

## 5. Limitações (honestas)

- **Dataset pequeno e sintético.** São 20 documentos, todos escritos à mão pelo autor. Serve para
  validar o insight e expor mecanismos, **não** para afirmar desempenho em produção.
- **Cortes calibrados neste dataset.** Os limiares (`fundamentacao ≥ 0.65`, `fatos ≥ 0.85`,
  `difflib ≥ 0.90`) foram ajustados a mão até os pares conhecidos saírem certos. Há risco de
  **overfitting**; em produção viriam de dados reais rotulados, via validação cruzada. A fronteira
  **`parente_de_tese` × `diferente`** é a mais frágil — a janela entre parentes e impostores era
  estreita já nos pares curados, e o Ato 3 mostrou que ela **não se sustenta** no acervo inteiro.
- **Chunking por regex de títulos canônicos.** Funciona porque as petições seguem uma estrutura
  fixa (`DOS FATOS`, `DO DIREITO`/`DA FUNDAMENTAÇÃO JURÍDICA`, `DOS PEDIDOS`), com fallback por
  parágrafo. A versão real seria **corte por argumento via LLM** — já mapeado como próximo passo.
- **Distinção `copia` × `versao` com poucos exemplos.** Só há 4 versões, e o corte de `difflib`
  (0.90) não pôde ser calibrado na fronteira inferior por falta de casos limítrofes. Foi ajustado
  apenas para que as cópias conhecidas (difflib 0.93–0.96) não fossem confundidas com versões.

---

## 6. Como rodar

**Requisitos:** Python 3, uma chave da OpenAI, e as dependências:

```bash
python -m pip install openai numpy
```

**Chave da API** (necessária só na 1ª execução, para gerar os embeddings; depois o cache assume):

```powershell
# PowerShell (Windows)
$env:OPENAI_API_KEY = "sk-..."
```

**Rodar o demo — a narrativa da Maria:**

```bash
python demo.py
```
Mostra, de cima a baixo: entra a petição da Maria, o sistema varre o acervo, e para cada candidato
relacionado imprime o rótulo, os scores por seção, o motivo rastreável e a recomendação acionável.
A cópia, a versão e o parente aparecem; o impostor (Ana) é **visto e descartado** — evidência de que
o jargão dos `pedidos` não arrasta a decisão.

**Rodar a avaliação — as métricas:**

```bash
python src/evaluate.py
```
Imprime os três blocos: a **matriz de confusão 4×4** (classificação 16/16), a **precisão 0.28 ao
lado do recall 1.00** (o trade-off pró-cobertura), e o **diagnóstico** com a sobreposição de
distribuições (o verdadeiro 0.667 abaixo do falso 0.751).

> Cada módulo em `src/` também tem seu próprio demo executável (`python src/chunking.py`,
> `python src/embeddings.py`, etc.) para inspecionar cada fase isoladamente.

---

_POC construída em fases; congelada no estado da Fase 7 (avaliação) como baseline "antes" de qualquer remédio._
