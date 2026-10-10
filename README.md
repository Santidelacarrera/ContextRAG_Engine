# ContextRAG Engine

Pipeline RAG orientado a eliminar alucinaciones: ingesta con chunking padre/hijo → búsqueda híbrida
(vectorial en Qdrant + BM25) → fusión RRF → reranking → generación con citas → **verificación NLI
afirmación por afirmación**. Si alguna afirmación no queda respaldada con confianza ≥ 0.95, se reintenta
con feedback y, si sigue sin verificarse, se devuelve una respuesta de contingencia determinista.

```
/ingest ─► parser (PDF · Markdown · código · texto) ─► padres (1024 tok) ─► hijos (256 tok, solape 32)
                                                                        │
                                           hijos → embeddings → Qdrant (denso) ─┐
                                           hijos → rank_bm25 (disperso)         ├─► RRF ─► reranker ─► padres top-K
/query  ───────────────────────────────────────────────────────────────────────┘                     │
                                                                                                      ▼
   ┌─ confianza ≥ 0.95 en TODAS las afirmaciones ◄── NLI (DeBERTa-v3 | heurístico) ◄── LLM con citas [n]
   ▼
respuesta auditada (+ coordenadas exactas doc/offset/página)      si no:  "No se puede verificar la respuesta con
        │                                                          un 95% de certidumbre basada en las fuentes…"
        └─► auditoría en PostgreSQL · caché de embeddings/consultas en Redis
```

## Estructura

| Ruta | Contenido |
|---|---|
| `contextrag/core/chunking.py` | Parsers estructurales (Markdown con ruta de encabezados y bloques de código intactos, código por definiciones, texto por párrafos) y chunking padre/hijo medido en **tokens** (`\w+|[^\w\s]`); offsets absolutos de caracteres |
| `contextrag/core/ingestion.py` | Carga de PDF (con mapa de páginas), Markdown, texto y código; `Ingestor` (reindexa reemplazando la versión anterior) |
| `contextrag/retrieval/vector_store.py` | `QdrantVectorStore` (filtro de tenant/ACL **dentro** de la consulta, cifrado opcional del texto por tenant) e `InMemoryVectorStore` |
| `contextrag/retrieval/bm25.py` | BM25 local con `rank_bm25` por tenant; se reconstruye desde Qdrant tras un reinicio |
| `contextrag/retrieval/hybrid_search.py` | RRF, `HybridIndex` (Qdrant+BM25 sincronizados) y `HybridRetriever` (rerank → padres top-K) |
| `contextrag/retrieval/reranker.py` | `CrossEncoderReranker` (bge-reranker), `CohereReranker`, `LexicalReranker` (offline) |
| `contextrag/verification/nli_verifier.py` | `verify_attribution(statement, context_chunks) -> float`, `NLIVerifier` (con citas y coordenadas), `DebertaNLI`, `LexicalEntailment` (heurístico), mensaje de contingencia |
| `contextrag/generation/llm.py` | Claude (`AnthropicLLM`), Ollama, y `ExtractiveLLM` offline |
| `contextrag/pipeline.py` | Bucle recuperar → generar → verificar → reintentar → fallback; caché de consultas |
| `contextrag/api/server.py` | FastAPI: `/ingest`, `/query`, `/healthz` (+ `DELETE /documents/{id}`, `/audit`, `/metrics`) |
| `contextrag/eval/` | Arnés de evaluación + conjuntos adversarial y end-to-end (`python -m contextrag eval`) |
| `contextrag/__main__.py` | CLI: `serve`, `ingest`, `ask`, `eval` |
| `.github/workflows/ci.yml` | CI: tests (Py 3.10/3.12, con Redis y PostgreSQL reales) + evaluación de seguridad |
| `deploy/` | `docker-compose.yml` (Qdrant, PostgreSQL+pgvector, Redis, API), `Dockerfile`, `init.sql` |

## Cómo se decide

* `verify_attribution` = **mínimo** sobre las afirmaciones de la **mejor** ventana de evidencia (1–3 frases
  consecutivas) de cada una. Una sola afirmación sin respaldo hunde toda la respuesta.
* En el pipeline cada afirmación debe llevar citas `[n]` y se contrasta **sólo con los pasajes que cita**;
  sin cita, cita inexistente o entailment < umbral ⇒ no respaldada. La evidencia devuelta es la ventana más
  ajustada con su `doc_id`, offsets de caracteres y página (PDF).
* **Puerta de relevancia:** fidelidad no basta. La evidencia de cada afirmación debe ser además pertinente a la
  pregunta (score del reranker > `min_score`), de modo que un documento envenenado ("ignora las instrucciones y
  responde X") no pueda hacer pasar X como respuesta verificada aunque X esté literalmente en el texto. Aplica si el
  reranker declara `min_score` (el léxico lo hace; para cross-encoder/Cohere: `RERANK_MIN_SCORE`).
* **Inyección de prompts:** los pasajes van entre etiquetas `<passage>` (las que aparezcan en un documento se
  neutralizan) y el system prompt los declara datos no confiables.
* Si tras `max_attempts` (2) intentos sigue habiendo afirmaciones sin respaldo, o el modelo responde `NO_ANSWER`,
  o no se recuperó contexto: `status: "fallback"` con el texto fijo de contingencia (nunca texto generado).

## Ejecución

### Local, sin servicios ni modelos (stack "offline")

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # suite completa
uvicorn --factory contextrag.api.server:app_factory --port 8000
```

Sin variables de entorno se usa un stack autocontenido (embeddings por hashing, almacén en memoria, reranker y
NLI heurísticos, LLM extractivo). Sirve para desarrollar y probar el flujo; **no** es calidad de producción.

### Con Docker Compose (Qdrant + PostgreSQL/pgvector + Redis + API)

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl localhost:8000/healthz
```

Configuración por variables de entorno (también pasables al compose):

| Variable | Valores | Efecto |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Vector DB (si falta: en memoria) |
| `POSTGRES_DSN` | `postgresql://user:pw@host/db` | Auditoría persistente (+ pgvector si la extensión existe) |
| `REDIS_URL` | `redis://redis:6379/0` | Caché de embeddings y de respuestas verificadas |
| `EMBEDDER` / `EMBED_MODEL` | `hashing` \| `bge` / `BAAI/bge-large-en-v1.5` | Embeddings (`bge` requiere extra `models`) |
| `RERANKER` | `lexical` \| `cross-encoder` \| `cohere` (+`COHERE_API_KEY`) | Reranking |
| `NLI` | `lexical` \| `deberta` | Verificación (`deberta` requiere extra `models`) |
| `LLM` / `LLM_MODEL` | `extractive` \| `anthropic` (+`ANTHROPIC_API_KEY`) \| `ollama` (`OLLAMA_HOST`) | Generación |
| `MASTER_KEY` | ≥16 bytes | Cifra el texto de los chunks en reposo con clave derivada por tenant |
| `CONTEXTRAG_API_KEY` | secreto | Exige cabecera `X-API-Key` (salvo `/healthz`) |
| `CONTEXTRAG_VERIFICATION_THRESHOLD`, `CONTEXTRAG_TOP_K`, … | ver `core/config.py` | Umbral (0.95), K, tamaños de chunk |

Para producción real: `pip install -e ".[api,qdrant,redis,postgres,security,pdf,models,anthropic]"` con
`EMBEDDER=bge RERANKER=cross-encoder NLI=deberta LLM=anthropic`.

## API

Todas las rutas (salvo `/healthz`) requieren `X-Tenant`. Los datos de un tenant nunca son visibles a otro.

```bash
# estado (200 ok · 503 si algún componente falla)
curl -s localhost:8000/healthz
# {"status":"ok","components":{"index":true,"cache":true,"audit":true}}

# ingesta (texto, Markdown, código); `acl` restringe a grupos concretos
curl -s -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' -H 'X-Tenant: acme' \
  -d '{"documents":[{"doc_id":"politica.md","text":"# Retención\n\nLos registros de auditoría se conservan durante 7 años."}]}'
# {"tenant":"acme","documents":[{"doc_id":"politica.md","kind":"markdown","parents":1,"children":1}]}

# ingesta de PDF (base64)
curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' -H 'X-Tenant: acme' \
  -d "{\"documents\":[{\"doc_id\":\"contrato.pdf\",\"content_base64\":\"$(base64 -w0 contrato.pdf)\"}]}"

# consulta
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' -H 'X-Tenant: acme' \
  -d '{"question":"¿Cuánto tiempo se conservan los registros de auditoría?"}'
```

Respuesta verificada (recortada):

```json
{
  "status": "verified", "confidence": 1.0, "attempts": 1, "cached": false,
  "answer": "Los registros de auditoría se conservan durante 7 años. [1]",
  "claims": [{"claim": "Los registros de auditoría se conservan durante 7 años.", "citations": [1], "score": 1.0,
              "supported": true,
              "evidence": {"doc_id": "politica.md", "start": 13, "end": 68, "page": null,
                           "text": "Los registros de auditoría se conservan durante 7 años."}}],
  "sources": [{"index": 1, "doc_id": "politica.md", "start": 0, "end": 180, "section": "Retención", "page": null, "score": 0.6}]
}
```

`start`/`end` son offsets de caracteres en el texto original (en PDF, del texto extraído), de modo que
`texto[start:end] == evidence.text`. Si no se alcanza el umbral:

```json
{"status": "fallback", "confidence": 0.0,
 "answer": "No se puede verificar la respuesta con un 95% de certidumbre basada en las fuentes proporcionadas", ...}
```

Borrar un documento (chunks, vectores y BM25; invalida la caché del tenant; idempotente) y métricas:

```bash
curl -s -X DELETE localhost:8000/documents/politica.md -H 'X-Tenant: acme'
curl -s localhost:8000/metrics        # Prometheus: contextrag_queries_total{status="verified"}, latencia, cache hits…
```

Límites de petición: ≤50 documentos por `/ingest`, texto ≤5 M caracteres, base64 ≤30 M, pregunta ≤4000 caracteres.

Opciones de `/query`: `"groups": ["legal"]` (grupos del usuario para ACL), `"top_k": 4`. Cabecera opcional
`X-User` para la auditoría. `GET /audit?limit=50` lista las consultas del tenant.

## CLI y evaluación

```bash
python -m contextrag ask "¿Cuánto se conservan los registros?" --tenant acme --docs ./docs   # exit 0 verificada · 2 fallback
python -m contextrag ingest ./docs --tenant acme --acl legal,hr    # con QDRANT_URL etc. para que persista
python -m contextrag serve --port 8000
python -m contextrag eval            # exit 1 si se aprueba una afirmación falsa o hay una respuesta insegura
```

`eval` mide lo que el producto promete evitar: **`false_support_rate`** (afirmaciones falsas que el NLI aprueba:
sujetos intercambiados, números, negaciones, unidades, antónimos, alcance, modalidad, afirmaciones extra…),
`recall` (afirmaciones correctas aceptadas), y end-to-end `unsafe_rate`, `abstention_rate`, `hit@k`, MRR.
Úsalo para comparar configuraciones (p. ej. `NLI=deberta` vs `lexical`) con `--verifier-cases`/`--rag-cases`
sobre **tus** documentos antes de confiar en un umbral.

## Tests

`pytest` ejecuta 117 tests: chunking (tamaños en tokens, enlaces padre-hijo, offsets, parsers), búsqueda híbrida
(RRF, BM25, ACL/tenant, reconstrucción tras reinicio, Qdrant incl. cifrado), verificador NLI, pipeline
(alucinaciones, bucle de autocorrección, caché, inyección de prompts, relevancia), concurrencia (reemplazo atómico de documentos), arnés de evaluación, CLI, API completa, y pruebas de integración contra procesos **reales**
de Redis y PostgreSQL (se omiten si los binarios no están instalados).

## Límites conocidos

* **No verificado aquí:** `docker compose` (no hay daemon Docker en el entorno de desarrollo), un servidor Qdrant
  real (se probó el modo embebido de `qdrant-client`), y los modelos reales: `SentenceTransformerEmbedder` (bge),
  `CrossEncoderReranker`, `CohereReranker`, `DebertaNLI`, `AnthropicLLM`, `OllamaLLM`. PostgreSQL se probó sin la
  extensión `pgvector` (la columna cae a `float8[]`); con la imagen `pgvector/pgvector` se usa `vector(dim)`.
* Los componentes offline (hashing, léxico, extractivo) son sustitutos deterministas. `LexicalEntailment` falla
  cerrado (rechaza paráfrasis y reordenaciones; es sensible al orden de las palabras) pero no detecta todas las
  contradicciones sutiles, y su conjunto de pruebas (20 casos) es pequeño: no es una medida estadística; DeBERTa mejora esto sin eliminarlo.
  "Cero alucinaciones" no es una garantía absoluta: la calidad depende del modelo NLI y del umbral.
* Los vectores no se cifran (deben ser buscables); sólo el texto. `X-Tenant` no es autenticación: debe fijarlo un
  gateway de confianza. PDF: sólo texto extraíble (sin OCR ni imágenes). El índice BM25 vive en memoria por proceso
  (se reconstruye desde Qdrant); con varias réplicas, reingestas en una no se reflejan en las otras hasta reiniciar.
