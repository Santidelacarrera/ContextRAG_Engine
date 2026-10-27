# ContextRAG Engine

Pipeline RAG orientado a eliminar alucinaciones: ingesta con chunking padre/hijo → búsqueda híbrida
(vectorial en Qdrant + BM25) → fusión RRF → reranking → generación con citas → **verificación NLI
afirmación por afirmación**. Si alguna afirmación no queda respaldada con confianza ≥ 0.95, se reintenta
con feedback y, si sigue sin verificarse, se devuelve una respuesta de contingencia determinista.

```
/ingest ─► parser (PDF · Markdown · código · texto) ─► padres (1024 tok) ─► hijos (256 tok, solape 32)
                                                                        │
                                           hijos → embeddings → Qdrant (denso) ─┐
                                           hijos → BM25 (rank_bm25)  (disperso) ├─► RRF ─► reranker ─► padres top-K
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
| `contextrag/core/ingestion.py` | Carga de PDF (con mapa de páginas), HTML (texto visible; `<h1>`–`<h6>` → secciones), Markdown, texto y código; rechaza documentos sin texto extraíble (PDF escaneado) en vez de indexar nada en silencio; `Ingestor` (idempotente: no reindexa documentos sin cambios; reemplazo atómico en cambios) |
| `contextrag/retrieval/vector_store.py` | `QdrantVectorStore` (filtro de tenant/ACL **dentro** de la consulta, cifrado opcional del texto por tenant) e `InMemoryVectorStore` |
| `contextrag/retrieval/bm25.py` | BM25 local por tenant: `BM25Index` (`rank_bm25`, por defecto) e `IncrementalBM25Index` (mismas puntuaciones, actualizable in situ); ambos se reconstruyen desde Qdrant tras un reinicio |
| `contextrag/retrieval/hybrid_search.py` | RRF, `HybridIndex` (Qdrant+BM25 sincronizados) y `HybridRetriever` (rerank → padres top-K) |
| `contextrag/retrieval/reranker.py` | `CrossEncoderReranker` (bge-reranker), `CohereReranker`, `LexicalReranker` (offline) |
| `contextrag/verification/nli_verifier.py` | `verify_attribution(statement, context_chunks) -> float`, `NLIVerifier` (con citas y coordenadas), `DebertaNLI`, `LexicalEntailment` (heurístico), mensaje de contingencia |
| `contextrag/generation/llm.py` | Claude (`AnthropicLLM`), Ollama, y `ExtractiveLLM` offline |
| `contextrag/pipeline.py` | Bucle recuperar → generar → verificar → reintentar → fallback; caché de consultas |
| `contextrag/api/server.py` | FastAPI: `/ingest`, `/query`, `/healthz` (+ `GET /documents`, `DELETE /documents/{id}`, `DELETE /tenant`, `/audit`, `/metrics`) |
| `contextrag/api/retrieval.py` | `POST /api/v1/retrieve` para otros servicios: fragmentos con metadatos y puntuaciones, borrador extractivo **no verificado**, auditoría de cada recuperación dentro de una transacción con `app.tenant` fijado (RLS). Falla cerrado (503) si no puede auditar |
| `contextrag/db.py` | `TenantDB`: pool de conexiones con el rol de aplicación (`POSTGRES_DSN`), una transacción por tenant y tabla `retrieval_audit` solo-inserción |
| `contextrag/api/auth.py`, `ratelimit.py` | Claves ligadas a tenant/grupos/rol (solo digests SHA-256) y límite de tasa por tenant |
| `contextrag/eval/` | Arnés de evaluación + conjuntos adversarial y end-to-end (`python -m contextrag eval`) |
| `contextrag/__main__.py` | CLI: `serve`, `ingest`, `ask`, `eval` (`--sweep` calibra el umbral), `keygen`, `erase`, `audit-purge` |
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
| `POSTGRES_DSN` | `postgresql://user:pw@host/db` | Auditoría persistente (+ pgvector si la extensión existe). Debe ser un rol **no** superusuario, sin `BYPASSRLS` y no propietario de la tabla: así RLS aísla los tenants en la base de datos |
| `POSTGRES_ADMIN_DSN` | `postgresql://owner:pw@host/db` | Propietario del esquema: creación de tabla, particiones mensuales y retención (opcional; por defecto usa `POSTGRES_DSN`) |
| `REDIS_URL` | `redis://redis:6379/0` | Caché de embeddings y de respuestas verificadas |
| `BM25` | `rank` (defecto, `rank_bm25`) \| `incremental` | `incremental`: mismas puntuaciones que `rank_bm25` (probado) pero sin reconstruir el índice tras cada ingesta |
| `EMBEDDER` / `EMBED_MODEL` | `hashing` \| `bge` / `BAAI/bge-large-en-v1.5` | Embeddings (`bge` requiere extra `models`) |
| `RERANKER` | `lexical` \| `cross-encoder` \| `cohere` (+`COHERE_API_KEY`) | Reranking |
| `NLI` / `NLI_MODEL` | `lexical` \| `deberta` / ruta o id HF | Verificación (`deberta` requiere extra `models`) |
| `LLM` / `LLM_MODEL` | `extractive` \| `anthropic` (+`ANTHROPIC_API_KEY`) \| `ollama` (`OLLAMA_HOST`) | Generación |
| `MASTER_KEY` | ≥16 bytes | Cifra el texto de los chunks en reposo con clave derivada por tenant |
| `CONTEXTRAG_KEYS_FILE` | ruta a JSON | **Autenticación por clave ligada a tenant/grupos/rol** (recomendado; ver abajo) |
| `CONTEXTRAG_API_KEY` | secreto | Alternativa simple: una clave compartida; confía en `X-Tenant`/`groups` que envíe el llamante |
| `CONTEXTRAG_RATE_LIMIT_PER_MIN` | entero | Límite por tenant (HTTP 429 + `Retry-After`), por proceso |
| `CONTEXTRAG_VERIFICATION_THRESHOLD`, `CONTEXTRAG_TOP_K`, … | ver `core/config.py` | Umbral (0.95), K, tamaños de chunk |

Para producción real: `pip install -e ".[api,qdrant,redis,postgres,security,pdf,models,anthropic]"` con
`EMBEDDER=bge RERANKER=cross-encoder NLI=deberta LLM=anthropic`.

## Seguridad y autenticación

Sin claves ligadas, **quien llama declara su tenant y sus grupos** (`X-Tenant`, `groups`), es decir, sólo es seguro
detrás de un gateway que los fije. Con `CONTEXTRAG_KEYS_FILE` cada `X-API-Key` queda ligada a un tenant, unos
grupos ACL y un rol; el servidor ignora lo que el cliente afirme y rechaza (403) un `X-Tenant` distinto o `groups`
que la clave no tenga (el cuerpo sólo puede *restringir* los grupos, nunca ampliarlos).

```bash
python -m contextrag keygen --id ana --tenant acme --role admin --add-to keys.json   # imprime la clave UNA vez
python -m contextrag keygen --id lia --tenant acme --groups legal --add-to keys.json  # reader con grupo 'legal'
python -m contextrag keygen --id prometheus --role ops --add-to keys.json            # sólo /metrics
CONTEXTRAG_KEYS_FILE=keys.json python -m contextrag serve
curl -s -X POST localhost:8000/query -H 'X-API-Key: crag_…' -H 'Content-Type: application/json' -d '{"question":"…"}'
```

Roles: `reader` (consultar) · `admin` (consultar, ingestar, borrar, listar, `/audit`, supresión de su tenant) · `ops` (sólo `/metrics`).
El fichero guarda únicamente digests SHA-256 (permisos 0600). La auditoría registra el id de la clave, no una
cabecera falsificable.

## API

Todas las rutas (salvo `/healthz`) requieren `X-Tenant` (o una clave ligada). Los datos de un tenant nunca son
visibles a otro. `/ingest` es idempotente: un documento sin cambios (contenido, ACL y configuración de chunking
iguales) no recalcula embeddings (`"unchanged": true`; `"force": true` lo obliga). La ingesta escribe hijos primero
y padres al final, de modo que una ingesta interrumpida nunca se confunde con una completa.

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
curl -s localhost:8000/documents -H 'X-Tenant: acme'   # documentos indexados (doc_id, source, parents, chars, acl, doc_hash)
curl -s localhost:8000/metrics        # Prometheus: contextrag_queries_total{status="verified"}, latencia, cache hits…
```

**Supresión de datos y retención** (cumplimiento): `DELETE /tenant` borra de forma irreversible *todo* lo de un
tenant —documentos, vectores, índice BM25, respuestas en caché y registros de auditoría— y exige la cabecera
`X-Confirm-Tenant: <tenant>` con el nombre exacto (y rol `admin`; con claves ligadas sólo puede borrar el tenant de su
propia clave). `?keep_audit=true` conserva la auditoría (retención legal). Las claves API no se tocan: se gestionan en
el fichero de claves. Para la retención de la auditoría: `contextrag audit-purge --older-than-days 365`; para
borrar desde consola: `contextrag erase --tenant acme --yes [--keep-audit]` (con los backends de producción en el
entorno). Cada supresión queda registrada sólo en el log de la aplicación (no en la auditoría que se borra).

```bash
curl -s -X DELETE localhost:8000/tenant -H 'X-API-Key: crag_…' -H 'X-Confirm-Tenant: acme'
# {"tenant":"acme","documents":12,"audit_records":340}
```

Códigos de error: `422` documento ilegible o **sin texto extraíble** (p. ej. PDF escaneado: la versión previa del
documento queda intacta), `401/403` autenticación/rol/tenant, `429` límite de tasa, `503` + `Retry-After` si el LLM no
responde (error de generación: no se muestra el texto de contingencia porque no hubo verificación fallida, sólo una
caída; el detalle se registra en logs y no se devuelve al cliente).

Límites de petición: ≤50 documentos por `/ingest`, texto ≤5 M caracteres, base64 ≤30 M, pregunta ≤4000 caracteres.

Opciones de `/query`: `"groups": ["legal"]` (grupos del usuario para ACL), `"top_k": 4`. Cabecera opcional
`X-User` para la auditoría. `GET /audit?limit=50` lista las consultas del tenant.

## CLI y evaluación

Tras `pip install .` existe el comando `contextrag` (equivale a `python -m contextrag`).

```bash
python -m contextrag ask "¿Cuánto se conservan los registros?" --tenant acme --docs ./docs   # exit 0 verificada · 2 fallback
python -m contextrag ingest ./docs --tenant acme --acl legal,hr    # con QDRANT_URL etc. para que persista
python -m contextrag serve --port 8000
python -m contextrag eval --sweep    # + falsos respaldos/recall por umbral y el umbral más bajo seguro
python -m contextrag eval            # exit 1 si se aprueba una afirmación falsa o hay una respuesta insegura
```

`eval` mide lo que el producto promete evitar: **`false_support_rate`** (afirmaciones falsas que el NLI aprueba:
sujetos intercambiados, números, negaciones, unidades, antónimos, alcance, modalidad, afirmaciones extra…),
`recall` (afirmaciones correctas aceptadas), y end-to-end `unsafe_rate`, `abstention_rate`, `hit@k`, MRR.
`--sweep` calibra el umbral: con el heurístico, umbrales ≤0.8 ya aprueban afirmaciones parcialmente respaldadas, por
lo que 0.95 no es arbitrario; con DeBERTa la curva será otra, mídela. Úsalo para comparar configuraciones (p. ej. `NLI=deberta` vs `lexical`) con `--verifier-cases`/`--rag-cases`
sobre **tus** documentos antes de confiar en un umbral.

## Tests

`pytest` ejecuta 179 tests (170 + 2 omitidos si no están instalados torch/transformers; `ruff check` limpio): chunking (tamaños en tokens, enlaces padre-hijo, offsets, parsers), búsqueda híbrida
(RRF, BM25, ACL/tenant, reconstrucción tras reinicio, Qdrant incl. cifrado), verificador NLI, pipeline
(alucinaciones, bucle de autocorrección, caché, inyección de prompts, relevancia), concurrencia (reemplazo atómico de documentos), arnés de evaluación, CLI, autenticación por clave/roles/ACL, límite de tasa, ingesta idempotente, API completa, y pruebas de integración contra procesos **reales**
de Redis y PostgreSQL, y contra el código real de `transformers`/`sentence-transformers`/`torch` (`DebertaNLI`,
`CrossEncoderReranker`, `SentenceTransformerEmbedder`, cableado por variables de entorno) usando modelos diminutos con
pesos aleatorios construidos al vuelo: verifican la integración (tokenización por pares, índice de la etiqueta
*entailment*, lotes, truncado, normalización), **no la calidad**. Se omiten si faltan los binarios/extras.

## Rendimiento (medido, no estimado)

Máquina de desarrollo, stack en memoria, 1500 documentos de ~300 palabras (3000 fragmentos hijo): ingesta
≈1000 docs/s; consulta con índice caliente ≈ 29 ms (mediana); consulta inmediatamente después de una ingesta
≈ 210 ms, porque `rank_bm25` es inmutable y el índice BM25 del tenant se reconstruye (la tokenización ya se
cachea por fragmento). Esa reconstrucción crece linealmente con el tamaño del tenant, por eso existe
`BM25=incremental` (`IncrementalBM25Index`): un índice invertido que se actualiza in situ, con puntuaciones
idénticas a `rank_bm25.BM25Plus` (verificado en tests, incluidas actualizaciones y borrados) y sin coste de
reconstrucción. No se ha medido con Qdrant ni con modelos reales.

## Límites conocidos

* **No verificado aquí:** `docker compose` (no hay daemon Docker en el entorno de desarrollo), un servidor Qdrant
  real (se probó el modo embebido de `qdrant-client`), los **pesos** reales (Hugging Face está bloqueado en el entorno:
  la calidad de bge/bge-reranker/DeBERTa no se ha medido; sólo su integración, con modelos diminutos), y
  `CohereReranker`, `AnthropicLLM`, `OllamaLLM` (requieren red/claves). PostgreSQL se probó sin la
  extensión `pgvector` (la columna cae a `float8[]`); con la imagen `pgvector/pgvector` se usa `vector(dim)`.
* Los componentes offline (hashing, léxico, extractivo) son sustitutos deterministas. `LexicalEntailment` falla
  cerrado (rechaza paráfrasis y reordenaciones; es sensible al orden de las palabras) pero no detecta todas las
  contradicciones sutiles, y su conjunto de pruebas (20 casos) es pequeño: no es una medida estadística; DeBERTa mejora esto sin eliminarlo.
  "Cero alucinaciones" no es una garantía absoluta: la calidad depende del modelo NLI y del umbral.
* Los vectores no se cifran (deben ser buscables); sólo el texto. `X-Tenant` no es autenticación: debe fijarlo un
  gateway de confianza. PDF: sólo texto extraíble (sin OCR ni imágenes). El índice BM25 vive en memoria por proceso
  (se reconstruye desde Qdrant); con varias réplicas, reingestas en una no se reflejan en las otras hasta reiniciar.

## Operación: retención, migraciones y observabilidad

**Retención de auditoría.** `answer_audit` y `retrieval_audit` están particionadas por mes (UTC). Para conservar los
últimos N meses calendario (incluido el actual):

```bash
POSTGRES_ADMIN_DSN=postgresql://contextrag:…@postgres:5432/contextrag \
  python -m contextrag retention --months 12 --dry-run   # informa, no cambia nada
POSTGRES_ADMIN_DSN=… python -m contextrag retention --months 12
```

Las particiones completas anteriores al corte se eliminan con `DROP TABLE` (operación de catálogo, sin borrado fila a
fila); las filas sueltas anteriores al corte se borran. Cada tabla se procesa en su propia transacción. No hay archivado
previo: lo eliminado no se recupera. Programación sugerida (cron, el día 1 de cada mes a las 03:00):
`0 3 1 * * cd /app && POSTGRES_ADMIN_DSN=… python -m contextrag retention --months 12`. Usa el DSN del propietario, nunca el
del rol de la aplicación.

**Migración 001 (bases existentes).** Si `retrieval_audit` se creó como tabla no particionada (versiones anteriores),
la aplicación se niega a arrancar hasta aplicarla. Es una transacción única que renombra la tabla antigua a
`retrieval_audit_legacy` (se conserva; bórrela a mano tras verificar los recuentos):

```bash
psql "$POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 -v app_role=contextrag_app \
     -f deploy/migrations/001_retrieval_audit_partitioned.sql
```

**Eventos y métricas.** Cada rechazo o fallo de `POST /api/v1/retrieve` emite una línea JSON en el logger
`contextrag.retrieve`, con `event` (`retrieve.served`, `retrieve.rejected` o `retrieve.failed`), `request_id` (también en
la cabecera `X-Request-Id`), `tenant_id`, `user_id`, `reason`, `http_status` y, en fallos de base de datos,
`error_class` (nunca el mensaje de error, que puede contener el DSN). Motivos: `tenant_mismatch` (403),
`rls_denied` (403), `database_unavailable` (503, con `Retry-After`), `database_not_configured` (503), `database_error`
(500). En `/metrics`: `contextrag_retrieve_rejected_total{reason}`, `contextrag_retrieve_failed_total{reason}` y
`contextrag_retrieve_served_total{status}`.

**Pool.** `TenantDB` usa `psycopg_pool` con `max_size=10` por defecto y espera de 5 s; con varios workers, la suma de los
`max_size` debe quedar por debajo de `max_connections` de PostgreSQL. Si el pool se agota, la petición falla cerrada
(503) en lugar de encolarse indefinidamente.

## Presupuesto de contexto, reranking y memoria conversacional

Esta sección describe lo que **está implementado y probado** en esta versión. Lo que no está implementado figura en
«Pendiente» más abajo.

```mermaid
flowchart TD
    Q[Pregunta] --> R[Recuperación híbrida: hijos Qdrant+BM25 → RRF]
    R --> RR[Reranker: timeout acotado; error si falla o responde incompleto]
    RR --> P[Padres ordenados por relevancia]
    P --> A[assemble_evidence: dedupe Jaccard → presupuesto de tokens → colocación en bordes]
    A --> G[LLM: instrucciones como prefijo cacheable, pasajes en el turno de usuario]
    G --> V[Verificación NLI + puerta de relevancia]
    V -->|verificado| OK[Respuesta con citas [n] = posición en el prompt]
    V -->|no verificado| FB[Fallback determinista]
    A -->|nada cabe en el presupuesto| FB
    RR -->|error| ERR[RerankerError: la petición falla, nunca se inventa un ranking]
```

**Presupuesto de evidencia (`contextrag/context/budget.py`).** Los padres recuperados se deduplican (Jaccard sobre
tokens de palabra, umbral `CONTEXTRAG_CONTEXT_DEDUPE_JACCARD`), se admiten en orden de relevancia mientras quepan en
`CONTEXTRAG_CONTEXT_BUDGET_TOKENS` menos `CONTEXTRAG_CONTEXT_SAFETY_MARGIN`. Si un padre no cabe, se usa **su mejor hijo**
como evidencia (compresión por span): sus coordenadas son absolutas en el documento, así que la atribución sigue siendo
exacta. La colocación es `descending` por defecto (el orden previo); `edges` (el mejor primero, el segundo al final) es
opt-in. Las citas `[n]` se renumeran según la posición final. El recuento de tokens es **una estimación** (palabras y
signos), no el tokenizador del modelo; por eso existe el margen de seguridad. Que `edges` mejore las respuestas **no está
medido**: la métrica de evaluación penaliza `edges` por construcción, así que no sirve para decidirlo.

**Conversación en `/query`.** Con `conversation_id` (y `MEMORY_DB` configurado), el turno se guarda después de la
respuesta. El prompt recibe las memorias relevantes y los turnos recientes en un bloque `<memory>` separado, marcado como
datos y **no citable**; sus etiquetas de pasaje se neutralizan. Las respuestas de conversación no usan la caché de
respuestas. `DELETE /conversations/{id}` borra una conversación del tenant (requiere permiso `erase`).

**Evaluación sin juez LLM (`contextrag/eval/context_metrics.py`).** `context_recall` y `average_precision` usan la
respuesta de referencia como oráculo y se calculan con la misma ruta de recuperación que `/query`. No son RAGAS: la
faithfulness, la relevancia de la respuesta y la precisión con juez necesitan un LLM, y este entorno no lo tiene. La línea
base está en `contextrag/eval/data/context_baseline.json` y un test falla si una métrica baja. Medido en el corpus sintético
(`contextrag/eval/long_corpus.py`, determinista, 6 documentos largos, 18 preguntas):

| Configuración | Recall | Precisión promedio |
|---|---|---|
| Padre 1024 / hijo 256, presupuesto 4096 (defecto) | 1.000 | 0.755 |
| Padre 2000 / hijo 400, presupuesto 4096 | 0.944 | 0.815 |
| Padre 1024 / hijo 256, presupuesto 2048 | 0.889 | 0.727 |

Con estos datos se mantienen los tamaños por defecto: son los que no pierden evidencia. Son cifras sintéticas y léxicas;
no dicen nada sobre documentos reales.

**Reranking (`contextrag/retrieval/reranker.py`).** `CohereReranker` exige una puntuación por candidato: un índice que no
llega, o llega dos veces, es un `RerankerError`, nunca un 0.0 inventado. `BoundedReranker` aplica un timeout
(`CONTEXTRAG_RERANKER_TIMEOUT_SECONDS`), valida el número de puntuaciones y envuelve el error sin exponer el texto de la
excepción. La fábrica lo aplica a los rerankers remotos o basados en modelo; el léxico local no lo necesita.

**Memoria conversacional (`contextrag/memory/`).** Almacén SQLite de la biblioteca estándar, durable entre reinicios
(WAL). Modelo de consistencia:
- Los turnos son inmutables y tienen `seq` contiguo; la versión de la conversación es el último `seq`.
  `append_turn(expected_version=…)` rechaza escritores obsoletos con `VersionConflict` y no escribe nada.
- Un job de compactación fija su rango `(compacted_through, through_seq]` al planificarse. Al confirmarse, en una sola
  transacción: comprueba que sigue siendo el dueño del claim, inserta sus memorias y avanza `compacted_through` **solo
  hacia delante**. Un job obsoleto o duplicado no puede retroceder el estado ni duplicar memorias. Los turnos que
  llegaron después del snapshot no se tocan.
- Los turnos fuente nunca se borran por compactación; cada memoria apunta a su `source_seq`. El borrado solo ocurre
  con `delete_conversation` (derecho de supresión).
- Resumidor extractivo: solo promueve frases **escritas por el usuario** que expresan decisión, preferencia o tarea,
  copiadas textualmente, con `origin=user_stated`, confianza 1.0. Nunca promueve lo que dice el asistente.
- Worker: pool de hilos acotado (`CONTEXTRAG_MEMORY_WORKERS`), cola acotada (`CONTEXTRAG_MEMORY_QUEUE_LIMIT`; más allá
  se rechaza con `QueueFull`), reintentos hasta `CONTEXTRAG_MEMORY_JOB_MAX_ATTEMPTS`, fallos registrados solo con un
  código de error. `recover()` reencola los jobs cuyo lease expiró tras un fallo del proceso.

**Caché de prompts (`contextrag/generation/llm.py`).** `AnthropicLLM` envía las instrucciones del sistema como bloque
con `cache_control: ephemeral`; los pasajes, la pregunta y el feedback van en el turno de usuario, fuera del prefijo.
Las métricas de caché que devuelve la API quedan en `last_usage`. No se comprueba la longitud mínima cacheable del
proveedor: un prefijo demasiado corto simplemente no se cachea. La caché de respuestas incluye ahora en su clave el
presupuesto, la colocación, el umbral, el tipo de LLM y su modelo, para no servir respuestas de otra configuración.

**Verificación.** Pruebas unitarias en `tests/test_context_budget.py`, `tests/test_reranker_guard.py`,
`tests/test_llm_request.py` y `tests/test_memory.py` (consistencia de versiones, job obsoleto, claim de zombi, reintentos,
cola acotada, recuperación tras reinicio, aislamiento por tenant y borrado). Ninguna prueba llama a un proveedor externo.

**Cifrado de la memoria conversacional.** Turnos y memorias se cifran con la clave del tenant derivada de `MASTER_KEY`
(el mismo esquema que el texto de los chunks). Cada valor cifrado va ligado a su posición (`turn:<conversación>:<seq>` o
`memory:<conversación>:<id>`), así que una fila copiada o movida dentro de la misma conversación no se acepta. Las lecturas
fallan cerrado (`MemoryIntegrityError`): clave incorrecta, otro tenant, manipulación, formato inválido o una fila en claro
leída por una tienda que cifra. No hay vuelta atrás a texto plano ni a la clave de otro tenant. Los identificadores,
roles, números de turno, estados de trabajo y códigos de error siguen en claro: son necesarios para el orden y la retención.
*Migración:* las filas anteriores al cifrado (`encrypted = 0`) no se leen hasta ejecutar `python -m contextrag memory-migrate`
(idempotente: un segundo pase no cambia nada). Tras cifrar hace `wal_checkpoint(TRUNCATE)` y `VACUUM`, y el modo
`secure_delete` pone a cero las páginas liberadas. Verificado en un fichero SQLite real: 2 turnos y 1 memoria cifrados,
segundo pase sin cambios, el marcador en claro ya no aparece en el fichero. **No implementado:** rotación de `MASTER_KEY`
(exige volver a cifrar con la clave nueva; hoy no hay herramienta para ello).

**Métricas del worker de compactación** (Prometheus, en `/metrics`; registro propio, sin etiquetas): 
`contextrag_memory_queue_depth` (gauge, trabajos en cola o en curso según la tienda), 
`contextrag_memory_compaction_duration_seconds` (histograma, reloj monótono, cada intento que reclamó un trabajo),
`contextrag_memory_compaction_jobs_served_total` (trabajos cuyo commit se hizo), 
`contextrag_memory_compaction_jobs_failed_total` (fallos terminales: se agotaron los intentos),
`contextrag_memory_compaction_retries_total` (intentos fallidos que se reencolan) y
`contextrag_memory_compaction_cancelled_total` (trabajos encolados cancelados al cerrar; quedan en la tienda y
`recover()` los reanuda). Un intento que se reintenta no cuenta como fallo terminal. Requiere el extra `memory`
(`prometheus-client`).

**Calibración (`python -m contextrag calibrate`).** Rejilla de chunks parent {1024, 2000} × child {256, 400}, umbral
`RERANK_MIN_SCORE` por precisión-recall y umbral NLI por frase. La política de chunks prioriza el recall (ningún candidato
puede perder recall frente al mejor) y después la precisión media. Cada sección lleva su estado: `computed` o
`insufficient_data` (menos de 50 casos por clase). El informe es JSON; `--proposal` escribe candidatos en una ruta
explícita para revisar y no toca `.env` ni los valores por defecto. **Resultado en el corpus sintético (verificado,
reproducible con semilla 7):** la rejilla recomienda 1024/256 (recall 1.000, precisión media 0.785). 2000/256 tiene
precisión media 0.870 pero recall 0.944 (un caso de 18 perdido). `RERANK_MIN_SCORE` y el umbral NLI quedan en
`insufficient_data`: no hay suficientes etiquetas.

**Hallazgo de esta revisión.** Los empates en el ranking se desempataban por el id de chunk, un hash que incluye el tenant.
La precisión media dependía del nombre del tenant (0.7346 a 0.7623 según el nombre). Ahora los empates se resuelven por
posición en el documento (RRF, BM25 y presupuesto de evidencia) y el valor es idéntico para cualquier tenant. Eso cambió
la línea base de 0.7546 a 0.7854 sin cambiar la configuración recomendada. La precisión media es sensible al desempate:
diferencias menores de ~0.03 en este corpus no son señal.

**RAGAS (`python -m contextrag ragas`).** Faithfulness (contra los párrafos padre completos), answer relevancy (necesita
embeddings de OpenAI; sin ellos queda *not run*, nunca 0) y context recall (frente a la respuesta de referencia). Concurrencia
acotada, tiempo de espera y reintentos por llamada, y un informe donde cada par caso-métrica es *scored*, *failed* o
*not run*. La media solo cuenta los puntuados. Código de salida 1 si alguna llamada al juez falló.
**Verificado sin proveedor:** la configuración se valida; `RagasScorer` construye las tres métricas con un cliente
Anthropic y un modelo de embeddings sin llamar a la red; la orquestación se prueba con un juez simulado (fallos, reintentos,
timeouts, concurrencia, valores fuera de rango). **No verificado:** cualquier puntuación real del juez. El test en vivo
(`RAGAS_INTEGRATION=1`) está escrito y omitido porque no hay claves ni acceso a los proveedores en este entorno. Con el
paquete `ragas` 0.4.3 hace falta `langchain-community<0.4`; el extra `ragas` lo fija.

**Límites de ingestión.** Antes de analizar un archivo se comprueba el tamaño (`CONTEXTRAG_MAX_DOCUMENT_BYTES`, 25 MB por
defecto; HTTP 413) y que la firma coincida con el tipo declarado: un `.pdf` sin cabecera `%PDF-`, o un archivo de texto que
empieza como PDF, ZIP, PNG, JPEG, GIF, ELF u OLE2, o que contiene bytes NUL, se rechaza (HTTP 422). Los PDF cifrados con
contraseña, corruptos o con más de `CONTEXTRAG_MAX_PDF_PAGES` páginas (500 por defecto) también. Cada rechazo lleva un código
estable (`too_large`, `signature_mismatch`, `binary_content`, `encrypted`, `too_many_pages`, `malformed_pdf`) y nunca repite
el contenido ni el texto de error del analizador. Un PDF con solo contraseña de propietario (clave de usuario vacía) se
abre. *Formatos:* PDF, Markdown, HTML, texto, código y los sufijos de `INGESTIBLE`; el texto se decodifica como UTF-8, de modo
que UTF-16 se rechaza como binario. *Límite conocido:* el archivo se lee entero en memoria (acotado por el tamaño máximo); no
hay extracción en streaming ni OCR (un PDF escaneado se rechaza por no tener texto).

**Controles de seguridad del prompt y de la salida.**
- El nombre de origen (`source`) de un documento es dato no confiable: se reduce a un atributo sin `<`, `>`, `"`, `&`,
  saltos de línea ni caracteres de control, con un máximo de 200 caracteres. Antes solo se sustituían las comillas dobles y un
  nombre hostil podía cerrar la etiqueta `<passage>`, abrir una línea `SYSTEM:` fuera de ella y fabricar un pasaje falso.
- Una respuesta con imagen markdown, `<img>`, `<script>`, `<iframe>`, `javascript:` y similares se rechaza aunque sea fiel a la
  evidencia (un documento envenenado podría usarla para exfiltrar datos desde un cliente que renderice imágenes). Se pide una
  reescritura y, si persiste, se abstiene. Los enlaces normales se permiten. Se paga con respuestas rechazadas sobre
  documentación que contenga HTML.
- **Cobertura de entidades.** El NLI demuestra que una afirmación se deduce de un pasaje, no que responda a la pregunta. Con la
  pregunta «¿Cuál es el periodo de retención de los logs de Halxanx?» (un servicio que no existe), el sistema devolvía como
  *verificadas* afirmaciones fieles sobre otros servicios. Ahora, si la pregunta nombra algo (palabra con mayúscula que no abre
  frase, o identificador con dígitos como `AES-256`) que la evidencia no menciona en absoluto, se abstiene sin llamar al modelo
  (`CONTEXTRAG_REQUIRE_QUESTION_ENTITIES`, activado por defecto). **Límites:** solo mira formas de superficie; un alias en los
  documentos («el sistema de tickets» para «Jira») provoca una abstención innecesaria, y una pregunta que no nombra nada, o que
  pide un atributo equivocado, no se detecta. No se ha medido la tasa de falsas abstenciones con preguntas reales.

**Benchmark reproducible (`python -m contextrag bench`).** Corpus sintético versionado (40 runbooks ficticios, 70 preguntas:
30 directas, 15 parafraseadas, 15 multi-documento con 2 a 8 documentos relevantes, 10 sin respuesta), generado con semilla 11
y verificable con `python -m contextrag.eval.bench_corpus --check`. Relevancia a nivel de documento. Pila sin conexión
(embedder hashing, reranker y NLI léxicos, generador extractivo): **describe el comportamiento de la tubería en este corpus, no
la calidad de recuperación con modelos reales ni con documentos reales.**

| estrategia | R@1 | R@5 | R@10 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.817 | 0.975 | 1.000 | 1.000 | 1.000 | 0.15 | 0.36 |
| dense (hashing) | 0.747 | 0.892 | 0.931 | 0.930 | 0.890 | 0.08 | 0.14 |
| hybrid (RRF) | 0.809 | 0.939 | 0.978 | 0.992 | 0.962 | 0.23 | 0.32 |
| hybrid + rerank | 0.817 | 0.971 | 0.994 | 1.000 | 0.997 | 1.48 | 2.22 |

Latencias en ms con 120 muestras por estrategia (60 preguntas × 2 repeticiones), dependientes de la máquina; no forman parte de
la línea base. **Lectura honesta:** en este corpus, que se resuelve por el nombre (término raro), BM25 solo iguala o supera al
híbrido; el reranker recupera casi toda la diferencia del RRF (nDCG@5 0.962 → 0.997) sin superar a BM25. La fusión no mejora
aquí y no se afirma que lo haga en general. Fundamentación sobre 45 preguntas con respuesta: verificadas 0.667 (las 15
parafraseadas se abstienen porque el generador extractivo no sabe reformular), correctas 0.667, **verificadas pero erróneas
0.0**, precisión de citas 0.70 (se mide sobre la evidencia en que se apoya cada afirmación; el generador extractivo une frases
de varios pasajes), cobertura de citas 1.0. Sobre las 10 sin respuesta: abstención 1.0 (0.5 con la cobertura de entidades
desactivada); son solo 10 preguntas y cinco nombran una entidad inexistente, el patrón al que apunta el control.
`--check-baseline` falla si una métrica baja del suelo registrado en `retrieval_bench_baseline.json`. **No medido:** consumo de
tokens y coste, rendimiento bajo concurrencia, calidad con embeddings semánticos y reranker entrenado, documentos reales.

**Dependencias.** `pip-audit` sobre las dependencias de ejecución encontró 18 avisos en `pypdf` 6.17.0 (todos corregidos en
6.21.2) y ninguno en los otros quince paquetes auditados; el suelo de `pypdf` pasó de `>=4` a `>=6.21.2` y una instalación nueva
desde los extras no tiene avisos conocidos. La CI ejecuta `pip-audit` en cada cambio y cada lunes.

**Cierre ordenado.** Al apagar la API se cierran el worker de compactación y las conexiones que la propia aplicación creó
desde el entorno (no las que le pasa quien la integra).

**Pendiente (no implementado o no verificado).**
- **Fuentes contradictorias:** no se detectan ni se comunican. El NLI sin conexión solo puntúa implicación; hace falta un
  modelo con la clase *contradicción* y etiquetas para validar un umbral.
- **Trazas OpenTelemetry y paneles:** no implementados. Hay ID de petición, métricas Prometheus y logs estructurados.
- **Comparación antes/después de la recuperación:** no aplica en esta iteración (no se cambió el algoritmo); el benchmark fija la
  línea base. Los empates se resuelven por posición desde el cambio anterior.
- **Benchmark con documentos reales y modelos reales:** no hay documentos autorizados en el repositorio.
- **Puntuaciones reales del juez RAGAS:** no ejecutadas (sin claves ni red a los proveedores).
- **Calibración sobre documentos reales:** los resultados son sintéticos; ningún valor de producción (tamaños de chunk,
  `RERANK_MIN_SCORE`, umbral NLI) puede darse por validado.
- **Reranker BGE:** el adaptador `CrossEncoderReranker` existe, pero no se ha verificado contra un modelo real: en el
  entorno de desarrollo `torch`, `sentence-transformers` y el host de Hugging Face no estaban disponibles.
- **Adaptador Cohere:** la validación de la respuesta está probada con un cliente simulado; no contra la API real.
- **Caché de prompts de Anthropic:** la forma de la petición está probada; el efecto real no se ha medido contra la API.
- **Memoria:** el resumen es solo extractivo; no hay caducidad (`expires_at`) ni redacción de datos personales, ni
  rotación de `MASTER_KEY`.
- **Redacción de datos sensibles en logs:** los logs de consulta no incluyen el contenido de documentos, pero no hay un filtro
  genérico de datos personales.

---

## Estado para un piloto jurídico (fase de endurecimiento)

La documentación completa del piloto (API, claves, retención, eliminación, incidentes, modelo de amenazas y lista de comprobación) está en [`docs/PILOT.md`](docs/PILOT.md). La licencia y su estado, en [`LICENSING.md`](LICENSING.md).

**Aviso.** Este sistema no está listo para documentos confidenciales de clientes. Lo que sigue separa lo implementado de lo
verificado con pruebas, y lo verificado con datos sintéticos de lo que solo se puede verificar con datos y servicios reales.
Pasar las pruebas no es evidencia de seguridad para producción, de cumplimiento de ninguna norma ni de confidencialidad
garantizada. No se ha hecho ninguna evaluación de seguridad externa.

### Arquitectura y flujo de datos

```
ingesta:   archivo → validación (tamaño, firma, PDF cifrado/páginas) → análisis (PDF/HTML sin texto oculto)
           → fragmentos padre/hijo (tokens) → embeddings → índice vectorial (Qdrant o memoria) + BM25
consulta:  pregunta → recuperación híbrida (filtro ACL antes de devolver) → reranking (opcional, con tiempo máximo)
           → cuarentena de pasajes con patrones de inyección → alias / ambigüedad / entidades
           → conflictos entre fuentes → generación (solo con evidencia) → verificación por afirmación (NLI)
           → control de salida (contenido activo) → respuesta con citas (documento, página, desplazamientos)
```

Cada etapa tiene un tramo de traza (`ingest`, `chunk`, `embed`, `index`, `retrieve`, `rerank`, `context`, `entities`,
`conflicts`, `generate`, `verify`, `verify.nli`, `citations`). Sin `CONTEXTRAG_TRACING=otel` solo se cuentan llamadas, errores
y milisegundos por etapa en `/metrics`; con él se exportan tramos por OpenTelemetry (ver «Observabilidad»).

### Controles de seguridad, con sus límites

| Control | Qué hace | Límite conocido |
|---|---|---|
| Autenticación por clave (`CONTEXTRAG_KEYS_FILE`) | Cada clave liga inquilino, casos (grupos) y rol; solo se guarda el hash SHA-256. | Sin rotación de claves de API ni expiración; la gestión del fichero es manual. |
| Casos dentro de un inquilino | Un admin con casos solo escribe y borra en sus casos; el listado, la auditoría y el borrado respetan el caso; un caso ajeno devuelve 404 idéntico al inexistente. | Un caso es un grupo ACL: no hay jerarquía ni roles por documento. |
| Identificador de conversación | Se liga al usuario: otro usuario con el mismo identificador obtiene una conversación distinta y vacía. | Solo para la memoria conversacional. |
| `CONTEXTRAG_REQUIRE_AUTH=true` | El servidor se niega a arrancar en modo abierto o con secreto compartido. | Hay que activarlo a mano; por defecto no está activado en el código, sí en `docker-compose`. |
| Inyección por el nombre del archivo | Los metadatos se reducen a texto sin etiquetas, comillas, saltos de línea ni caracteres de control. | Lista de caracteres; no es un clasificador. |
| Inyección en el contenido | Pasajes con patrones típicos (ignora instrucciones, «system:», revelar la clave, enviar la conversación a una URL) se retiran de la generación y se cuentan en `quarantined_passages_total`. | Lista de patrones: una frase redactada como hecho, otro idioma o un texto ofuscado la pasan. Un documento envenenado que afirme un dato falso se cita como cualquier otro. |
| Contenido activo en la respuesta | Imágenes markdown, `<img>`, `<script>`, `javascript:` y similares se rechazan. | Los enlaces normales se permiten: un enlace puede llevar datos en su URL. |
| HTML oculto | Se descarta el texto con `hidden`, `aria-hidden` o estilos que lo ocultan, y los comentarios. | El texto oculto por hoja de estilo externa o por color no se detecta sin renderizar. |
| Errores | Respuesta genérica con id de petición; los mensajes de excepciones y de proveedores no se devuelven ni se registran. | Los tracebacks solo aparecen con `CONTEXTRAG_DEBUG_ERRORS` (no usar con datos reales). |
| Redacción en logs | RUT, correos, teléfonos, números de causa, claves y contraseñas de URL se sustituyen antes de escribirse. | Patrones, no comprensión: nombres y direcciones no se detectan. |
| Cifrado en reposo | Texto de fragmentos (Qdrant), memoria conversacional y caché (Redis) con clave derivada por inquilino. | Vectores, metadatos (nombres de archivo, ids, fechas) y auditoría en claro. Clave en variable de entorno, sin gestor de secretos. |
| Rotación de `MASTER_KEY` | `MASTER_KEY_PREVIOUS` + `contextrag rotate-key`: descifra con todas las claves, re-cifra lo que queda con la anterior, verifica cada registro antes de reemplazarlo. | Probada con SQLite y con Qdrant en memoria, no con un Qdrant servidor ni con datos reales. |
| Caché no disponible | Se degrada a «sin caché», pero una invalidación fallida se recuerda y bloquea las lecturas de caché hasta aplicarse. | Verificado con un doble de caché, no con Redis real. |
| Contrapresión | Pasado `CONTEXTRAG_BUSY_WAIT_SECONDS`, 503 con `Retry-After`. | Límites por proceso; sin límite por usuario. |

**Auditoría de dependencias.** `pip-audit` sobre las dependencias de ejecución: sin avisos conocidos tras subir `pypdf` a
6.21.2 (la versión instalada tenía 18). La comprobación se repite en CI cada semana. No se ha auditado el contenedor con
ninguna herramienta de escaneo de imágenes.

### Qué debe configurarse antes de usar datos confidenciales

1. `MASTER_KEY` de 32 bytes o más, generada al azar, guardada fuera del repositorio y con copia de seguridad separada: sin
   ella los datos cifrados son irrecuperables.
2. `CONTEXTRAG_KEYS_FILE` con una clave por usuario y caso; `CONTEXTRAG_REQUIRE_AUTH=true`.
3. TLS delante de la API (el compose publica solo en 127.0.0.1; un proxy con TLS es responsabilidad del despliegue).
4. Contraseñas de Postgres, Redis y Qdrant distintas de las de ejemplo (el compose no arranca sin ellas).
5. Decidir la retención de la auditoría, de la memoria y de los logs; hoy la retención de auditoría existe
   (`python -m contextrag retention`) pero no hay una política para logs ni para la caché.
6. Revisar y aceptar que los vectores y los metadatos no están cifrados.

### Cómo ejecutar las pruebas y los benchmarks

```bash
pip install -e ".[dev,api,pdf,security,memory,qdrant,redis,postgres,otel]" mypy pip-audit
pytest -q                                        # pruebas unitarias e integración sin servicios externos
ruff check contextrag tests && mypy contextrag --ignore-missing-imports
python -m contextrag.eval.bench_corpus --check   # el corpus de recuperación es el generado
python -m contextrag.eval.entity_corpus --check
python -m contextrag bench --check-baseline      # recuperación y fundamentación (pila sin conexión)
python -m contextrag eval                        # evaluación de seguridad heredada
python -m contextrag loadtest --levels 1,4,8,16  # carga en proceso (comparar ejecuciones, no capacidad)
pip-audit --local --skip-editable               # auditoría de dependencias instaladas (la misma que CI)
```

Los modelos reales (`EMBEDDER=bge`, `RERANKER=cross-encoder|cohere`, `NLI=deberta`, `LLM=anthropic|ollama`) no se han
verificado en este repositorio. Para añadir uno: instálelo con su extra, configúrelo por variable de entorno, ejecute
`python -m contextrag bench` y `python -m contextrag ragas` (que necesita `RAGAS_PROVIDER` y la clave del proveedor) y compare
con la línea base. Ninguna comparación entre estrategias es válida sin esos datos en la misma máquina y con el mismo corpus.

### Cómo verificar citas y abstenciones

- Cada afirmación verificada trae `evidence` con documento, `start`, `end`, texto exacto y página cuando existe. Para
  comprobarla independientemente: `document[start:end] == text`. `contextrag.eval.citations.citation_problems` lo hace por
  respuesta.
- Una respuesta de tipo `fallback` es una abstención; `reason` dice por qué (sin evidencia, conflicto, ambigüedad, entidad no
  documentada, inyección retirada).
- Una respuesta `verified` solo existe si cada afirmación está respaldada por un pasaje que menciona la entidad preguntada.

### Evaluación con documentos sintéticos

El corpus de entidades (`eval/entity_corpus.py`, 46 preguntas, 20 documentos) cubre alias, homónimos, entidades inexistentes,
preguntas ambiguas y acceso restringido. `python -c "from contextrag.eval.entity_bench import run_entity_bench as r; print(r()['configs'])"` (no tiene línea de comandos propia)
reporta falsas respuestas, falsas abstenciones, respuestas incorrectas y abstenciones correctas por configuración, y separa
los casos conocidos fuera de la cifra principal.

### Mediciones registradas (pila sin conexión, corpus sintético)

Solo se citan resultados que se han ejecutado en este repositorio. Ver los ficheros JSON en `contextrag/eval/data/` y el
informe de la sesión; no son evidencia de calidad con modelos reales.

- **Entidades** (`alias_resolution`, configuración por defecto): falsas respuestas 0.0; falsas abstenciones 0.0; respuestas
  incorrectas 0.0; abstenciones correctas 1.0; aclaraciones correctas 1.0. Sin control de entidades, falsas respuestas 0.86.
  Sobre el conjunto principal; los 4 casos conocidos (`limit_*`) quedan fuera de esas cifras y se reportan aparte.
- **Recuperación y fundamentación**: `python -m contextrag bench --check-baseline` sale con código 0 en esta sesión (ninguna métrica bajo su suelo registrado). Las cifras de la tabla de la sección anterior no se han vuelto a imprimir; la comprobación es la garantía vigente.
- **Carga**: medida en esta máquina, en proceso y sin red; sirve para comparar ejecuciones, no para estimar usuarios.

### Lista de comprobación previa al piloto

| Estado | Elemento |
|---|---|
| Implementado y verificado con pruebas | Validación de subida, PDF cifrado/páginas/corrupto; casos y permisos en lectura, escritura, borrado, listado, auditoría y conversaciones; contenido activo y cuarentena de inyección; redacción de logs; cifrado y rotación de la clave con SQLite y Qdrant en memoria; caché resistente; contrapresión; alias, ambigüedad y conflicto estructurado; trazas y métricas por etapa. |
| Implementado y verificado solo con dobles o datos sintéticos | Conflictos semánticos (`nli`, con un modelo de prueba); evaluación RAGAS (orquestación con un juez simulado); verificación de citas por documento y página; carga en proceso. |
| Parcialmente verificado | `docker-compose` (valida la configuración y falla si faltan secretos; **no se ha levantado** porque no hay daemon Docker aquí); exportación OpenTelemetry (probada contra el SDK en memoria, no contra un colector real). |
| Pendiente, no verificado | Modelos reales (BGE, DeBERTa, Cohere, Anthropic, Ollama); puntuaciones reales del juez; calibración con documentos reales; detección de contradicciones semánticas con modelo real; Redis y Postgres reales en CI; escaneo de imágenes de contenedor; prueba de penetración; revisión de cumplimiento normativo (no se afirma ninguno); retención de logs y caché; gestión de claves de API con caducidad. |

**Riesgos que bloquean un piloto con datos reales** (ver el informe de la sesión para la clasificación): ausencia de
evaluación con modelos reales; conflictos detectados por heurística; inyección y redacción por listas de patrones; vectores y
metadatos en claro; clave maestra en variable de entorno; autenticación por fichero sin rotación; falta de prueba de
despliegue real; ausencia de revisión externa de seguridad.
