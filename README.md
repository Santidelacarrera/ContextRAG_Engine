# ContextRAG Engine

Pipeline RAG orientado a eliminar alucinaciones: búsqueda híbrida (densa + BM25) → reranking →
expansión Parent-Child → generación con citas → **verificación NLI afirmación por afirmación**.
Si alguna afirmación no queda respaldada con ≥ 0.95, se reintenta con feedback y, si sigue sin
verificarse, se devuelve una respuesta de contingencia determinista (sin texto generado).

```
ingesta ─► chunks padre/hijo ─► store (Qdrant | memoria)
pregunta ─► dense + sparse ─► RRF ─► reranker ─► padres ─► LLM ─► verificador NLI ─┬► VERIFIED (con coordenadas)
                                                              ▲                   └► reintento (feedback) ─► FALLBACK
                                                              └──────── auditoría (PostgreSQL / pgvector)
```

## Módulos

| Módulo | Rol |
|---|---|
| `chunking.py` | Padres alineados a párrafos, hijos con solapamiento; offsets absolutos en el documento |
| `store.py` / `qdrant_store.py` | `InMemoryStore` (cosine + BM25) y `QdrantStore` (vector denso + sparse IDF); filtro de tenant/ACL **dentro** de la consulta |
| `retrieval.py`, `rerank.py` | Fusión RRF, rerank (léxico, CrossEncoder, Cohere), dedupe por padre |
| `llm.py` | Claude (`AnthropicLLM`), Ollama, y `ExtractiveLLM` offline. Contrato: frases con `[n]` o `NO_ANSWER` |
| `nli.py`, `verifier.py` | `DebertaNLI` (DeBERTa-v3) y `LexicalEntailment` offline; ventanas de 1–3 frases; confianza = mínimo por afirmación |
| `security.py` | Cifrado Fernet del texto en reposo con clave derivada por tenant; ACL por grupos |
| `audit.py`, `sql/schema.sql` | Auditoría de cada consulta (claims, evidencias, fuentes, embedding en pgvector) |
| `api.py` | FastAPI: `POST /ingest`, `POST /ask`, `GET /audit` (cabecera `X-Tenant`) |

## Uso

```python
from contextrag import factory
rag = factory.offline()            # sin modelos ni servicios
rag.ingestor.ingest_text(open("politica.md").read(), tenant="acme", doc_id="politica.md")
a = rag.ask("¿Cuánto se conservan los registros de auditoría?", tenant="acme")
print(a.status, a.confidence, a.text, a.claims[0].evidence)
```

Producción: `factory.production(qdrant_url=..., postgres_dsn=..., master_key=...)`
(extras: `pip install -e ".[api,qdrant,models,anthropic,postgres,security,cohere,pdf]"`).
Servidor: `uvicorn --factory contextrag.api:create_app`.

## Tests

```
pip install -e ".[dev]" && pytest
```

## Límites actuales (honestos)

- Los componentes *offline* (`HashingEmbedder`, `LexicalReranker`, `LexicalEntailment`, `ExtractiveLLM`) son
  sustitutos deterministas para tests; la calidad real requiere bge + reranker + DeBERTa + LLM.
  Las rutas de modelos reales (`SentenceTransformerEmbedder`, `DebertaNLI`, `CrossEncoder/Cohere`,
  `AnthropicLLM`, `OllamaLLM`, `PostgresAudit`, Qdrant en servidor) **no están probadas** en este entorno.
- "Cero alucinaciones" no es una garantía absoluta: depende de la calidad del modelo NLI y del umbral;
  un NLI puede fallar en contradicciones sutiles. El diseño lo acota (falla cerrado), no lo elimina.
- Los vectores no se cifran (deben ser buscables); sólo el texto del payload. La autenticación no está
  incluida: `X-Tenant` debe ser fijado por un gateway de confianza.
- Sin soporte multimodal real aún (PDF → texto vía `pypdf`; sin OCR/imágenes).
