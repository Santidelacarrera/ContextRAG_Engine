# Contrato de integración: DOCMIND_AI → ContextRAG_Engine

**Estado: no integrado.** Se ha buscado el adaptador en ambos repositorios (`docmind_ai` en HEAD `1b21fa5` y
`ContextRAG_Engine` en HEAD `a8c6191`): no existe ninguna referencia a ContextRAG en DOCMIND_AI, ni código, ni configuración,
ni documentación. DOCMIND_AI es hoy una API de extracción de datos de PDF (OCR, esquemas, revisión humana, exportación).
La variable `DOCMIND_INTEGRATION` de DOCMIND_AI activa solo sus pruebas Docker contra su propio stack.

Este documento describe lo que ContextRAG expone hoy y lo que DOCMIND necesitaría. Las propuestas de DOCMIND no están
implementadas y no se han probado contra un servidor.

## 1. Lo que ContextRAG expone (verificado en `contextrag/api/server.py`, `contextrag/api/retrieval.py`)

Autenticación: cabecera `X-API-Key`. La clave está ligada a un **tenant**, un conjunto de **grupos** (ACL) y un **rol**
(`reader`, `member`, `admin`…). `X-Tenant` solo puede confirmar el tenant de la clave; nunca lo sustituye.

| Operación | Método y ruta | Cuerpo principal | Respuesta relevante |
|---|---|---|---|
| Ingesta / actualización | `POST /ingest` | `{"documents": [{doc_id, text \| content_base64, source?, kind?, acl?, force?}]}`, máx. 50 documentos | `{tenant, documents: [{…, unchanged}]}` |
| Consulta con fuentes | `POST /query` | `{question (1–4000), groups?, top_k? (1–20), conversation_id?}` | `{status, answer, confidence, claims[], sources[], cached, …}` |
| Recuperación sin generación | `POST /api/v1/retrieve` | `{tenant_id, query, top_k?}` (requiere PostgreSQL configurado) | fragmentos con `doc_id`, `score` |
| Eliminar documento | `DELETE /documents/{doc_id}` | — | `{tenant, deleted, conversation_turns_removed}` |
| Listar documentos | `GET /documents` | — | `{tenant, documents[]}` (vista admin; un reader ve lo que su clave puede leer) |
| Borrar conversación | `DELETE /conversations/{id}` | — | `{tenant, deleted}` |
| Borrar tenant (irreversible) | `DELETE /tenant` + `X-Confirm-Tenant: <tenant>` | — | conteos; `?keep_audit=true` conserva la auditoría |
| Salud / métricas / auditoría | `GET /healthz`, `GET /metrics`, `GET /audit?limit=` | — | `/audit` solo del tenant de la clave; con `CONTEXTRAG_AUDIT_*=omit` no contiene texto de preguntas, respuestas ni afirmaciones |

**Actualización.** No hay `PATCH`. Una actualización es un nuevo `POST /ingest` con el mismo `doc_id`:
- Contenido distinto: se reindexa; la caché del tenant queda invalidada (época del tenant).
- Mismo contenido y misma ACL: `unchanged: true`, sin trabajo.
- Mismo contenido con ACL distinta: se reindexa (no cuenta como "sin cambios").
- `force: true` reindexa aunque no haya cambios.

**Errores.** `401` clave ausente o no válida; `403` rol insuficiente, tenant o grupos no concedidos; `413`/`422` documento
rechazado (código estable en el mensaje); `429` límite de tasa (`Retry-After`); `503` servidor ocupado o dependencia
caída (`Retry-After`); `500` con `{"error":"internal_error","request_id":…}` sin detalles.

**Límites.** Texto ≤ 5 000 000 caracteres; base64 ≤ 30 000 000 caracteres (~22 MB); 50 documentos por petición;
`doc_id` ≤ 512; `acl` ≤ 64 entradas.

## 2. Lo que DOCMIND necesitaría (propuesta, no implementada)

**Mapeo de identificadores.**

| DOCMIND | ContextRAG | Notas |
|---|---|---|
| `organization_id` (UUID) | tenant | Los UUID cumplen `^[A-Za-z0-9._:-]{1,128}$`. |
| `document_id` (UUID) | `doc_id` | Único por tenant. |
| `project_id` | entrada de `acl` (p. ej. `project:<uuid>`) | Un documento sin ACL es visible a todo el tenant: no usar vacío para documentos de proyectos. |
| Usuario (`user_id`) | clave de API ligada a sus grupos | Ver punto de decisión A. |
| Nombre de archivo | `source` | Se usa como etiqueta visible en las citas. |

**Flujo propuesto.**
1. Al completarse la extracción de un documento, DOCMIND llama a `POST /ingest` con el texto OCR por página (`kind: text`).
2. Al revisar o corregir un campo, no hay reindexado: la corrección no cambia el texto fuente. Solo se reindexa si cambia el texto.
3. Al eliminar un documento en DOCMIND, llamar a `DELETE /documents/{document_id}`. Si la llamada falla, reintentar
   (es idempotente) y no marcar el documento como eliminado en el lado de ContextRAG hasta confirmar `200`.
4. Las consultas de los usuarios pasan por `POST /query` con la clave de ese usuario; DOCMIND no reenvía la clave de
   usuario al navegador.

**Configuración de DOCMIND (de su gestor de secretos, nunca en el repositorio).** `CONTEXTRAG_BASE_URL`,
`CONTEXTRAG_API_KEY` (o una clave por usuario, ver A), y un interruptor `CONTEXTRAG_ENABLED=false` por defecto.

## 3. Decisiones necesarias antes de implementar

- **A. Autorización por usuario.** Los grupos pertenecen a la clave, y una petición solo puede restringirlos. Una clave de
  servicio con todos los grupos de todos los usuarios expondría los documentos de un usuario a otro. Opciones:
  (1) una clave por usuario de DOCMIND, gestionada en el fichero de claves (`CONTEXTRAG_KEYS_FILE`); (2) una clave por
  (organización, grupo) con un proxy de DOCMIND que nunca reenvíe la clave al cliente. La opción 1 es la única que
  aplica el límite dentro de ContextRAG. Coste: rotación y altas/bajas de claves en sincronía con los miembros de DOCMIND.
- **B. Atribución de auditoría.** La auditoría registra `c.id` (identificador de la clave), no el usuario de DOCMIND. Con
  claves compartidas, la auditoría no dice quién preguntó. Con claves por usuario, sí.
- **C. Texto que sale del perímetro.** Con `LLM=anthropic` o `ollama` remoto, el texto de los documentos y las preguntas
  se envían a ese proveedor. DOCMIND_AI mantiene `LLM_PROVIDER=mock` hasta aprobar un proveedor. ContextRAG por defecto
  usa `LLM=extractive`, sin envío externo. Esta decisión es de los propietarios de los datos.
- **D. Fuentes de texto.** Ingerir el OCR (texto) o el PDF (`content_base64`). El OCR evita cargar el PDF otra vez; el PDF
  da el parser de ContextRAG y su misma ruta de validación.
- **E. Retención.** La auditoría, la memoria de conversación y la caché tienen su propia retención. DOCMIND tiene la suya
  (`docs/DATA_RETENTION.md`). Deben alinearse antes de un piloto con datos reales.

## 4. Lo que falta para probar la integración

- Un despliegue de ContextRAG con PostgreSQL, Redis, Qdrant y `MASTER_KEY` reales (ver `deploy/docker-compose.yml`; no levantado).
- Un cliente de DOCMIND (no existe) y su propia prueba de integración contra ese despliegue.
- La decisión A.
- Datos sintéticos de DOCMIND (`backend/eval_corpus` contiene PDFs de prueba, que no deben usarse aquí sin revisar su licencia y su contenido).


## 5. Flujo de identidad (verificable)

1. DOCMIND autentica a su usuario con su propio mecanismo (sesión, JWT o clave de API de DOCMIND). ContextRAG no conoce
   ese mecanismo.
2. DOCMIND elige la **clave de ContextRAG** que corresponde a ese usuario o a ese rol, leyéndola de su gestor de secretos.
   La elección la hace el servidor de DOCMIND a partir de la identidad ya autenticada, nunca a partir de un campo que
   envíe el navegador.
3. Cada llamada a ContextRAG lleva solo `X-API-Key`. ContextRAG resuelve de la clave: **tenant**, **grupos** (ACL), **rol**
   e **identificador de auditoría** (`id` de la entrada del fichero de claves). Estos valores vienen del fichero de claves
   gestionado por el operador (`CONTEXTRAG_KEYS_FILE`), que se recarga al cambiar y rechaza una clave caducada.
4. Lo que el cliente pueda enviar en el cuerpo no amplía nada: `groups` solo restringe; `X-User` solo se usa en modo sin
   claves, y el identificador de la clave tiene prioridad en la auditoría.
5. La revocación es efectiva en la siguiente llamada: quitar la clave del fichero hace que devuelva `401`
   (verificado por HTTP en `tests/test_docmind_contract.py`).

**Lo que ContextRAG no puede verificar** y DOCMIND debe garantizar: que la clave entregada corresponde al usuario que
realiza la petición. Si el servidor de DOCMIND elige mal la clave, ContextRAG lo aplica sin saberlo. Esto es el límite
de confianza del contrato.

**Pendiente crítico:** el fichero de claves se gestiona a mano. Una clave por usuario de DOCMIND exige un proceso de alta y
baja sincronizado con los miembros de DOCMIND (decisión A, sección 3).

## 6. Cliente de referencia y pruebas contractuales (implementados aquí)

- `contextrag/client.py`: `ContextRAGClient` con `health`, `ingest`, `query`, `delete_document`, `list_documents`, `audit`.
  Biblioteca estándar, sin redirecciones, con tiempo límite, proxies desactivados, errores con el código estable del
  servidor y sin la clave en ningún mensaje. **Es una referencia**: DOCMIND debería copiarla o reimplementarla; no es
  código de DOCMIND_AI.
- `tests/test_docmind_contract.py` (23 casos): arranca el servidor real con uvicorn en un puerto local y ejercita el
  cliente por HTTP. Cubre éxito (ingesta, consulta con fuentes, actualización, eliminación, eliminación con `/` en el
  identificador), error (401 clave incorrecta sin eco de la clave, 403 de grupos y de rol, transporte sin servidor,
  límites del lado del cliente, URLs fuera de contrato, redirección no seguida), aislamiento entre organizaciones,
  revocación de una clave en el fichero, idempotencia del borrado, auditoría por tenant y auditoría sin texto con `omit`.
- `tests/test_e2e_integration.py` (7 casos): mismas garantías a nivel de la aplicación con claves, sin HTTP real.

**Qué no cubren estas pruebas:** DOCMIND_AI, PostgreSQL, Qdrant, Redis en el despliegue, proveedores de modelos y
certificados TLS. Son pruebas de contrato entre el cliente de referencia y el servidor de ContextRAG, no pruebas de
integración entre dos sistemas.

## 7. Estado del adaptador

| Parte | Estado |
|---|---|
| Adaptador dentro de DOCMIND_AI | **No existe.** No hay código, rama ni referencia a ContextRAG en ningún repositorio disponible (`main`, `qa/main-review`, `chore/untrack-coverage`, `feat/schemas-confidence-collab-observability`, historial completo). |
| Cliente de referencia en ContextRAG | Implementado y probado por HTTP contra el servidor real. |
| Integración de extremo a extremo entre ambos sistemas | **No ejecutada.** Bloqueada por el punto anterior. |

## 8. Requisitos para la prueba de extremo a extremo

- Del repositorio DOCMIND_AI: el módulo cliente, la configuración (`CONTEXTRAG_BASE_URL`, clave desde gestor de secretos),
  el mapeo de identidad (sección 5), los disparadores de ingesta y de borrado, y una prueba que arranque el stack
  de DOCMIND con ContextRAG apuntando a un servidor real.
- Del entorno: Docker (no hay daemon en este entorno), PostgreSQL y Qdrant del compose, `MASTER_KEY`, `CONTEXTRAG_KEYS_FILE`
  con claves de prueba generadas para la ocasión, y un Redis si se usa caché compartida.
- Pasos: ver `docs/PILOT.md`, sección 12, y los comandos de este documento.

## 9. Criterios objetivos para declarar la integración preparada

1. El adaptador existe en DOCMIND_AI y su código está revisado.
2. Las pruebas de contrato de este repositorio pasan contra el servidor desplegado con PostgreSQL, Qdrant y Redis reales.
3. La prueba de extremo a extremo de DOCMIND cubre ingesta, consulta con fuentes, aislamiento entre organizaciones y
   usuarios, actualización, eliminación, revocación y auditoría, todas contra el despliegue real.
4. La decisión A (autorización por usuario) está tomada y aplicada.
5. La auditoría se verifica en el despliegue con `CONTEXTRAG_AUDIT_*=omit` y no contiene texto.
6. Los riesgos de la sección 7 de `docs/PILOT.md` tienen una decisión registrada.

Hasta cumplir 1–3, **la integración no está terminada**.


## 10. Adaptador de referencia y punto de identidad (actualización de esta fase)

- **`GET /me`** (nuevo): con una clave válida devuelve `key_id`, `tenant`, `role` y `groups` de esa clave. Sin clave, `401`.
  Sirve para que el consumidor compruebe, antes de actuar, que la clave pertenece a la organización esperada.
- **`contextrag/integrations/docmind.py`** (nuevo, referencia): `DocmindRagAdapter` con `ingest`, `query`, `delete_document`,
  `list_documents` y `audit`. Recibe un `VerifiedPrincipal(organization_id, user_id)` que DOCMIND debe obtener de su propia
  autenticación, y un `CredentialDirectory` de DOCMIND que devuelve `Credential(key, key_id)`.
  Antes de cada operación llama a `/me` y falla cerrado si el tenant no coincide con `organization_id` (`IdentityMismatch`)
  o si el `key_id` no es el registrado para ese usuario. Si no hay credencial activa, `AccessDenied` sin ninguna petición.
- **Pruebas:** `tests/test_docmind_adapter.py` (17 casos) y `tests/test_docmind_contract.py` (23 casos), contra el servidor real
  por HTTP con un fichero de claves cargado como en producción. Cubren autenticación (sin cabecera, clave incorrecta,
  caducada, revocada), revocación de permisos por cambio de grupos, aislamiento entre organizaciones y entre usuarios de la
  misma organización, la atribución de auditoría (la cabecera `X-User` no la altera; `X-Tenant` de otra organización da 403)
  y la idempotencia del borrado.

## 11. Mapeo de roles y consecuencias para DOCMIND

ContextRAG no tiene un rol de "miembro" que suba documentos. Los roles reales son `reader` (solo consulta), `admin` (consulta,
ingesta, borrado, auditoría y listado) y `ops` (métricas). Un `admin` con grupos está **restringido a esos casos**: solo
ingesta y borra dentro de ellos.

| Rol en DOCMIND (propuesta) | Clave en ContextRAG | Efecto verificado |
|---|---|---|
| Visor / lector de un caso | `reader` con los grupos del caso | Consulta; sin texto de documentos de otros casos |
| Miembro que sube y revisa | `admin` con los grupos del caso (restringido) | Ingesta y borrado solo dentro de sus casos |
| Administrador de la organización | `admin` sin grupos (tenant completo) | Borrado idempotente; auditoría y listado de todo el tenant |

**Borrado y reintentos.** Con una clave de tenant, borrar un documento ausente da `200` (idempotente). Con una clave
restringida, un documento ausente o de otro caso da `404`, a propósito, para no revelar qué existe. Un reintento de DOCMIND
debe leer ese `404` como "ya eliminado" solo cuando su propia base de datos ya marca el documento como borrado.

## 12. Almacenamiento: qué se guarda en claro y qué no (verificado en el código)

| Almacén | Contenido | Estado |
|---|---|---|
| Qdrant, fragmentos hijo | vector, `tenant`, `doc_id`, `acl`, `public`, `start`, `end`, `pid`, `cid` | Vectores y metadatos en claro. El texto (`text`) va cifrado solo si hay `MASTER_KEY`. |
| Qdrant, fragmentos padre | además: `source` (nombre de archivo), `section`, `page`, `page_breaks`, `doc_hash` | En claro. El nombre de archivo es un metadato visible para quien lea Qdrant. |
| PostgreSQL, auditoría | `tenant`, `user` (id de la clave), `sources` (incluye `source`), huellas de las afirmaciones | Con `omit`, no guarda pregunta, respuesta ni texto de afirmaciones. Sí guarda nombres de archivo en las fuentes. |
| PostgreSQL, `retrieval_audit` | `query` (con `omit`, solo longitud), `doc_ids` | `doc_ids` en claro. |
| SQLite, memoria conversacional | contenido de turnos y memorias | Cifrado por inquilino si hay `MASTER_KEY`. |
| Redis, caché | valores de respuesta | Cifrado si hay `MASTER_KEY`; las claves son hashes. |
| Memoria del proceso (BM25) | tokens del texto | En claro mientras el proceso vive; no se persiste. |

**Lo que este cifrado NO es.** Cifrar el texto de los fragmentos no protege vectores ni metadatos. Un lector de Qdrant
ve los vectores, los identificadores de documento y los nombres de archivo. No debe presentarse como cifrado completo.

**Medidas necesarias para un piloto (no implementadas aquí; son de infraestructura y de configuración de DOCMIND):**
1. Cifrado de volumen para Qdrant, PostgreSQL, Redis y las copias de seguridad, con claves en gestor de secretos.
2. Qdrant con `QDRANT__SERVICE__API_KEY` y sin puerto publicado fuera de la red interna (el compose de ContextRAG ya enlaza a 127.0.0.1).
3. Roles de PostgreSQL separados (propietario y aplicación con RLS), como en el compose.
4. DOCMIND debe enviar como `source` un identificador opaco o una etiqueta sanitizada, no el nombre real del archivo,
   si el nombre es sensible. Esto afecta a las citas que ven los usuarios: decisión de producto.
5. Política de retención de la auditoría (`python -m contextrag retention --months N`) y de copias de seguridad.
6. Decisión registrada sobre cifrado de vectores y metadatos (opciones A–D en `docs/PILOT.md`, sección 5). Hasta entonces,
   el piloto se limita a documentos sintéticos o no sensibles.

## 13. Dependencias fuera del camino crítico

`ragas` y `diskcache` no se cargan al importar el adaptador, el cliente, el servidor o la fábrica (verificado en tiempo de
ejecución). Solo los usa `contextrag ragas`. Sus vulnerabilidades (PYSEC-2026-3046, PYSEC-2026-2447) **no tienen versión
corregida verificada**; no se declaran resueltas. Ver `docs/PILOT.md`, sección 9.
