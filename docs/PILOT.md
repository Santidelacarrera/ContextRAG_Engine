# Documentación para un piloto controlado

**Advertencia.** Este documento describe lo que el repositorio hace y lo que se ha verificado. No es una declaración de
cumplimiento normativo, de certificación ni de confidencialidad garantizada. No debe usarse con documentos reales de
clientes hasta completar la lista de la última sección.

## 1. Arquitectura y flujo de datos

```
cliente ──HTTPS(proxy, responsabilidad del despliegue)──► API (FastAPI, usuario sin privilegios en contenedor)
   │  X-API-Key → inquilino + casos + rol (fichero de claves, solo hashes)
   ▼
ingesta:  validación (tamaño, firma, PDF cifrado/páginas) → análisis → fragmentos padre/hijo
          → embeddings (identidad del modelo en la huella) → Qdrant (texto cifrado) + BM25 en memoria
consulta: recuperación híbrida (filtro ACL antes de devolver) → reranking (con tiempo máximo)
          → cuarentena de inyección → alias/ambigüedad/entidades → conflictos entre fuentes
          → generación (solo con evidencia) → verificación por afirmación → control de contenido activo
          → auditoría (PostgreSQL con RLS, sin texto de evidencia) → respuesta con citas
caché:    Redis (opcional): valores cifrados, invalidación por época; si cae, se degrada sin perder invalidaciones
memoria:  SQLite (opcional): turnos y memorias cifrados por inquilino
```

Cada etapa emite un tramo de traza (`ingest`, `chunk`, `embed`, `index`, `retrieve`, `rerank`, `context`, `entities`,
`conflicts`, `generate`, `verify`, `verify.nli`, `citations`) y contadores por etapa en `/metrics`.

## 2. Contrato de la API (resumen)

| Método | Ruta | Permiso | Notas |
|---|---|---|---|
| POST | `/ingest` | ingest | Un documento por elemento; 413 si supera el tamaño, 422 otros rechazos con código. |
| POST | `/query` | query | Respuesta con `status` (`verified` / `fallback`), `answer`, `claims`, `sources`, `meta`. Puede devolver 503 con `Retry-After`. |
| POST | `/api/v1/retrieve` | query | Fragmentos recuperados; el borrador no está verificado (`draft_verified` siempre falso). |
| GET | `/documents` | list | Solo los documentos visibles para la clave. |
| DELETE | `/documents/{id}` | delete | 404 idéntico para «no existe» y «es de otro caso». |
| DELETE | `/tenant` | erase | Solo claves sin restricción de casos; exige `X-Confirm-Tenant`. |
| DELETE | `/conversations/{id}` | erase | Solo la conversación del propio usuario. |
| GET | `/audit` | audit | Un caso ve solo sus propias consultas. |
| GET | `/metrics` | metrics | Contadores; sin etiquetas de inquilino ni de documento. |
| GET | `/healthz` | público | Devuelve solo la clase del fallo, nunca su mensaje. |

## 3. Configuración de proveedores

Solo se usa el proveedor que se configure explícitamente. Ningún documento sale del proceso sin `LLM=anthropic|ollama`,
`RERANKER=cohere` o `EMBEDDER=bge` (los que requieren red o modelos). Parámetros de los adaptadores:

- Anthropic: `ANTHROPIC_API_KEY` (solo por entorno), timeout 60 s y 2 reintentos del SDK, entrada limitada a 200 000
  caracteres (rechazo antes de enviar), respuesta truncada (`max_tokens`) o vacía = error, nunca una respuesta parcial.
- Ollama: `OLLAMA_HOST`, timeout configurable, respuesta malformada o vacía = error.
- Errores: el usuario recibe una clase (p. ej. `RateLimited`), nunca el mensaje del proveedor, la URL ni el cuerpo.
- Embeddings: cada modelo tiene una identidad (`model_id`). Cambiar de modelo fuerza el reindexado de los documentos; los
  vectores de dos modelos no se comparan entre sí. **Efecto de actualización:** los índices anteriores se reindexan una
  vez al cambiar la huella. Un índice con documentos de dos modelos no se detecta automáticamente en consulta: tras cambiar
  de modelo, reindexe todos los documentos o borre el inquilino.

**Salida de red hacia Ollama (`OLLAMA_HOST`).** El cliente acepta solo `http(s)://host[:puerto][/prefijo]`, sin
credenciales, query ni fragmento. Resuelve el nombre una sola vez y conecta a la dirección comprobada (sin segunda
resolución, así que no hay ventana de DNS). Rechaza direcciones de enlace local, multicast, no especificadas y los
endpoints de metadatos conocidos (AWS, Azure, Alibaba, IPv6 de AWS). **Loopback y redes privadas están permitidas**: un
Ollama local o interno es el despliegue normal; si el piloto debe impedirlo, es una decisión de red del despliegue. No
usa proxies del entorno (`HTTP(S)_PROXY`), no sigue redirecciones y limita la respuesta a 2 MB. Probado contra un servidor
HTTP local real y con resolución simulada (`tests/test_ollama_transport.py`, 22 casos). **La prueba HTTPS positiva no se
ha ejecutado** (requiere certificado); solo se prueba la negativa previa al TLS.

**Revisión del modelo NLI (`NLI_MODEL_REVISION`): criterios, no un valor.** El valor por defecto de `NLI` es `lexical`
(sin modelo). Para usar `NLI=deberta`, fijar una revisión exige estos pasos, en este orden, y ninguno se ha cumplido todavía:
1. Confirmar el identificador exacto del modelo (`NLI_MODEL`), su organización y que es el previsto para verificación de
   entailment; comprobar en su ficha que tiene una clase de contradicción, que el código necesita.
2. Revisar la licencia del modelo y de sus pesos, y si es compatible con el uso previsto (ver `LICENSING.md`).
3. Elegir un commit de la rama principal del repositorio del modelo en el Hub (un hash de 40 caracteres), no una etiqueta
   móvil. Anotar fecha y quién lo eligió.
4. Cargar ese commit con `transformers` en un entorno aislado y registrar: versión de `transformers`, `torch`, dispositivo,
   tiempo de carga y que la salida de clases coincide con la ficha.
5. Guardar el hash en `NLI_MODEL_REVISION` del entorno de preproducción (no en el código), y ejecutar
   `pytest tests/test_nli_verifier.py` con los extras `models` instalados (hoy se omite).
6. Medir sobre un conjunto de desarrollo, nunca sobre el held-out de conflictos o de inyección, antes de fijar umbrales.

Sin estos pasos, no hay un hash verificable en el repositorio: cualquier valor que se escribiera aquí sería inventado.

## 4. Autorización por inquilino y caso

- La identidad sale de la clave, nunca de cabeceras del cliente. Un `X-Tenant` que no coincide con la clave se rechaza (403).
- Un **caso** es un grupo ACL. Una clave con rol `admin` y grupos opera solo sobre esos casos; sin grupos, opera sobre el
  inquilino. Un `reader` solo consulta.
- El filtro de permisos se aplica en la búsqueda vectorial, en BM25 y en la búsqueda de alias, antes de que un pasaje
  llegue al modelo. Hay pruebas que inspeccionan lo que recibe el modelo, no solo la respuesta final.
- La caché de respuestas se indexa por inquilino y grupos del llamante.
- Límites conocidos: las claves no caducan; no hay roles por documento; no hay integración con un proveedor de identidad
  (OIDC/SAML); la gestión del fichero de claves es manual.

## 5. Cifrado, claves y retención

| Dato | Cifrado en reposo | Notas |
|---|---|---|
| Texto de fragmentos (Qdrant) | Sí, con `MASTER_KEY` derivada por inquilino (Fernet) | Requiere `MASTER_KEY`. |
| Vectores | No | Necesarios para la búsqueda. Pueden reconstruir parcialmente el contenido. |
| Metadatos (doc_id, nombre de archivo, secciones, fechas, ACL) | No | Ver riesgos. |
| Turnos y memorias conversacionales (SQLite) | Sí, ligados a su posición | `MEMORY_PLAINTEXT=1` lo desactiva (solo desarrollo). |
| Caché Redis | Sí (valores), claves y contadores en claro | Las claves son hashes. |
| Auditoría (PostgreSQL) | No (lo que guarda) | Guarda pregunta, respuesta generada, referencias y **huella** del texto de evidencia, sin el fragmento. La respuesta puede repetir palabras del documento. Los registros anteriores a este cambio conservan el fragmento literal. |
| Documentos originales | Fuera del sistema | El sistema no almacena el archivo subido, solo su texto indexado. |
| Logs y trazas | No aplica cifrado | Sin contenido de documentos ni prompts; RUT, correos, teléfonos, números de causa y claves se redactan. |
| Copias de seguridad | Responsabilidad del despliegue | Cualquier copia de Postgres/Qdrant/SQLite contiene datos en el estado en que se tomó. |

**Clave maestra.** Generación: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Almacenamiento: gestor de
secretos, nunca en el repositorio ni en imágenes. Validación: al arrancar, menos de 16 bytes falla, menos de 32 avisa.
**Desde fichero (recomendado):** `MASTER_KEY_FILE` y `MASTER_KEY_PREVIOUS_FILE` apuntan a ficheros montados por el
orquestador (secreto de Docker o Kubernetes). No aparecen en `docker inspect` ni en el entorno del proceso. El arranque
**rechaza** un fichero legible por grupo u otros (permisos distintos de 0600 o 0400); el mensaje nombra la ruta, nunca el
contenido. Definir a la vez la variable y su versión `_FILE` es un error de arranque: no se adivina cuál vale. **Rotación
sin pérdida:** `MASTER_KEY` nueva + `MASTER_KEY_PREVIOUS` con la anterior; `python -m contextrag rotate-key --dry-run` y luego
`python -m contextrag rotate-key`. Cada registro se descifra, se re-cifra y se verifica antes de reemplazarse; un registro que
ninguna clave abre detiene el proceso sin modificar nada. Solo después retire la clave anterior. Verificado con SQLite y con
Qdrant en memoria, no con un servidor Qdrant ni con datos reales.

**Retención (configurable por despliegue).**
- Auditoría: `python -m contextrag retention --months N` (conserva N meses calendario); se ejecuta desde cron con la
  conexión de administrador. Verificado con PostgreSQL 16 real en las pruebas.
- Caché de respuestas: TTL `CONTEXTRAG_CACHE_TTL_SECONDS`; invalidación inmediata al cambiar un documento.
- Memoria conversacional: borrado por conversación (`DELETE /conversations/{id}`) o por inquilino.
- Logs y trazas: los produce la plataforma de logs del despliegue; el sistema no define su retención. **No implementado
  aquí.**

**Minimización de la auditoría.** Por defecto la auditoría guarda la pregunta y la respuesta completas (sin cambio). Con
`CONTEXTRAG_AUDIT_QUESTION=omit` y `CONTEXTRAG_AUDIT_ANSWER=omit` se guarda solo la longitud (`[omitted: N chars]`) y el
estado. Aplica a la auditoría de respuestas (PostgreSQL y en memoria) y a la de recuperación (`/retrieve`). Probado con
PostgreSQL real (`tests/test_retrieve_api.py`) y con la pila en memoria (`tests/test_audit_minimisation.py`). **Recomendación
para el piloto: `omit` salvo que se necesite el texto para investigar incidentes.** Retención: la tabla no tiene caducidad
propia; usar `python -m contextrag retention --months N`.

**Cifrado en reposo frente a cifrado durante el procesamiento.** Lo implementado es cifrado en reposo del texto de
fragmentos, memoria conversacional y caché. Los vectores y los metadatos siguen en claro porque la búsqueda los necesita en
claro. Ningún dato está cifrado *durante* el procesamiento: el proceso que busca y genera ve el contenido, como cualquier
sistema de búsqueda sin cifrado homomórfico.

**Estado actual por almacén (verificado en el código):**

| Almacén | Contenido en claro | Cifrado |
|---|---|---|
| Qdrant (hijos y padres) | vectores, `tenant`, `doc_id`, `source`, `acl`, identificadores | texto del fragmento (Fernet por inquilino) |
| SQLite (memoria) | `tenant`, `conversation_id`, roles, posiciones | contenido de cada turno y memoria |
| PostgreSQL (auditoría) | **pregunta, respuesta generada**, usuario, referencias, huellas | no cifrado |
| Redis (caché) | claves (hashes), contadores | valores (`EncryptedCache`) |
| BM25 en memoria | tokens del texto | no persistido; se reconstruye al arrancar |

Un hallazgo a decidir: **la pregunta y la respuesta de cada consulta se guardan en claro en PostgreSQL.** Son datos del
usuario que pueden ser confidenciales aunque el documento no lo sea.

**Opciones para vectores y metadatos (ninguna implementada; requieren decisión):**

| Opción | Protege | No protege | Coste |
|---|---|---|---|
| A. Cifrado de volumen y de copias (LUKS, KMS del proveedor) | discos robados, copias de seguridad | un proceso o operador con acceso al servicio en marcha | bajo; infraestructura |
| B. Metadatos cifrados a nivel de campo, con búsqueda por HMAC (`doc_id`, `tenant`, grupos de la ACL como etiquetas HMAC) | lectura directa de Qdrant o de sus ficheros | el filtro de ACL sigue revelando qué grupos existen si se conoce la clave de búsqueda | medio; cambia el esquema y requiere reindexar |
| C. Vectores cifrados con una transformación que preserve distancias (p. ej. DCPE) | vectores en reposo frente a quien no tenga la clave | las distancias aproximadas; la recuperación se degrada; no hay garantías formales sólidas en la literatura para todas las variantes | alto; calidad de recuperación no medida aquí |
| D. Modelo de embeddings propio por inquilino | reconstrucción de texto a partir de vectores ajenos | no evita el acceso al proceso | alto; operaciones y evaluación por inquilino |

Recomendación para el piloto: **A + controles de acceso** (roles de base de datos con RLS, clave de API de Qdrant, contraseña de Redis,
red privada). Tratar B como siguiente paso si el piloto usa documentos sensibles. No recomendar C sin medir la pérdida de
recuperación con documentos reales. Los vectores de embeddings pueden reconstruir parcialmente el texto; ese riesgo se
documenta, no se resuelve con las opciones anteriores.

**Gestión y rotación de claves.** La clave maestra se deriva por inquilino; la rotación con `MASTER_KEY_PREVIOUS` está
probada para SQLite y Qdrant en memoria. Perder la clave significa perder los datos cifrados: el despliegue debe guardar la
clave en un gestor de secretos con copia controlada. Las copias de seguridad contienen texto cifrado pero no la clave; la
clave debe respaldarse por separado. Los datos en claro (vectores, auditoría) quedan fuera de la rotación.

**Control de acceso.** Roles de PostgreSQL separados (propietario de esquema y aplicación con RLS; ver `deploy/`),
autenticación de Qdrant y de Redis activada en `docker-compose.yml` (sin verificar en un servicio real), y puertos
enlazados a 127.0.0.1 en el compose de ejemplo.

## 6. Procedimiento de eliminación de documentos

1. `DELETE /documents/{doc_id}` (o `python -m contextrag erase` para un inquilino completo).
2. Se eliminan fragmentos, vectores, entradas BM25 y la caché del inquilino queda invalidada.
3. Verificado en pruebas: el documento no vuelve a aparecer en búsquedas, respuestas ni caché. Actualizar un documento también: tras el cambio, la respuesta cacheada anterior no se sirve y el texto antiguo no aparece (`tests/test_update_propagation.py`).
4. **Conversaciones:** los turnos del asistente que citaron el documento se borran; la respuesta de la API informa
   `conversation_turns_removed`. Los turnos creados antes de guardar la procedencia (columna `cites`) no se pueden
   vincular: se cuentan como `unlinked_assistant_turns` en el resultado del borrado y pueden repetir contenido del
   documento hasta el borrado de la conversación. Las memorias resumidas se generan solo a partir de turnos del usuario.
   Verificado con SQLite en pruebas (`tests/test_deletion_coherence.py`, `tests/test_conversation_erasure.py`).
5. **Remanente conocido:** el texto de la respuesta generada (en la auditoría y en la caché hasta su TTL) puede repetir
   palabras del documento. Borrar el registro de auditoría de una consulta requiere `audit-purge` o el borrado del
   inquilino; un borrado de documento no lo hace. Un *legal hold* (`keep_audit=true`) conserva la auditoría a propósito.
   La auditoría guarda una huella del texto de evidencia, no el fragmento; los registros anteriores al cambio sí lo
   conservan.
6. Las copias de seguridad anteriores al borrado conservan el documento; su tratamiento es responsabilidad del despliegue.

## 7. Modelo de amenazas (resumen)

| Amenaza | Control | Estado verificado |
|---|---|---|
| Acceso a datos de otro caso o inquilino | Clave → caso; filtro antes de recuperar y de generar | Pruebas negativas por operación y por contenido del prompt. |
| Inyección desde un documento | Cuarentena por patrones; pasajes como datos; metadatos saneados | **Riesgo alto.** Conjunto de desarrollo: 13/17 detectados, 0/20 falsos positivos. **Conjunto held-out congelado: 3/12 detectados (25%), 0/12 falsos positivos.** No debe considerarse una protección fiable frente a inyección. |
| Conflicto no detectado entre fuentes | Revisión humana exigida en la respuesta; detección estructurada (fechas, cifras, negaciones) | Desarrollo: precisión 1.0, recall 0.73 sobre 22 pares. **Held-out congelado: precisión 1.0, recall 0.67 (4 de 6), 0 falsos positivos sobre 13 pares**; fallos: cifras en palabras fuera de vocabulario y una diferencia de cifra bajo el umbral de similitud. |
| Exfiltración por imagen o script en la respuesta | Rechazo de contenido activo | Pruebas con variantes; los enlaces normales se permiten. |
| Información filtrada en errores | Respuestas genéricas y clases, no mensajes | Pruebas con mensajes que contienen rutas y claves. |
| Clave maestra expuesta | Solo por entorno; redacción en logs; `repr` sin material | Pruebas. Gestión real de secretos: no verificada. |
| Caché obsoleta tras un cambio | Invalidación por época; pendiente si Redis cae | Pruebas con Redis real y caída. |
| Saturación | Límites de concurrencia y 503 | Medido en proceso; no es prueba de despliegue. |
| Dependencia vulnerable | `pip-audit` en CI | Sin avisos conocidos en la última auditoría local. |
| Compromiso del contenedor | Usuario sin privilegios; puertos en 127.0.0.1 | Configuración revisada; imagen no construida aquí. |
| Mezcla de vectores de dos modelos | Identidad del modelo en la huella; dimensión fijada en el índice; escrituras incompatibles rechazadas | Pruebas en el almacén en memoria y en Qdrant en memoria. Un índice ya mezclado antes de la corrección no se repara solo: reindexar. |

**No cubierto:** ataques de red sin TLS (el proxy es responsabilidad del despliegue), robo de credenciales de la
máquina, suplantación de usuarios legítimos, ingeniería social, y cualquier vulnerabilidad no conocida de las dependencias.

## 8. Respuesta ante incidentes (procedimiento mínimo)

1. **Contener:** revocar las claves afectadas en el fichero de claves y reiniciar la API; si la clave maestra pudo
   filtrarse, rotarla (sección 5) antes de reabrir el servicio.
2. **Determinar alcance:** consultar `/audit` (por caso o inquilino) y los contadores `contextrag_*` de `/metrics`.
   Los logs no contienen el contenido de documentos, por lo que el análisis se apoya en identificadores y referencias.
3. **Notificar:** el procedimiento de notificación legal y contractual no está definido en el repositorio; debe definirlo el
   responsable del piloto.
4. **Recuperar:** restaurar desde copia verificada anterior al incidente; reindexar si cambió el modelo de embeddings.
5. **Documentar:** registrar causa, alcance, fecha y medidas en un registro fuera del repositorio.

## 9. Evaluación RAG

Ver `README.md`, sección «Estado para un piloto jurídico», y `contextrag/eval/`:

| Evaluación | Comando | Qué mide | Dónde están los resultados |
|---|---|---|---|
| Recuperación y fundamentación | `python -m contextrag bench --check-baseline` | Recall@k, precision@k, MRR, nDCG@k; verificación, corrección, citas | `contextrag/eval/data/retrieval_bench_*.json` |
| Entidades y alias | `entity_bench.run_entity_bench()` | Falsas respuestas, falsas abstenciones, aclaraciones | `entity_bench_v1.json` |
| Conflictos entre fuentes | `conflict_bench.run_conflict_bench()` | Matriz de confusión por categoría | `conflict_bench_v1.json` |
| Inyección | `injection_bench.run_injection_bench()` | Detección y falsos positivos | — (sintético, en código) |
| Carga en proceso | `python -m contextrag loadtest` | Rendimiento y percentiles | Ver sección 10 |
| Juez RAGAS | `python -m contextrag ragas --cases ...` | Faithfulness, answer relevancy, context recall | **No ejecutado: requiere proveedor y clave.** |
| Proveedores (adaptadores) | `pytest tests/test_provider_adapters.py` | Timeouts, truncado, respuesta vacía o malformada, errores sin contenido | Con clientes simulados; no prueban ningún modelo. |

Todas las cifras son de la pila sin conexión (embeddings hash, reranker y NLI léxicos, generador extractivo) sobre
corpus sintéticos. No miden la calidad de modelos reales ni de documentos reales.

**Conjuntos held-out.** `contextrag/eval/heldout.py` contiene casos congelados (13 pares de conflicto, 12 ataques de
inyección, 12 textos legítimos parecidos). Regla: no se ajusta código ni umbrales con ellos. Resultados actuales:

| Conjunto | Desarrollo | Held-out |
|---|---|---|
| Conflictos entre fuentes | recall 0.73 (22 pares), precisión 1.0 | recall 0.67, precisión 1.0, 0 falsos positivos (13 pares) |
| Inyección (ataques detectados) | 13/17 | **3/12** |
| Inyección (falsos positivos sobre textos legítimos) | 0/20 | 0/12 |

**Análisis estático (bandit 1.9.x, `bandit -q -r contextrag -f txt`).** Resultado final: **0 hallazgos**, 0 advertencias.
- B608 (SQL con f-string, `audit.py` y `db.py`): eliminado. El SQL se compone con `psycopg.sql` (identificador de tabla y
  fragmento de cast) y los valores son parámetros. No se suprimió.
- B310 (`urlopen` en `OllamaLLM`): corregido. `OLLAMA_HOST` debe ser `http(s)://host[:puerto][/prefijo]`, sin credenciales,
  query ni fragmento; el cliente rechaza cualquier redirección (ver `contextrag/generation/llm.py`, `_NoRedirects`).
  Probado contra un servidor HTTP local real (`tests/test_ollama_transport.py`).
- B311 (`random.Random(seed)` en los generadores de corpus de evaluación, cuatro líneas): suprimido con justificación.
  Son datos sintéticos reproducibles, no material criptográfico.

**Configuración de modelos.** `NLI_MODEL_REVISION` fija el commit del modelo NLI (sin él, se advierte al arrancar).
`OLLAMA_HOST` se valida como URL http(s). Ninguna revisión real de modelo está fijada en este repositorio.

**Evaluación de inyección: alcance y plan.** No hay datos etiquetados reales. Los conjuntos de desarrollo (17 ataques, 20
casos legítimos) y held-out (12 + 12) son sintéticos y **no** se han ajustado contra el held-out en esta ronda. Medidas
actuales: detección 3/12 (held-out), 0/12 falsos positivos. Esto no es una medida de resistencia general a inyección. Plan
para un piloto con datos: (1) reunir de 200 a 500 fragmentos etiquetados de documentos reales, con permiso y anonimizados,
incluyendo casos legítimos que citen frases de instrucción; (2) separar un held-out congelado antes de ver cualquier
resultado; (3) medir recall, precisión y falsos positivos por tipo de ataque; (4) decidir si el resultado justifica un
clasificador. Mientras tanto, la protección que sí se aplica no depende de la detección: el texto recuperado va como
datos, el modelo no tiene herramientas ni acciones, el acceso se filtra antes de construir el contexto, y las respuestas
con contenido activo se rechazan.

**Dependencias de extras (pip-audit `-r`, extras declarados).** Resultado antes del ajuste: 6 vulnerabilidades en 3 paquetes,
todas por el extra `ragas`. Tras fijar `langchain-openai>=1.1.14` (corrige PYSEC-2026-76; resolución verificada con
`ragas==0.4.3`, sin ejecutar RAGAS), quedan:
- `ragas 0.4.3` (PYSEC-2026-3046): fallo en el procesamiento de ficheros locales; **sin versión corregida publicada**.
  Mitigación: no usar `contextrag ragas` con rutas o contenidos no confiables.
- `diskcache 5.6.3` (PYSEC-2026-2447): serialización con pickle; requiere escritura en el directorio de caché. **Sin
  versión corregida**. Lo arrastra el extra, y `contextrag` no lo usa directamente. Mitigación: no compartir el directorio.

**Alcance de `ragas` y `diskcache` (verificado en esta fase).** Ni la API, ni la fábrica, ni la CLI principal cargan
`ragas`, `diskcache` ni `langchain` (comprobado en tiempo de ejecución). Solo los usa `contextrag ragas`
(`contextrag/eval/ragas_eval.py`), que no se ejecuta con entrada de usuario. `diskcache` lo requiere únicamente `ragas`.
**No afirmamos que estén corregidas:** PYSEC-2026-3046 (`ragas` 0.4.3) y PYSEC-2026-2447 (`diskcache` 5.6.3) no tienen
versión corregida publicada que se haya verificado. Mitigaciones: no ejecutar `contextrag ragas` con casos o rutas no
confiables; no apuntar su caché a un directorio escribible por terceros; no instalar el extra `ragas` en la imagen de
producción si el piloto no lo necesita.

## 10. Mediciones registradas

Máquina de desarrollo: 4 CPU, 16 GB RAM, Python 3.13, sin red, sin Docker. Carga en proceso, 300 peticiones por nivel,
semilla 7, **caché de respuestas desactivada**:

| Concurrencia | Lectura (req/s, p50/p95 ms) | Lectura + reindexación 1 de cada 10 (req/s, p50/p95 ms) |
|---|---|---|
| 1 | 171 · 5.6 / 10.0 | 178 · 5.5 / 7.9 |
| 4 | 152 · 24.8 / 38.8 | 153 · 24.3 / 36.2 |
| 8 | 162 · 43.1 / 108.5 | 152 · 44.9 / 71.2 |

Lectura: el rendimiento no crece con la concurrencia (un solo proceso de Python: el intérprete serializa el cálculo). Con la
caché activada, las mismas lecturas llegan a unas 500 req/s, pero esa cifra mide aciertos de caché y no el procesamiento.
Memoria máxima del proceso: 53 MB. **Esto no es capacidad de producción.**

**Corrección de la versión anterior de este documento.** Antes se afirmaba que la reindexación bloqueaba a los lectores y
que era el cuello de botella (610 a 137 req/s). Esa medición tenía la caché activada: las lecturas eran aciertos de caché y la
reindexación la invalidaba. Medido después, con el embedder de ingesta lento (150 ms por lote), los lectores completan
199 consultas durante 1.05 s de reindexación (latencia máxima 25 ms): el embedding ocurre fuera del cerrojo y no bloquea. La
prueba queda en `tests/test_reindex_concurrency.py`.

## 11. Configuración recomendada para un piloto

```
CONTEXTRAG_REQUIRE_AUTH=true            # solo con claves por caso
CONTEXTRAG_KEYS_FILE=/run/keys.json     # montado de solo lectura
MASTER_KEY=<32+ bytes, de un gestor de secretos>
CONTEXTRAG_RESOLVE_ENTITY_ALIASES=true
CONTEXTRAG_QUARANTINE_SUSPECTED_INJECTION=true
CONTEXTRAG_CONFLICT_DETECTION=structured   # revisión humana siempre; no es semántico
CONTEXTRAG_AMBIGUITY_POLICY=clarify
CONTEXTRAG_LOG_REDACTION=on
CONTEXTRAG_TRACING=off                     # o otel con un colector propio
```

Usar solo documentos sintéticos o autorizados por escrito. TLS en el proxy. Sin copias de seguridad con datos reales
hasta definir su cifrado y su retención.

**Preproducción (ya en `deploy/docker-compose.yml`).** `CONTEXTRAG_AUDIT_QUESTION` y `CONTEXTRAG_AUDIT_ANSWER` valen
`omit` por defecto en el servicio de la API; el valor por defecto del código sigue siendo `full`. Para investigar un incidente
se sobrescribe por entorno y se documenta. Comprobado con `docker compose config` (sin daemon) y con
`tests/test_audit_minimisation.py`. La integración con DOCMIND_AI no existe todavía: ver `docs/INTEGRATION_DOCMIND.md`.

## 12. Lista de comprobación previa al piloto

| Estado | Elemento |
|---|---|
| Verificado | Suite: **800 pruebas pasando, 3 omitidas** (torch, transformers y RAGAS en vivo, no disponibles). `ruff check .` sin errores (`build/` excluido, ver abajo). `mypy contextrag --ignore-missing-imports` sin errores en 61 ficheros. `bandit -r contextrag` 0 hallazgos. pip-audit sobre dependencias base: sin vulnerabilidades conocidas; sobre extras: 2 vulnerabilidades sin corrección publicada (ver sección 9). |
| Pendiente de decisión | `build/` está versionado (31 ficheros), es salida de setuptools y no lo usa CI, el Dockerfile ni el empaquetado. Ruff lo excluye y `.gitignore` lo ignora desde ahora. **No se ha eliminado del repositorio**: eso es una decisión del propietario. |
| Verificado con infraestructura real local | Redis 7 (caché; caída con snapshot guardado y recuperación; una invalidación perdida durante la caída no devuelve la respuesta antigua); PostgreSQL 16 (auditoría con RLS, retención, roles). |
| Parcialmente verificado | `docker-compose`: la configuración se valida y falla sin secretos; **no se ha levantado** (no hay daemon). Qdrant servidor: no hay binario; las pruebas usan el modo en memoria. |
| No verificado | Modelos reales (BGE, DeBERTa, Cohere, Anthropic, Ollama); juez RAGAS; calibración con documentos reales; contradicciones semánticas; gestión de secretos del despliegue; TLS; integración con un proveedor de identidad. |
| Implementado en esta fase | Caducidad de claves de API (`keygen --expires-in-days`), recarga del fichero de claves al cambiar y cierre ante un fichero ilegible; secretos desde fichero con control de permisos; borrado de turnos de conversación que citaron un documento. |
| Pendiente | Política de retención de logs; notificación de incidentes; cifrado de vectores y metadatos (opciones en la sección 5, sin implementar); revisión de los turnos legados sin procedencia; reindexado automático de índices mezclados; defensa frente a inyección que no dependa de patrones (held-out 3/12); revisión externa de seguridad; decisión de licencia (ver `LICENSING.md`); |
