# Operación: claves, retención, incidentes y copias

Complementa `docs/PILOT.md`. Describe procedimientos; las decisiones de plazo son del responsable del tratamiento de datos.

## Claves de API
* Alta: `python -m contextrag keygen --id <etiqueta> --tenant <t> --role reader|admin --expires-in-days 90 --add-to keys.json` (la clave se imprime una vez).
* Inventario: `python -m contextrag keys list --file keys.json` (id, tenant, rol, grupos, caducidad; nunca claves ni digests).
* Baja inmediata: `python -m contextrag keys revoke --file keys.json --id <etiqueta>`. El servidor recarga el fichero al cambiar.
* Rotación: crear la nueva, entregarla, revocar la anterior. Recomendado: caducidad ≤ 90 días y revisión trimestral del inventario.

## Retención
| Dato | Mecanismo | Valor sugerido (a confirmar) |
|---|---|---|
| Auditoría PostgreSQL | `python -m contextrag retention` (cron mensual, DSN de administrador) | 12 meses |
| Texto de preguntas/respuestas en auditoría | `CONTEXTRAG_AUDIT_QUESTION/ANSWER=omit` (por defecto en el compose) | no guardar |
| Caché Redis | invalidación por época y TTL propios de la caché | sin datos persistentes sin necesidad |
| Logs de aplicación | del orquestador (Docker `json-file` con `max-size`/`max-file`, o el agregador) | 30 días; sin contenido de documentos |
| Conversaciones | `DELETE` por conversación / documento citado; `erase` por tenant | según contrato |

## Incidentes (sospecha de fuga o uso indebido)
1. Contener: `keys revoke` de las claves implicadas; si hay duda sobre el tenant, detener la API.
2. Preservar: exportar la auditoría del tenant (`GET /audit`) y los logs del periodo antes de cualquier purga o `retention`.
3. Alcance: filtrar la auditoría por id de clave y ventana temporal; los registros incluyen `request_id`.
4. Si hubo datos personales: notificar al responsable del tratamiento de inmediato para que valore la notificación a la autoridad y a los afectados en los plazos de la normativa aplicable (no se afirma aquí ningún cumplimiento).
5. Si se sospecha de `MASTER_KEY`: `python -m contextrag rotate-key` con la nueva clave en `MASTER_KEY` y la anterior en `MASTER_KEY_PREVIOUS`.
6. Registrar causa, impacto y medidas; revisar `docs/PILOT.md` sección 7.

## Copias de seguridad y restauración
* Qdrant: snapshots de la colección (`/snapshots`), almacenados cifrados; probar la restauración en un entorno aparte.
* PostgreSQL: `pg_dump` o WAL archivado, cifrado en reposo; la auditoría está particionada por mes.
* Tras restaurar Qdrant el índice BM25 se reconstruye al arrancar. Los datos borrados por `erase` reaparecerían al restaurar una copia anterior: reaplicar los borrados registrados.
* No hay cifrado de volumen ni de copias en este repositorio: es responsabilidad del despliegue.

## Antes de producción (fuera del repositorio)
TLS en el borde, gestor de secretos, versión fija de imágenes por digest, revisión externa de seguridad, prueba de penetración y prueba de restauración.
