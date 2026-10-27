# Licencia y distribución: estado y opciones

**Estado actual: no hay licencia declarada.** El repositorio no contiene archivo `LICENSE`, `pyproject.toml` no declara
ninguna licencia y no hay avisos de copyright. Por defecto eso significa que todos los derechos están reservados: nadie
distinto del titular puede copiar, modificar o redistribuir el código. Esto no es una recomendación legal; es la
consecuencia de no declarar nada.

## Titularidad y autoría (lo que el historial muestra)

- 18 commits tienen como autor `Claude <noreply@anthropic.com>` y 5 tienen como autor la persona titular del repositorio.
- La titularidad de código generado con asistencia de IA no está resuelta de forma uniforme entre jurisdicciones. Antes de
  elegir una licencia pública conviene confirmar qué contribuciones se aceptan como propias y cuáles requieren revisión.
- No hay contribuciones de terceros registradas en el historial. Si se aceptan contribuciones externas, hace falta un
  acuerdo de contribución (CLA o DCO) antes de cambiar la licencia.

## Dependencias de ejecución (licencias leídas de los metadatos instalados)

| Paquete | Licencia | Implicación |
|---|---|---|
| rank-bm25, redis, pydantic, fastapi, prometheus-client | MIT / Apache-2.0 | Permisivas: sin obligación de copyleft. |
| numpy, pypdf, httpx, starlette, uvicorn | BSD-3-Clause | Permisivas: conservar el aviso de copyright al redistribuir. |
| anthropic, openai, qdrant-client, opentelemetry-api/sdk, ragas, cryptography | Apache-2.0 (cryptography: Apache-2.0 o BSD) | Permisivas: conservar avisos y el fichero NOTICE si existe. |
| psycopg, psycopg-pool | LGPL-3.0-only | Copyleft débil sobre la biblioteca. Se usa como dependencia sin modificarla; redistribuir una imagen que la contenga exige cumplir la LGPL (permitir su sustitución). |

No aparece ninguna dependencia AGPL ni GPL en el árbol de ejecución instalado. Los extras `models` (torch, transformers,
sentence-transformers) no se han instalado ni auditado aquí; su licencia debe revisarse antes de una distribución.

## Qué permite y qué no permite la situación actual

- Permite: que la persona titular use, modifique y ejecute el código en su propio entorno.
- No permite: que terceros lo copien, modifiquen, desplieguen o redistribuyan, incluso en un servicio interno de otra
  organización, salvo acuerdo expreso.
- No impide: que alguien aprenda de ideas abstractas (arquitecturas, técnicas de recuperación, métodos de evaluación) ni
  que reimplemente el comportamiento sin copiar el código. La protección por derecho de autor cubre la expresión, no las ideas.

## Opciones y recomendación

1. **Privado y propietario (recomendado por ahora).** Adecuado mientras no se hayan resuelto la titularidad de las
   contribuciones asistidas por IA, las dependencias LGPL en imágenes, y la estrategia comercial. Un piloto con un cliente
   puede hacerse bajo un acuerdo de licencia o de prueba escrito, sin publicar el código.
2. **Abierto con licencia permisiva (Apache-2.0).** Compatible con todas las dependencias actuales y con uso comercial.
   Incluye concesión de patentes, que suele preferirse en software empresarial. Requiere decidir antes qué parte del código
   se publica, porque una vez publicado no se retira.
3. **Abierto con copyleft fuerte (AGPL-3.0).** Obliga a publicar las modificaciones de quien ofrezca el sistema como
   servicio en red. Protege frente a la explotación cerrada, pero dificulta la venta de licencias comerciales a empresas
   que no quieren esa obligación, y debe usarse con licencia comercial dual si se quiere ese modelo.

**Recomendación razonada:** mantener el repositorio privado hasta decidir la estrategia comercial, y entonces elegir
entre la opción 2 (si el modelo es servicio, soporte o software integrable) o la opción 3 con licencia comercial dual (si
el modelo es venta de licencias). No se ha cambiado ninguna licencia automáticamente. Las decisiones pendientes son:
titularidad de las contribuciones asistidas, estrategia comercial y política de contribuciones externas.
