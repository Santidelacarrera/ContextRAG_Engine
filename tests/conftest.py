import pytest

from contextrag import factory

DOC = """# Retención de datos

La política de retención establece que los registros de auditoría se conservan durante 7 años. Los registros de acceso se eliminan a los 90 días.

# Cifrado

Todos los datos en reposo se cifran con AES-256. Las claves se rotan cada 12 meses.
"""


@pytest.fixture
def rag():
    r = factory.offline()
    r.ingestor.ingest_text(DOC, tenant="acme", doc_id="policy.md")
    return r
