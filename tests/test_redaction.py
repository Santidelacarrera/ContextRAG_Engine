"""Log redaction: what it removes, what it must leave alone, and how far it can be trusted."""
import logging

import pytest

from contextrag.core.redact import install_log_redaction, redact


@pytest.mark.parametrize("text, expected", [
    ("cliente 12.345.678-9 solicitó", "cliente [RUT] solicitó"),
    ("rut 7654321-K y 12345678-5", "rut [RUT] y [RUT]"),
    ("escriba a maria.perez+caso@estudio-legal.cl hoy", "escriba a [EMAIL] hoy"),
    ("llame al +56 9 8765 4321 o al 987654321", "llame al [PHONE] o al [PHONE]"),
    ("causa Rol 1234-2020 y C-5678-2021", "causa [CASE] y [CASE]"),
    ("RIT O-123-2022 ante el juzgado", "[CASE] ante el juzgado"),  # the keyword is part of the identifier
    ("key sk-ant-api03-abcdefghijklmnop1234 usada", "key [SECRET] usada"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def", "Authorization: Bearer [REDACTED]"),
    ("crag_" + "A" * 30 + " expired", "[SECRET] expired"),
    ("connect postgresql://admin:S3cret!pw@10.0.0.5:5432/legal failed", "connect postgresql://admin:[REDACTED]@10.0.0.5:5432/legal failed"),
])
def test_known_identifiers_and_secrets_are_redacted(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize("text", [
    "La audiencia es el 5 de mayo de 2026.", "El monto es de 120.000.000 pesos.", "artículo 1.234 del código",
    "05-05-2026 fue lunes", "tiene 12 páginas y 3 anexos", "version 0.4.3 de la librería", "status=verified attempts=2 ms=14",
    "tenant=acme user=ana", "https://example.com/path?x=1", "plazo de 2555 días",
])
def test_ordinary_text_numbers_and_dates_are_left_alone(text):
    assert redact(text) == text


def test_names_and_addresses_are_not_detected_and_this_is_a_stated_limit():
    """Documents the limit rather than hiding it: pattern matching cannot find a person's name or a street address."""
    text = "Pedro Soto Rojas vive en Avenida Providencia 1234, Santiago"
    assert redact(text) == text


def test_the_filter_redacts_records_from_any_logger_including_formatted_arguments(caplog):
    assert install_log_redaction() is True
    assert install_log_redaction() is True  # idempotent: no stacking of factories
    with caplog.at_level(logging.INFO):
        logging.getLogger("some.third.party").error("db error for rut %s: %s", "12.345.678-9",
                                                    "postgresql://u:pw123456@db/x unreachable")
        logging.getLogger("contextrag.test").warning("contact %s", "ana@estudio.cl")
    assert "12.345.678-9" not in caplog.text and "pw123456" not in caplog.text and "ana@estudio.cl" not in caplog.text
    assert "[RUT]" in caplog.text and "[REDACTED]" in caplog.text and "[EMAIL]" in caplog.text


def test_a_record_without_sensitive_content_is_untouched(caplog):
    install_log_redaction()
    with caplog.at_level(logging.INFO):
        logging.getLogger("contextrag.test").info("query tenant=%s status=%s", "acme", "verified")
    assert "query tenant=acme status=verified" in caplog.text


def test_a_malformed_log_call_is_not_swallowed_or_turned_into_an_error_by_the_filter():
    """Checked on the record factory itself: pytest deliberately fails a test whose log call is malformed, which would hide
    what we want to know, namely that the filter returns the record instead of raising."""
    install_log_redaction()
    record = logging.getLogRecordFactory()("contextrag.test", logging.INFO, __file__, 1, "bad format %s %s",
                                           ("only-one-arg",), None)
    assert record.msg == "bad format %s %s" and record.args == ("only-one-arg",)  # untouched; the caller's bug stays visible


def test_it_can_be_disabled_explicitly_for_debugging_with_synthetic_data(monkeypatch):
    monkeypatch.setenv("CONTEXTRAG_LOG_REDACTION", "off")
    assert install_log_redaction() is False
