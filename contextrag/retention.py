"""Monthly partitions and retention for the audit tables.

Both audit tables (``answer_audit``, ``retrieval_audit``) are RANGE-partitioned on ``ts`` by calendar month
(UTC). Retention drops whole expired months, which is a catalog operation rather than a row-by-row DELETE, and
then deletes any remaining rows older than the cutoff. Creating and dropping partitions needs table ownership,
so every function here takes an *admin* (owner) connection, never the application role.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger("contextrag.retention")

AUDIT_TABLES = ("answer_audit", "retrieval_audit")


@dataclass(frozen=True)
class TableRetention:
    table: str
    dropped_partitions: tuple[str, ...]
    rows_removed: int  # rows older than the cutoff, whether they lived in a dropped partition or not
    dry_run: bool


def first_of_month(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def add_months(first: datetime, n: int) -> datetime:
    """``first`` must be the first instant of a month; returns the first instant of the month ``n`` months later."""
    year, month = divmod(first.month - 1 + n, 12)
    return first.replace(year=first.year + year, month=month + 1)


def month_of(ts: float) -> datetime:
    return first_of_month(datetime.fromtimestamp(ts, tz=timezone.utc))


def partition_name(table: str, start: datetime) -> str:
    return f"{table}_{start:%Y%m}"


def ensure_partition(conn, table: str, ts: float) -> str:
    """Create the monthly partition of ``table`` that holds ``ts`` if it does not exist. Returns its name."""
    from psycopg import sql

    start = month_of(ts)
    end = add_months(start, 1)
    name = partition_name(table, start)
    conn.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})").format(
        sql.Identifier(name), sql.Identifier(table), sql.Literal(start), sql.Literal(end)))
    return name


def ensure_partitions(conn, table: str, months_ahead: int = 2, now: float | None = None) -> None:
    """Create the partitions of the current month and of the next ``months_ahead`` months."""
    start = month_of(time.time() if now is None else now)
    for k in range(months_ahead + 1):
        ensure_partition(conn, table, add_months(start, k).timestamp())


def list_partitions(conn, table: str) -> list[tuple[str, datetime]]:
    """(name, first instant of its month) for every monthly partition of ``table``, oldest first."""
    rows = conn.execute(
        """SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
           WHERE i.inhparent = to_regclass(%s)""", (table,)).fetchall()
    pattern = re.compile(rf"^{re.escape(table)}_(\d{{4}})(\d{{2}})$")
    out = []
    for (name,) in rows:
        m = pattern.match(name)
        if m:
            out.append((name, datetime(int(m[1]), int(m[2]), 1, tzinfo=timezone.utc)))
    return sorted(out, key=lambda p: p[1])


def retention_cutoff(months: int, now: float | None = None) -> datetime:
    """Keep the current month and the ``months - 1`` before it: the cutoff is the start of the oldest kept month."""
    if months < 1:
        raise ValueError("months must be >= 1")
    return add_months(month_of(time.time() if now is None else now), -(months - 1))


def apply_retention(conn, table: str, cutoff: datetime, *, dry_run: bool = False) -> TableRetention:
    """Delete everything in ``table`` older than ``cutoff`` (timezone-aware). Whole expired months are dropped;
    the rest is deleted row by row. Runs in one transaction. ``dry_run`` only reports what would be removed."""
    from psycopg import sql

    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    expired = [name for name, start in list_partitions(conn, table) if add_months(start, 1) <= cutoff]
    with conn.transaction():
        removed = conn.execute(sql.SQL("SELECT count(*) FROM {} WHERE ts < %s").format(sql.Identifier(table)),
                               (cutoff,)).fetchone()[0]
        if not dry_run:
            for name in expired:
                conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(name)))
            conn.execute(sql.SQL("DELETE FROM {} WHERE ts < %s").format(sql.Identifier(table)), (cutoff,))
    result = TableRetention(table, tuple(expired), removed, dry_run)
    log.info("retention table=%s cutoff=%s dropped_partitions=%d rows_removed=%d dry_run=%s",
             table, cutoff.isoformat(), len(expired), removed, dry_run)
    return result


def run_retention(conn, months: int, *, dry_run: bool = False, tables: tuple[str, ...] = AUDIT_TABLES,
                  now: float | None = None) -> list[TableRetention]:
    """Apply the same cutoff (keep ``months`` calendar months) to every audit table; one transaction per table."""
    cutoff = retention_cutoff(months, now)
    present = [t for t in tables if conn.execute("SELECT to_regclass(%s) IS NOT NULL", (t,)).fetchone()[0]]
    return [apply_retention(conn, t, cutoff, dry_run=dry_run) for t in present]  # a deployment may lack a table


def older_than(seconds: float, now: float | None = None) -> datetime:
    """Cutoff for an age-based purge: everything older than ``seconds``."""
    base = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    return base - timedelta(seconds=float(seconds))
