from __future__ import annotations

from typing import Optional

from neo4j import GraphDatabase, Driver

from src.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
            raise RuntimeError("Neo4j configuration missing: NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


def ensure_graph_indexes() -> None:
    driver = get_driver()
    # Create database if it doesn't exist (Enterprise requirement; Desktop supports multi-DB). If not supported, ignore.
    try:
        with driver.session(database="system") as sys_sess:
            sys_sess.run(f"CREATE DATABASE {NEO4J_DATABASE} IF NOT EXISTS")
    except Exception:
        pass
    with driver.session(database=NEO4J_DATABASE) as session:
        # Constraints / indexes for fast upsert
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Corpus) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        # Helpful property indexes for filtering and captions
        session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (cp:Corpus) ON (cp.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.doc_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.corpus_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.page)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (ch:Chunk) ON (ch.name)")
        # Entity indexes
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.corpus_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.doc_id)")


def close_driver():
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        finally:
            _driver = None


