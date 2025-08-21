"""
Synchronous Neo4j Table Ingestion + Qdrant Table Embedding

Schema-free data-cube pattern for ingesting table observations to Neo4j.
Hybrid embedding strategy for Qdrant: table-level + row-level embeddings.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
import re

from .graph_store import get_driver
from src.config import NEO4J_DATABASE, QDRANT_COLLECTION
from src.rag.indexing.indexer import index_text_payloads

logger = logging.getLogger(__name__)


def norm_label(s: str) -> str:
    """Normalize header/label whitespace"""
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)  # collapse all whitespace in labels
    s = s.replace("　", "")    # full-width spaces
    return s


def extract_panel_from_row(rec: dict, headers: List[str]) -> Optional[str]:
    """Extract panel from row data using heuristics"""
    # Look for likely panel columns (first column, or columns with descriptive names)
    panel_candidates = []
    
    # Check first column
    if headers and headers[0] in rec:
        panel_candidates.append(rec[headers[0]])
    
    # Check for columns that might be panels (e.g., "評估內容及方式")
    for header in headers:
        if any(keyword in header for keyword in ["內容", "方式", "類型", "分類", "項目"]):
            if header in rec:
                panel_candidates.append(rec[header])
    
    # Return first non-empty candidate, stripping parentheticals
    for candidate in panel_candidates:
        if candidate:
            # Strip parenthetical content
            panel = re.sub(r"\s*\(.*?\)\s*", "", str(candidate)).strip()
            if panel:
                return panel
    
    return None


def create_row_level_embedding(observations: List[dict], doc_id: str, table_id: str, page: Optional[int] = None, structured_data: dict = None) -> List[Dict[str, Any]]:
    """Create row-level embeddings for Qdrant"""
    embeddings = []
    
    # Group observations by row
    row_groups = {}
    for obs in observations:
        row_idx = obs['row_idx']
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(obs)
    
    for row_idx, row_obs in row_groups.items():
        if not row_obs:
            continue
            
        # Get row metadata
        first_obs = row_obs[0]
        
        # Extract panel using heuristics if not already set
        panel = first_obs.get('panel')
        if not panel and structured_data:
            headers = structured_data.get('headers', [])
            # Get the original record for this row
            records = structured_data.get('records', [])
            if row_idx < len(records):
                rec = records[row_idx]
                panel = extract_panel_from_row(rec, headers)
        
        # Normalize panel for consistent filtering
        panel = norm_label(panel) if panel else '-'
        
        # Build dimensions text (unique key-value pairs)
        dims_text = []
        seen_dims = set()
        for obs in row_obs:
            for key, value in obs['dimensions'].items():
                norm_key = norm_label(key)
                norm_value = norm_label(str(value))
                dim_pair = f"{norm_key}={norm_value}"
                if dim_pair not in seen_dims:
                    dims_text.append(dim_pair)
                    seen_dims.add(dim_pair)
        dims_str = " | ".join(dims_text)
        
        # Build metrics text
        metrics_text = []
        for obs in row_obs:
            metric = norm_label(obs['metric_leaf'])
            value = obs['value_raw']
            unit = obs.get('unit', '')
            unit_str = f" {unit}" if unit else ""
            metrics_text.append(f"{metric}={value}{unit_str}")
        metrics_str = " ; ".join(metrics_text)
        
        # Create embedding text with proper formatting
        page_str = f" p{page}" if page else ""
        text = f"[{doc_id}{page_str} tbl:{table_id} row:{row_idx} panel={panel}]\n{dims_str} || {metrics_str}"
        
        # Create embedding ID with proper formatting
        embedding_id = f"{doc_id}#tbl:{table_id}#row:{row_idx}"
        
        # Create flattened dimensions map
        dimensions = {}
        for obs in row_obs:
            for key, value in obs['dimensions'].items():
                norm_key = norm_label(key)
                dimensions[norm_key] = norm_label(str(value))
        
        # Add text hash for deduplication
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        embeddings.append({
            "text": text,
            "id": embedding_id,
            "level": "row",
            "doc_id": doc_id,
            "page": page,
            "table_id": table_id,
            "row_idx": row_idx,
            "panel": panel,
            "dimensions": dimensions,
            "metric_names": [norm_label(obs['metric_leaf']) for obs in row_obs],
            "value_count": len(row_obs),
            "scope": "table",  # Mark as table scope for retrieval
            "text_hash": text_hash
        })
    
    return embeddings


def create_table_level_embedding(observations: List[dict], structured_data: dict, doc_id: str, table_id: str, page: Optional[int] = None, analysis: dict = None) -> Dict[str, Any]:
    """Create table-level embedding for Qdrant"""
    
    # Get table metadata
    headers = structured_data.get('headers', [])
    notes = structured_data.get('notes', [])
    
    # Get table type from analysis or observations
    table_type = "unknown"
    if analysis and analysis.get('table_type'):
        table_type = analysis['table_type']
    elif observations:
        # Get table type from first observation (they should all be the same)
        table_type = observations[0].get('table_type', 'unknown')
    
    # Get unique panels (sorted for deterministic payloads)
    panels = sorted({p for p in (obs.get('panel') for obs in observations) if p})
    panel_names = " | ".join(panels) if panels else "none"
    
    # Get header terms (normalized and sorted for deterministic payloads)
    header_terms = " | ".join(sorted(norm_label(h) for h in headers)) if headers else "none"
    
    # Get notes text
    notes_text = ""
    if notes:
        notes_text = " ; ".join(note.get('text', '') for note in notes if note.get('text'))
    
    # Get example observations (first few)
    examples = []
    for obs in observations[:3]:  # First 3 observations as examples
        metric = norm_label(obs['metric_leaf'])
        value = obs['value_raw']
        unit = obs.get('unit', '')
        unit_str = f" {unit}" if unit else ""
        examples.append(f"{metric}={value}{unit_str}")
    examples_str = " | ".join(examples) if examples else "none"
    
    # Create embedding text
    page_str = f" p{page}" if page else ""
    text = f"[{doc_id}{page_str} tbl:{table_id} type={table_type}]\npanels={panel_names} ; headers={header_terms}"
    if notes_text:
        text += f" ; notes={notes_text}"
    text += f"\nexamples: {examples_str}"
    
    # Create embedding ID
    embedding_id = f"{doc_id}#tbl:{table_id}"
    
    # Add text hash for deduplication
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    
    return {
        "text": text,
        "id": embedding_id,
        "level": "table",
        "doc_id": doc_id,
        "page": page,
        "table_id": table_id,
        "table_type": table_type,
        "panel_names": panels,
        "header_terms": sorted(norm_label(h) for h in headers),
        "observation_count": len(observations),
        "scope": "table",  # Mark as table scope for retrieval
        "text_hash": text_hash
    }


def embed_table_to_qdrant(observations: List[dict], structured_data: dict, doc_id: str, table_id: str, page: Optional[int] = None, analysis: dict = None) -> dict:
    """Embed table to Qdrant using hybrid strategy"""
    
    if not observations:
        return {'row_embeddings': 0, 'table_embeddings': 0}
    
    try:
        # Build row groups first
        row_groups = {}
        for obs in observations:
            row_groups.setdefault(obs['row_idx'], []).append(obs)
        
        # Apply row capping for large tables (≤200 rows)
        if len(row_groups) > 200:
            selected = set()
            
            # 1) Always include aggregates
            for rid, group in row_groups.items():
                if any(o.get('aggregate', False) for o in group):
                    selected.add(rid)
            
            # 2) TODO: Include extremes per metric
            # TODO: Implement extreme value selection per metric
            
            # 3) Fill remaining slots with first rows
            for rid in sorted(row_groups.keys()):
                if len(selected) >= 200:
                    break
                if rid not in selected:
                    selected.add(rid)
            
            # Filter observations to selected rows
            row_embeddings_to_create = [o for o in observations if o['row_idx'] in selected]
            
            logger.info(f"Table {table_id} has {len(row_groups)} rows, capping to {len(selected)} rows for embedding")
        else:
            row_embeddings_to_create = observations
        
        # Create row-level embeddings
        row_embeddings = create_row_level_embedding(row_embeddings_to_create, doc_id, table_id, page, structured_data)
        
        # Create table-level embedding
        table_embedding = create_table_level_embedding(observations, structured_data, doc_id, table_id, page, analysis)
        
        # Prepare all embeddings for Qdrant
        all_embeddings = row_embeddings + [table_embedding]
        
        # Index to Qdrant
        qdrant_count = index_text_payloads(all_embeddings, collection_name=QDRANT_COLLECTION)
        
        logger.info(f"Embedded table {table_id} to Qdrant: {len(row_embeddings)} row embeddings + 1 table embedding = {qdrant_count} total")
        
        return {
            'row_embeddings': len(row_embeddings),
            'table_embeddings': 1,
            'total_embeddings': qdrant_count
        }
        
    except Exception as e:
        logger.error(f"Error embedding table {table_id} to Qdrant: {e}")
        return {'row_embeddings': 0, 'table_embeddings': 0, 'error': str(e)}


def process_neo4j_batch(observations: List[dict], doc_id: str, table_id: str) -> dict:
    """Process a batch of observations for Neo4j ingestion synchronously"""
    
    if not observations:
        return {'nodes_created': 0, 'relationships_created': 0}
    
    logger.info(f"Processing {len(observations)} observations for doc_id={doc_id}, table_id={table_id}")
    
    try:
        # Schema-free data-cube pattern
        cypher = """
        UNWIND $observations AS obs
        MERGE (ds:Dataset {doc_id: $doc_id})
        MERGE (tb:Table {doc_id: $doc_id, table_id: $table_id})-[:IN_DATASET]->(ds)

        WITH obs, tb
        WITH obs, tb, (CASE
        WHEN size(obs.metric_path) > 0 THEN last(obs.metric_path)
        ELSE obs.metric_leaf
        END) AS metric_leaf

        MERGE (m:Metric {name: metric_leaf})
        ON CREATE SET m.unit = coalesce(obs.unit, m.unit)

        MERGE (o:Observation {source_hash: obs.source_hash})
        ON CREATE SET
            o.value_raw = obs.value_raw,
            o.value_num = obs.value_num,
            o.unit = obs.unit,
            o.unit_source = obs.unit_source,
            o.unit_confidence = obs.unit_confidence,
            o.panel = obs.panel,
            o.panel_confidence = obs.panel_confidence,
            o.row_idx = obs.row_idx,
            o.col_idx = obs.col_idx,
            o.doc_id = $doc_id,
            o.table_id = $table_id
        ON MATCH SET
            o.value_raw = obs.value_raw,
            o.value_num = obs.value_num,
            o.unit = obs.unit,
            o.unit_source = obs.unit_source,
            o.unit_confidence = obs.unit_confidence,
            o.panel = obs.panel,
            o.panel_confidence = obs.panel_confidence

        MERGE (o)-[:IN_TABLE]->(tb)
        MERGE (o)-[:OF_METRIC]->(m)

        WITH o, obs
        UNWIND keys(obs.dimensions) AS dim_key
        WITH o, dim_key, obs.dimensions[dim_key] AS dim_value
        MERGE (dt:DimType {name: dim_key})
        MERGE (dv:DimVal {value: toString(dim_value)})-[:OF_TYPE]->(dt)
        MERGE (o)-[:FOR_DIM {key: dim_key}]->(dv);
        """
        
        # Execute with idempotency (source_hash prevents duplicates)
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, {
                'observations': observations,
                'doc_id': doc_id,
                'table_id': table_id
            })
            
            # Get summary statistics
            summary = result.consume()
            stats = {
                'nodes_created': summary.counters.nodes_created,
                'relationships_created': summary.counters.relationships_created
            }
            
            logger.info(f"Completed: {stats['nodes_created']} nodes created, {stats['relationships_created']} relationships created")
            return stats
            
    except Exception as e:
        logger.error(f"Error processing observations for doc_id={doc_id}, table_id={table_id}: {e}")
        raise


def create_chunk_nodes_for_tables(observations: List[dict], doc_id: str, table_id: str, page: Optional[int] = None) -> List[str]:
    """Create Chunk nodes in Neo4j for table embeddings to enable graph retrieval.
    
    Args:
        observations: List of table observations
        doc_id: Document ID
        table_id: Table ID
        page: Page number
        
    Returns:
        List of chunk IDs created
    """
    chunk_ids = []
    
    try:
        with get_driver().session(database=NEO4J_DATABASE) as session:
            # Create table-level chunk
            table_chunk_id = f"{doc_id}#tbl:{table_id}"
            table_text = f"Table {table_id} on page {page} with {len(observations)} observations"
            
            # Create or merge table chunk node
            cypher = """
            MERGE (c:Chunk {id: $chunk_id})
            ON CREATE SET 
                c.text = $text,
                c.doc_id = $doc_id,
                c.page = $page,
                c.table_id = $table_id,
                c.type = 'table_summary',
                c.observation_count = $obs_count
            ON MATCH SET 
                c.text = $text,
                c.observation_count = $obs_count
            RETURN c.id as chunk_id
            """
            
            result = session.run(cypher, {
                'chunk_id': table_chunk_id,
                'text': table_text,
                'doc_id': doc_id,
                'page': page,
                'table_id': table_id,
                'obs_count': len(observations)
            })
            
            chunk_ids.append(table_chunk_id)
            logger.info(f"Created table chunk node: {table_chunk_id}")
            
            # Create row-level chunks for each unique row
            row_groups = {}
            for obs in observations:
                row_idx = obs['row_idx']
                if row_idx not in row_groups:
                    row_groups[row_idx] = []
                row_groups[row_idx].append(obs)
            
            for row_idx, row_obs in row_groups.items():
                if not row_obs:
                    continue
                
                # Create row chunk ID
                row_chunk_id = f"{doc_id}#tbl:{table_id}#row:{row_idx}"
                
                # Build row text
                first_obs = row_obs[0]
                dims_text = " | ".join([f"{k}={v}" for k, v in first_obs['dimensions'].items()])
                metrics_text = " | ".join([f"{obs['metric_leaf']}={obs['value_raw']}" for obs in row_obs])
                row_text = f"Row {row_idx}: {dims_text} || {metrics_text}"
                
                # Create or merge row chunk node
                result = session.run(cypher, {
                    'chunk_id': row_chunk_id,
                    'text': row_text,
                    'doc_id': doc_id,
                    'page': page,
                    'table_id': table_id,
                    'obs_count': len(row_obs)
                })
                
                chunk_ids.append(row_chunk_id)
                logger.info(f"Created row chunk node: {row_chunk_id}")
            
            # Link chunks to table observations
            for obs in observations:
                obs_hash = obs['source_hash']
                row_chunk_id = f"{doc_id}#tbl:{table_id}#row:{obs['row_idx']}"
                
                # Link row chunk to observation
                link_cypher = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (o:Observation {source_hash: $obs_hash})
                MERGE (c)-[:HAS_OBSERVATION]->(o)
                """
                session.run(link_cypher, {
                    'chunk_id': row_chunk_id,
                    'obs_hash': obs_hash
                })
            
            logger.info(f"Created {len(chunk_ids)} chunk nodes for table {table_id}")
            return chunk_ids
            
    except Exception as e:
        logger.error(f"Failed to create chunk nodes for table {table_id}: {e}")
        return []


def extract_and_ingest_table_kg(structured_data: dict, doc_id: str, table_id: str) -> dict:
    """Extract and ingest table knowledge graph to Neo4j + embed to Qdrant.
    
    Args:
        structured_data: Table data with headers, records, etc.
        doc_id: Document ID
        table_id: Table ID
        
    Returns:
        Dictionary with ingestion results
    """
    try:
        # Extract table KG
        from src.data.table_kg_extractor import extract_table_kg
        analysis, observations = extract_table_kg(structured_data, doc_id, table_id, include_aggregates=True)
        
        if not observations:
            logger.warning(f"No observations extracted from table {table_id}")
            return {"observations": 0, "qdrant_embeddings": 0}
        
        # Get page from structured data
        page = structured_data.get('page', 1)
        
        # Create chunk nodes for graph retrieval
        chunk_ids = create_chunk_nodes_for_tables(observations, doc_id, table_id, page)
        logger.info(f"Created {len(chunk_ids)} chunk nodes for table {table_id}")
        
        # Ingest to Neo4j
        neo4j_count = process_neo4j_batch(observations, doc_id, table_id)
        
        # Embed to Qdrant
        qdrant_count = embed_table_to_qdrant(observations, analysis, doc_id, table_id, page, structured_data)
        
        logger.info(f"Table {table_id}: {neo4j_count} observations to Neo4j + {qdrant_count} embeddings to Qdrant")
        
        return {
            "observations": neo4j_count,
            "qdrant_embeddings": qdrant_count,
            "chunk_nodes": len(chunk_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to extract and ingest table KG for {table_id}: {e}")
        return {"observations": 0, "qdrant_embeddings": 0, "error": str(e)}
