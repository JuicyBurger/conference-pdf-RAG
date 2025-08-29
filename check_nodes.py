import json

# Load nodes
with open('data/prepared/113年報 20240531-16-18/nodes_all.json', 'r', encoding='utf-8') as f:
    nodes = json.load(f)

print(f"Total nodes: {len(nodes)}")

# Analyze by type
type_counts = {}
type_text_lengths = {}

for node in nodes:
    node_type = node.get('type', 'unknown')
    text_length = len(node.get('text', ''))
    
    if node_type not in type_counts:
        type_counts[node_type] = 0
        type_text_lengths[node_type] = []
    
    type_counts[node_type] += 1
    type_text_lengths[node_type].append(text_length)

print("\nNode types and counts:")
for node_type, count in type_counts.items():
    avg_length = sum(type_text_lengths[node_type]) / len(type_text_lengths[node_type]) if type_text_lengths[node_type] else 0
    print(f"  {node_type}: {count} nodes, avg text length: {avg_length:.1f} chars")

# Simulate the filtering logic
print("\nSimulating VectorIndexer filtering:")
valid_texts = []
skipped_count = 0

for node in nodes:
    node_type = node.get('type', 'unknown')
    text = node.get('text', '').strip()
    
    if not text:
        print(f"  Skipping {node_type}: empty text")
        continue
    
    if node_type in ['table_record', 'table_column']:
        print(f"  Skipping {node_type}: handled by KG pipeline")
        skipped_count += 1
        continue
    
    # Simulate chunking
    if node_type == 'paragraph':
        from src.data.chunker import chunk_text
        chunks = chunk_text(text)
        valid_texts.extend(chunks)
        print(f"  {node_type}: {len(chunks)} chunks from {len(text)} chars")
    elif node_type in ['table_summary', 'table_note', 'table_chunk']:
        from src.data.chunker import chunk_text
        chunks = chunk_text(text, content_type="table")
        valid_texts.extend(chunks)
        print(f"  {node_type}: {len(chunks)} chunks from {len(text)} chars")
    else:
        valid_texts.append(text)
        print(f"  {node_type}: 1 chunk from {len(text)} chars")

print(f"\nResults:")
print(f"  Skipped nodes: {skipped_count}")
print(f"  Valid texts after filtering: {len(valid_texts)}")
print(f"  Expected Qdrant points: {len(valid_texts)}")
