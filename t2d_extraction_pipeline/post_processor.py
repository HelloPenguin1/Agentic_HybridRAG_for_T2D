import json
from pathlib import Path
from collections import defaultdict

class PostProcessor:
    def __init__(self, entities_dir: Path, output_dir: Path):
        self.entities_dir = entities_dir
        self.output_dir = output_dir / "linked"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_reduce(self):
        """Map-Reduce: Consolidates entities across all PDF extractions."""
        global_nodes = defaultdict(lambda: {"occurrence_count": 0, "sources": set(), "metadata": {}})
        global_rels = []

        for file_path in self.entities_dir.glob("*_entities.json"):
            with open(file_path, 'r') as f:
                data_list = json.load(f)
            
            for chunk_data in data_list:
                for key, entities in chunk_data.items():
                    if key == "relationships":
                        global_rels.extend(entities)
                        continue
                    
                    for ent in entities:
                        name = ent['name'].upper().strip() # Clinical normalization
                        node = global_nodes[name]
                        node["canonical_name"] = ent['name']
                        node["type"] = key
                        node["occurrence_count"] += 1
                        node["sources"].add(file_path.stem.replace("_entities", ""))
                        if not node["metadata"]: node["metadata"] = ent

                        # Track metadata for citations
                        if ent.get('page_number'):
                            node["page_numbers"].add(ent['page_number'])
                        if ent.get('source_text'):
                            node["evidence_snippets"].append(ent['source_text'])

        final_nodes = {k: {
            **v, 
            "sources": list(v["sources"]),
            "page_numbers": sorted(list(v["page_numbers"])),
            "evidence_snippets": list(set(v["evidence_snippets"]))[:5] # Keep top 5 for brevity
        } for k, v in global_nodes.items()}
        
        with open(self.output_dir / "unified_graph.json", 'w') as f:
            json.dump({"nodes": final_nodes, "relationships": global_rels}, f, indent=2)
        
        print(f"✅ Map-Reduce Complete: {len(final_nodes)} unique entities found.")