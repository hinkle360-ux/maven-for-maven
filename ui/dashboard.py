#!/usr/bin/env python3
from pathlib import Path
import json

from brains.maven_paths import get_brains_path, get_maven_root, get_reports_path

def count_lines(path: Path) -> int:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0

def load_json(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}

def main():
    root = get_maven_root()
    reports = get_reports_path()
    config_dir = root / 'config'
    brains_dir = get_brains_path()
    # Goal count
    goals_file = brains_dir / 'personal' / 'memory' / 'goals.jsonl'
    goals_count = count_lines(goals_file)
    # QA memory count
    qa_file = reports / 'qa_memory.jsonl'
    qa_count = count_lines(qa_file)
    # Domain stats
    meta_conf = load_json(reports / 'meta_confidence.json')
    domain_count = len(meta_conf) if isinstance(meta_conf, dict) else 0
    # Knowledge graph facts count
    kg = load_json(reports / 'knowledge_graph.json')
    facts_count = len(kg.get('facts', [])) if isinstance(kg, dict) else 0
    # Synonyms count
    syn = load_json(config_dir / 'synonyms.json')
    synonyms_count = len(syn) if isinstance(syn, dict) else 0
    # User mood
    mood = load_json(reports / 'user_mood.json')
    current_mood = mood.get('mood', 'unknown') if isinstance(mood, dict) else 'unknown'
    print('Maven Dashboard')
    print('Active goals:', goals_count)
    print('QA memory entries:', qa_count)
    print('Tracked domains:', domain_count)
    print('Semantic facts stored:', facts_count)
    print('Synonym mappings:', synonyms_count)
    print('Current user mood:', current_mood)
    # Count self‑review improvement goals
    improve_count = 0
    try:
        with open(goals_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                title = str(rec.get('title', '')).strip().lower()
                if title.startswith('improve domain:'):
                    improve_count += 1
    except Exception:
        improve_count = 0
    print('Self‑review improvement goals:', improve_count)
    # Display safety and ethics telemetry
    telemetry = load_json(reports / 'telemetry.json')
    if isinstance(telemetry, dict):
        safety_count = telemetry.get('safety_filter', 0)
        eblock_count = telemetry.get('ethics_block', 0)
        ewarn_count = telemetry.get('ethics_warn', 0)
        print('Safety filter events:', safety_count)
        print('Ethics block events:', eblock_count)
        print('Ethics warn events:', ewarn_count)
    # Attempt to display top and bottom domain confidence adjustments
    try:
        from brains.personal.memory import meta_confidence as _mc
        # Top domains by total attempts (descending)
        top_domains = _mc.get_stats(limit=5)
        # Build bottom domains by sorting all stats by adjustment ascending
        all_stats = _mc.get_stats(limit=1000)
        bottom_domains = sorted(all_stats, key=lambda d: d.get('adjustment', 0.0))[:5]
        print('\nTop domains by usage (total attempts):')
        for rec in top_domains:
            dom = rec.get('domain')
            tot = rec.get('total')
            adj = rec.get('adjustment')
            print(f"  {dom}: total={tot}, adj={adj}")
        print('\nDomains with lowest confidence adjustments:')
        for rec in bottom_domains:
            dom = rec.get('domain')
            adj = rec.get('adjustment')
            print(f"  {dom}: adj={adj}")
    except Exception:
        # If meta confidence cannot be loaded, skip detailed domain stats
        pass
    print('Dashboard generated successfully.')

if __name__ == '__main__':
    main()
