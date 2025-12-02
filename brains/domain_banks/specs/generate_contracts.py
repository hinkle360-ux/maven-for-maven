#!/usr/bin/env python3
"""
Brain Contract Generation Script for Phase 6

This script generates detailed contracts for each brain operation.
Contracts specify:
- input: expected payload fields
- output: returned payload structure
- deterministic: whether same input produces same output
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Core brain contract templates
BRAIN_CONTRACTS = {
    "reasoning_brain": {
        "EVALUATE_FACT": {
            "input": {"proposed_fact": "dict", "evidence": "dict"},
            "output": {"verdict": "str", "confidence": "float", "routing_order": "dict", "answer": "str (optional)"},
            "deterministic": True
        },
        "GENERATE_THOUGHTS": {
            "input": {"query_text": "str", "intent": "str", "entities": "list", "retrieved_memories": "list", "context": "dict"},
            "output": {"thought_steps": "list"},
            "deterministic": True
        },
        "EXPLAIN_LAST": {
            "input": {"last_query": "str", "last_response": "str"},
            "output": {"answer": "str", "confidence": "float"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "planner_brain": {
        "PLAN": {
            "input": {"text": "str", "intent": "str", "context": "dict"},
            "output": {"plan_id": "str", "steps": "list", "priority": "float", "intent": "str"},
            "deterministic": False  # Uses timestamp
        },
        "PLAN_FROM_WM": {
            "input": {"entry": "dict"},
            "output": {"goal": "str"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "memory_librarian": {
        "RUN_PIPELINE": {
            "input": {"query": "str", "skip_stages": "list (optional)"},
            "output": {"stages": "dict", "final_answer": "str"},
            "deterministic": True
        },
        "UNIFIED_RETRIEVE": {
            "input": {"query": "str", "limit": "int", "tier": "str (optional)"},
            "output": {"results": "list", "total": "int"},
            "deterministic": True
        },
        "WM_PUT": {
            "input": {"key": "str", "value": "any", "confidence": "float (optional)"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "WM_GET": {
            "input": {"key": "str"},
            "output": {"entries": "list"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str", "memory_health": "dict"},
            "deterministic": True
        }
    },

    "language_brain": {
        "PARSE": {
            "input": {"text": "str"},
            "output": {"intent": "str", "entities": "list", "storable_type": "str"},
            "deterministic": True
        },
        "GENERATE_CANDIDATES": {
            "input": {"skeleton": "dict", "context": "dict"},
            "output": {"candidates": "list"},
            "deterministic": True
        },
        "FINALIZE": {
            "input": {"candidates": "list", "context": "dict"},
            "output": {"final_answer": "str"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "self_model_brain": {
        "CAN_ANSWER": {
            "input": {"query": "str"},
            "output": {"can_answer": "bool", "beliefs": "list"},
            "deterministic": True
        },
        "QUERY_SELF": {
            "input": {"query": "str"},
            "output": {"text": "str", "confidence": "float", "self_origin": "bool"},
            "deterministic": True
        },
        "DESCRIBE_SELF": {
            "input": {"mode": "str"},
            "output": {"identity": "dict", "capabilities": "list", "limitations": "list"},
            "deterministic": True
        },
        "GET_CAPABILITIES": {
            "input": {},
            "output": {"capabilities": "list"},
            "deterministic": True
        },
        "GET_LIMITATIONS": {
            "input": {},
            "output": {"limitations": "list"},
            "deterministic": True
        },
        "UPDATE_SELF_FACTS": {
            "input": {"updates": "dict"},
            "output": {"updated_facts": "dict"},
            "deterministic": True
        }
    },

    "self_review_brain": {
        "REVIEW_TURN": {
            "input": {"query": "str", "plan": "dict", "thoughts": "list", "answer": "str", "metadata": "dict"},
            "output": {"verdict": "str", "issues": "list", "recommended_action": "str", "notes": "str"},
            "deterministic": True
        },
        "RECOMMEND_TUNING": {
            "input": {"trace_path": "str (optional)"},
            "output": {"suggestions": "list"},
            "deterministic": True
        }
    },

    "self_dmn_brain": {
        "HEALTH": {
            "input": {},
            "output": {"status": "str", "memory_health": "dict"},
            "deterministic": True
        },
        "REGISTER": {
            "input": {"claim": "dict"},
            "output": {"claim_id": "str", "status": "str"},
            "deterministic": True
        },
        "TICK": {
            "input": {},
            "output": {"hum_order": "float", "memory_health": "dict"},
            "deterministic": False  # State changes over time
        },
        "REFLECT": {
            "input": {"window": "int (optional)"},
            "output": {"metrics": "dict", "drafts": "list"},
            "deterministic": True
        },
        "DISSENT_SCAN": {
            "input": {"window": "int (optional)"},
            "output": {"claims": "list", "flagged": "list", "decisions": "list"},
            "deterministic": True
        },
        "RUN_IDLE_CYCLE": {
            "input": {"system_history": "list", "recent_issues": "list", "motivation_state": "dict"},
            "output": {"insights": "list", "actions": "list"},
            "deterministic": True
        },
        "REFLECT_ON_ERROR": {
            "input": {"error_context": "dict", "turn_history": "list"},
            "output": {"insights": "list", "actions": "list"},
            "deterministic": True
        },
        "RUN_LONG_TERM_REFLECTION": {
            "input": {"tier_stats": "dict", "pattern_count": "int", "fact_count": "int", "idle_turns": "int"},
            "output": {"insights": "list", "actions": "list", "action_count": "int"},
            "deterministic": True
        }
    },

    "thought_synthesizer": {
        "SYNTHESIZE": {
            "input": {"plan": "dict", "thought_steps": "list", "memories": "list", "context": "dict"},
            "output": {"final_thoughts": "list", "answer_skeleton": "dict"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "motivation_brain": {
        "GET_STATE": {
            "input": {},
            "output": {"motivation": "dict"},
            "deterministic": True
        },
        "EVALUATE_QUERY": {
            "input": {"query": "str", "context": "dict"},
            "output": {"priority": "float", "drives": "dict"},
            "deterministic": True
        },
        "ADJUST_STATE": {
            "input": {"delta": "dict"},
            "output": {"new_state": "dict"},
            "deterministic": True
        },
        "FORMULATE_GOALS": {
            "input": {"context": "dict"},
            "output": {"goals": "list"},
            "deterministic": True
        },
        "SCORE_DRIVE": {
            "input": {"drive": "str"},
            "output": {"score": "float"},
            "deterministic": True
        },
        "SCORE_OPPORTUNITIES": {
            "input": {"context": "dict"},
            "output": {"opportunities": "list"},
            "deterministic": True
        }
    },

    "pattern_recognition_brain": {
        "EXTRACT_PATTERNS": {
            "input": {"memories": "list", "limit": "int (optional)"},
            "output": {"patterns": "list"},
            "deterministic": True
        },
        "ANALYZE": {
            "input": {"data": "list"},
            "output": {"patterns": "list", "insights": "list"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "integrator_brain": {
        "RESOLVE": {
            "input": {"competing_responses": "list", "context": "dict"},
            "output": {"resolution": "dict", "chosen_response": "str"},
            "deterministic": True
        },
        "STATE": {
            "input": {},
            "output": {"state": "dict"},
            "deterministic": True
        }
    },

    "autonomy_brain": {
        "TICK": {
            "input": {},
            "output": {"actions": "list"},
            "deterministic": False  # Can have spontaneous actions
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "personal_brain": {
        "ADD_FACT": {
            "input": {"subject": "str", "relation": "str", "object": "str"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "QUERY_FACT": {
            "input": {"subject": "str", "relation": "str"},
            "output": {"object": "str or None"},
            "deterministic": True
        },
        "LIST_FACTS": {
            "input": {"limit": "int (optional)"},
            "output": {"facts": "list"},
            "deterministic": True
        },
        "RECORD_LIKE": {
            "input": {"thing": "str", "boost": "float (optional)"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "RECORD_DISLIKE": {
            "input": {"thing": "str", "penalty": "float (optional)"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "GET_PROFILE": {
            "input": {},
            "output": {"profile": "dict"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    }
}

# Specialist brains
SPECIALIST_CONTRACTS = {
    "coder_brain": {
        "PLAN": {
            "input": {"task": "str", "context": "dict"},
            "output": {"plan": "dict"},
            "deterministic": True
        },
        "GENERATE": {
            "input": {"plan": "dict"},
            "output": {"code": "str"},
            "deterministic": True
        },
        "VERIFY": {
            "input": {"code": "str"},
            "output": {"valid": "bool", "errors": "list"},
            "deterministic": True
        },
        "REFINE": {
            "input": {"code": "str", "feedback": "str"},
            "output": {"refined_code": "str"},
            "deterministic": True
        }
    },

    "committee_brain": {
        "CONSULT": {
            "input": {"query": "str", "brains": "list"},
            "output": {"responses": "list", "consensus": "dict"},
            "deterministic": True
        }
    },

    "imaginer_brain": {
        "HYPOTHESIZE": {
            "input": {"premise": "str", "constraints": "dict (optional)"},
            "output": {"hypotheses": "list"},
            "deterministic": True
        }
    },

    "peer_connection_brain": {
        "CONNECT": {
            "input": {"peer_id": "str"},
            "output": {"status": "str"},
            "deterministic": True
        },
        "ASK": {
            "input": {"peer_id": "str", "query": "str"},
            "output": {"response": "str"},
            "deterministic": True
        },
        "DELEGATE": {
            "input": {"peer_id": "str", "task": "dict"},
            "output": {"result": "dict"},
            "deterministic": True
        }
    },

    "environment_brain": {
        "GET_LOCATION": {
            "input": {},
            "output": {"location": "dict"},
            "deterministic": True
        }
    },

    "hearing_brain": {
        "ANALYZE_AUDIO": {
            "input": {"audio_data": "bytes or path"},
            "output": {"transcript": "str", "features": "dict"},
            "deterministic": True
        }
    },

    "vision_brain": {
        "ANALYZE_IMAGE": {
            "input": {"image_data": "bytes or path"},
            "output": {"description": "str", "features": "dict"},
            "deterministic": True
        }
    },

    "replanner_brain": {
        "REPLAN": {
            "input": {"failed_plan": "dict", "failure_reason": "str"},
            "output": {"new_plan": "dict"},
            "deterministic": True
        }
    },

    "council_brain": {
        "ARBITRATE": {
            "input": {"proposals": "list", "criteria": "dict"},
            "output": {"decision": "dict"},
            "deterministic": True
        }
    }
}

# Diagnostic brains
DIAGNOSTIC_CONTRACTS = {
    "abstraction_brain": {
        "CREATE_CONCEPT": {
            "input": {"name": "str", "properties": "dict"},
            "output": {"concept_id": "str"},
            "deterministic": True
        },
        "QUERY_CONCEPT": {
            "input": {"concept_id": "str"},
            "output": {"concept": "dict or None"},
            "deterministic": True
        },
        "UPDATE_CONCEPT": {
            "input": {"concept_id": "str", "updates": "dict"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "affect_priority_brain": {
        "SCORE": {
            "input": {"text": "str"},
            "output": {"valence": "float", "arousal": "float", "priority_delta": "float"},
            "deterministic": True
        },
        "LEARN_FROM_RUN": {
            "input": {"run_data": "dict"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "personality_brain": {
        "PREFERENCES_SNAPSHOT": {
            "input": {},
            "output": {"preferences": "dict"},
            "deterministic": True
        },
        "ADAPT_WEIGHTS_SUGGEST": {
            "input": {"performance_data": "dict"},
            "output": {"suggestions": "list"},
            "deterministic": True
        },
        "LEARN_FROM_RUN": {
            "input": {"run_data": "dict"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "sensorium_brain": {
        "NORMALIZE": {
            "input": {"data": "dict"},
            "output": {"normalized": "dict"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    },

    "system_history_brain": {
        "SUMMARIZE": {
            "input": {"limit": "int (optional)"},
            "output": {"summary": "str", "events": "list"},
            "deterministic": True
        },
        "LOG_REFLECTIONS": {
            "input": {"reflections": "list"},
            "output": {"success": "bool"},
            "deterministic": True
        },
        "HEALTH": {
            "input": {},
            "output": {"status": "str"},
            "deterministic": True
        }
    }
}

def main():
    """Generate complete brain contracts."""
    # Merge all contracts
    all_contracts = {}
    all_contracts.update(BRAIN_CONTRACTS)
    all_contracts.update(SPECIALIST_CONTRACTS)
    all_contracts.update(DIAGNOSTIC_CONTRACTS)

    # Save contracts
    output_path = Path(__file__).parent / "brain_contracts_phase6.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_contracts, f, indent=2, ensure_ascii=False)

    print(f"Brain contracts saved to {output_path}")
    print(f"Total brains with contracts: {len(all_contracts)}")

    # Summary
    total_ops = sum(len(ops) for ops in all_contracts.values())
    deterministic_ops = sum(
        1 for brain_ops in all_contracts.values()
        for op_contract in brain_ops.values()
        if op_contract.get("deterministic", True)
    )

    print(f"Total operations: {total_ops}")
    print(f"Deterministic operations: {deterministic_ops}")
    print(f"Non-deterministic operations: {total_ops - deterministic_ops}")

if __name__ == "__main__":
    main()
