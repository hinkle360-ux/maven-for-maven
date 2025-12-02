"""
Domain Bank Seed Validator

This module provides validation for domain bank seed files.
It enforces schema compliance, uniqueness constraints, and consistency rules.

All validation is deterministic and will hard-fail on any violation.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# JSON Schema validation - using simple validation without external dependencies
# to maintain determinism and avoid external library issues


class ValidationError(Exception):
    """Raised when seed validation fails."""
    pass


class SeedValidator:
    """Validates domain bank seed files against the seed schema."""

    # Allowed values from schema
    ALLOWED_BANKS = {
        "science",
        "technology",
        "language_arts",
        "working_theories",
        "personal",
        "governance_rules",
        "coding_patterns",
        "planning_patterns",
        "creative_templates",
        "environment_rules",
        "conflict_resolution_patterns",
    }

    ALLOWED_KINDS = {
        "law",
        "concept",
        "definition",
        "principle",
        "rule",
        "pattern",
        "template",
        "constraint",
        "theory",
        "fact",
        "guideline",
        "strategy",
        "heuristic",
    }

    REQUIRED_FIELDS = {"id", "bank", "kind", "content", "tier"}
    OPTIONAL_FIELDS = {"confidence", "source", "deterministic"}
    ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

    ID_PATTERN = re.compile(r"^[a-z_]+:[a-z_]+:[a-z0-9_]+$")
    TAG_PATTERN = re.compile(r"^[a-z0-9_]+$")

    def __init__(self, seeds_dir: str):
        """
        Initialize validator.

        Args:
            seeds_dir: Path to seeds directory
        """
        self.seeds_dir = Path(seeds_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and parse a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of parsed JSON objects

        Raises:
            ValidationError: If file cannot be parsed
        """
        entries = []
        if not file_path.exists():
            raise ValidationError(f"Seed file not found: {file_path}")

        # Skip empty files
        if file_path.stat().st_size == 0:
            return entries

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        raise ValidationError(
                            f"JSON parse error in {file_path.name} at line {line_num}: {e}"
                        )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Error reading {file_path.name}: {e}")

        return entries

    def validate_entry_schema(
        self, entry: Dict[str, Any], file_name: str, line_num: int
    ) -> None:
        """
        Validate a single entry against the schema.

        Args:
            entry: Entry to validate
            file_name: Source file name
            line_num: Line number in file

        Raises:
            ValidationError: If validation fails
        """
        prefix = f"{file_name}:{line_num}"

        # Check required fields
        missing = self.REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ValidationError(
                f"{prefix}: Missing required fields: {sorted(missing)}"
            )

        # Check for unknown fields
        unknown = set(entry.keys()) - self.ALL_FIELDS
        if unknown:
            raise ValidationError(
                f"{prefix}: Unknown fields: {sorted(unknown)}"
            )

        # Validate id
        entry_id = entry.get("id")
        if not isinstance(entry_id, str):
            raise ValidationError(
                f"{prefix}: Field 'id' must be string, got {type(entry_id).__name__}"
            )
        if not self.ID_PATTERN.match(entry_id):
            raise ValidationError(
                f"{prefix}: Invalid id format '{entry_id}'. "
                f"Must match pattern: {self.ID_PATTERN.pattern}"
            )

        # Validate bank
        bank = entry.get("bank")
        if not isinstance(bank, str):
            raise ValidationError(
                f"{prefix}: Field 'bank' must be string, got {type(bank).__name__}"
            )
        if bank not in self.ALLOWED_BANKS:
            raise ValidationError(
                f"{prefix}: Invalid bank '{bank}'. Allowed: {sorted(self.ALLOWED_BANKS)}"
            )

        # Validate kind
        kind = entry.get("kind")
        if not isinstance(kind, str):
            raise ValidationError(
                f"{prefix}: Field 'kind' must be string, got {type(kind).__name__}"
            )
        if kind not in self.ALLOWED_KINDS:
            raise ValidationError(
                f"{prefix}: Invalid kind '{kind}'. Allowed: {sorted(self.ALLOWED_KINDS)}"
            )

        # Validate content
        content = entry.get("content")
        if not isinstance(content, dict):
            raise ValidationError(
                f"{prefix}: Field 'content' must be object, got {type(content).__name__}"
            )

        # Validate content.title
        title = content.get("title")
        if not isinstance(title, str):
            raise ValidationError(
                f"{prefix}: Field 'content.title' must be string, got {type(title).__name__}"
            )
        if len(title) < 3 or len(title) > 200:
            raise ValidationError(
                f"{prefix}: Field 'content.title' must be 3-200 chars, got {len(title)}"
            )

        # Validate content.description
        description = content.get("description")
        if not isinstance(description, str):
            raise ValidationError(
                f"{prefix}: Field 'content.description' must be string, "
                f"got {type(description).__name__}"
            )
        if len(description) < 10:
            raise ValidationError(
                f"{prefix}: Field 'content.description' must be at least 10 chars, "
                f"got {len(description)}"
            )

        # Validate optional content fields
        if "tags" in content:
            tags = content["tags"]
            if not isinstance(tags, list):
                raise ValidationError(
                    f"{prefix}: Field 'content.tags' must be array, "
                    f"got {type(tags).__name__}"
                )
            for i, tag in enumerate(tags):
                if not isinstance(tag, str):
                    raise ValidationError(
                        f"{prefix}: content.tags[{i}] must be string, "
                        f"got {type(tag).__name__}"
                    )
                if not self.TAG_PATTERN.match(tag):
                    raise ValidationError(
                        f"{prefix}: Invalid tag '{tag}'. Must match: {self.TAG_PATTERN.pattern}"
                    )

        if "examples" in content:
            examples = content["examples"]
            if not isinstance(examples, list):
                raise ValidationError(
                    f"{prefix}: Field 'content.examples' must be array, "
                    f"got {type(examples).__name__}"
                )

        if "related_ids" in content:
            related = content["related_ids"]
            if not isinstance(related, list):
                raise ValidationError(
                    f"{prefix}: Field 'content.related_ids' must be array, "
                    f"got {type(related).__name__}"
                )
            for i, rid in enumerate(related):
                if not isinstance(rid, str):
                    raise ValidationError(
                        f"{prefix}: content.related_ids[{i}] must be string, "
                        f"got {type(rid).__name__}"
                    )
                if not self.ID_PATTERN.match(rid):
                    raise ValidationError(
                        f"{prefix}: Invalid related_id '{rid}'. Must match ID pattern"
                    )

        # Validate tier
        tier = entry.get("tier")
        if tier != "ltm":
            raise ValidationError(
                f"{prefix}: Field 'tier' must be 'ltm', got '{tier}'"
            )

        # Validate optional fields
        if "confidence" in entry:
            confidence = entry["confidence"]
            if not isinstance(confidence, (int, float)):
                raise ValidationError(
                    f"{prefix}: Field 'confidence' must be number, "
                    f"got {type(confidence).__name__}"
                )
            if not 0.0 <= confidence <= 1.0:
                raise ValidationError(
                    f"{prefix}: Field 'confidence' must be 0.0-1.0, got {confidence}"
                )

        if "source" in entry:
            source = entry["source"]
            if not isinstance(source, str):
                raise ValidationError(
                    f"{prefix}: Field 'source' must be string, got {type(source).__name__}"
                )

        if "deterministic" in entry:
            deterministic = entry["deterministic"]
            if not isinstance(deterministic, bool):
                raise ValidationError(
                    f"{prefix}: Field 'deterministic' must be boolean, "
                    f"got {type(deterministic).__name__}"
                )

    def validate_consistency(
        self, entry: Dict[str, Any], file_name: str, line_num: int
    ) -> None:
        """
        Validate consistency rules.

        Args:
            entry: Entry to validate
            file_name: Source file name
            line_num: Line number

        Raises:
            ValidationError: If consistency check fails
        """
        prefix = f"{file_name}:{line_num}"

        entry_id = entry["id"]
        bank = entry["bank"]

        # ID prefix must match bank
        id_parts = entry_id.split(":")
        if len(id_parts) != 3:
            raise ValidationError(
                f"{prefix}: ID must have exactly 3 parts separated by ':', got {len(id_parts)}"
            )

        id_bank = id_parts[0]
        if id_bank != bank:
            raise ValidationError(
                f"{prefix}: ID bank prefix '{id_bank}' does not match bank '{bank}'"
            )

    def validate_all_seeds(
        self, seed_files: List[Tuple[str, Path]]
    ) -> Dict[str, Any]:
        """
        Validate all seed files.

        Args:
            seed_files: List of (bank_name, file_path) tuples

        Returns:
            Validation report dict

        Raises:
            ValidationError: If any validation fails
        """
        self.errors.clear()
        self.warnings.clear()

        all_entries = []
        entries_by_bank: Dict[str, List[Dict[str, Any]]] = {}
        ids_by_bank: Dict[str, Set[str]] = {}

        # Load and validate each file
        for bank_name, file_path in seed_files:
            entries = self.load_jsonl(file_path)

            if bank_name not in entries_by_bank:
                entries_by_bank[bank_name] = []
                ids_by_bank[bank_name] = set()

            for idx, entry in enumerate(entries, start=1):
                # Schema validation
                self.validate_entry_schema(entry, file_path.name, idx)

                # Consistency validation
                self.validate_consistency(entry, file_path.name, idx)

                # Track for uniqueness check
                entry_id = entry["id"]
                entry_bank = entry["bank"]

                if entry_id in ids_by_bank[entry_bank]:
                    raise ValidationError(
                        f"{file_path.name}:{idx}: Duplicate ID '{entry_id}' in bank '{entry_bank}'"
                    )

                ids_by_bank[entry_bank].add(entry_id)
                entries_by_bank[entry_bank].append(entry)
                all_entries.append(entry)

        # Build report
        report = {
            "ok": True,
            "total_entries": len(all_entries),
            "banks": {},
            "errors": self.errors,
            "warnings": self.warnings,
        }

        for bank_name, entries in entries_by_bank.items():
            report["banks"][bank_name] = {
                "count": len(entries),
                "unique_ids": len(ids_by_bank[bank_name]),
            }

        return report


def validate_seeds(seeds_dir: str) -> Dict[str, Any]:
    """
    Validate all seed files in the seeds directory.

    Args:
        seeds_dir: Path to seeds directory

    Returns:
        Validation report

    Raises:
        ValidationError: If validation fails
    """
    seeds_path = Path(seeds_dir)

    # Load registry to get seed file list
    registry_file = seeds_path / "seed_registry.json"
    if not registry_file.exists():
        raise ValidationError(f"Seed registry not found: {registry_file}")

    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    seed_files = []
    for bank_key, bank_info in registry["seed_files"].items():
        seed_file = seeds_path / bank_info["seed_file"]
        bank_name = bank_info["bank_name"]
        seed_files.append((bank_name, seed_file))

    validator = SeedValidator(seeds_dir)
    return validator.validate_all_seeds(seed_files)


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) > 1:
        seeds_dir = sys.argv[1]
    else:
        # Default path
        seeds_dir = Path(__file__).parent

    try:
        report = validate_seeds(str(seeds_dir))
        print("Validation successful!")
        print(json.dumps(report, indent=2))
    except ValidationError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
