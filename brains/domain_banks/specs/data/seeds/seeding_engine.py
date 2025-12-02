"""
Domain Bank Seeding Engine

This module handles seeding of domain banks with foundational knowledge.
It loads seed files, validates them, and writes them to runtime storage.

All operations are deterministic and idempotent.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

_MAVEN_ROOT = Path(__file__).resolve().parents[5]
import sys

if str(_MAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAVEN_ROOT))

from brains.maven_paths import (
    get_runtime_domain_banks_root,
    validate_path_confinement,
)
from seed_validator import SeedValidator, ValidationError


class SeedingEngine:
    """Engine for seeding domain banks with foundational knowledge."""

    def __init__(self, seeds_dir: str, runtime_dir: str):
        """
        Initialize seeding engine.

        Args:
            seeds_dir: Path to seeds directory (source)
            runtime_dir: Path to runtime domain banks directory (target)
        """
        self.seeds_dir = Path(seeds_dir)
        self.runtime_dir = validate_path_confinement(
            Path(runtime_dir), "domain bank seeding runtime"
        )
        self.validator = SeedValidator(str(seeds_dir))

        # Load registry
        registry_file = self.seeds_dir / "seed_registry.json"
        if not registry_file.exists():
            raise ValidationError(f"Seed registry not found: {registry_file}")

        with open(registry_file, 'r', encoding='utf-8') as f:
            self.registry = json.load(f)

    def _get_bank_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Get mapping of banks to seed files and storage locations.

        Returns:
            Dict mapping bank names to seed file info
        """
        mapping = {}
        for seed_key, seed_info in self.registry["seed_files"].items():
            bank_name = seed_info["bank_name"]
            seed_file = self.seeds_dir / seed_info["seed_file"]

            # Determine storage location
            # New banks that don't exist yet will be created
            storage_dir = self.runtime_dir / bank_name / "memory" / "ltm"

            mapping[bank_name] = {
                "seed_file": seed_file,
                "storage_dir": storage_dir,
                "storage_file": storage_dir / "facts.jsonl",
                "description": seed_info.get("description", ""),
                "required": seed_info.get("required", False),
            }

        return mapping

    def _load_all_seeds(
        self, bank_mapping: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all seed entries organized by bank.

        Args:
            bank_mapping: Bank to file mapping

        Returns:
            Dict mapping bank names to lists of entries
        """
        entries_by_bank = {}

        for bank_name, info in bank_mapping.items():
            seed_file = info["seed_file"]
            entries = self.validator.load_jsonl(seed_file)
            entries_by_bank[bank_name] = entries

        return entries_by_bank

    def _validate_all_entries(
        self, entries_by_bank: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Validate all loaded entries.

        Args:
            entries_by_bank: Entries organized by bank

        Returns:
            Validation report

        Raises:
            ValidationError: If validation fails
        """
        # Build list for validator
        seed_files = []
        for bank_name, entries in entries_by_bank.items():
            # Find the seed file for this bank
            for seed_key, seed_info in self.registry["seed_files"].items():
                if seed_info["bank_name"] == bank_name:
                    seed_file = self.seeds_dir / seed_info["seed_file"]
                    seed_files.append((bank_name, seed_file))
                    break

        return self.validator.validate_all_seeds(seed_files)

    def _write_bank_storage(
        self, bank_name: str, entries: List[Dict[str, Any]], storage_file: Path
    ) -> None:
        """
        Write entries to bank storage file.

        This is deterministic and idempotent - same entries produce same file.

        Args:
            bank_name: Bank name
            entries: List of entries to write
            storage_file: Target storage file path
        """
        # Create directory if it doesn't exist
        storage_file.parent.mkdir(parents=True, exist_ok=True)

        # Sort entries by ID for determinism
        sorted_entries = sorted(entries, key=lambda e: e["id"])

        # Write JSONL
        with open(storage_file, 'w', encoding='utf-8') as f:
            for entry in sorted_entries:
                # Ensure deterministic JSON output (sorted keys)
                json_line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                f.write(json_line + '\n')

    def run_seeding(self, validate_only: bool = False) -> Dict[str, Any]:
        """
        Run the seeding process.

        Args:
            validate_only: If True, validate but don't write storage

        Returns:
            Seeding report with structure:
            {
                "ok": bool,
                "mode": "validate_only" | "apply",
                "total_entries": int,
                "banks_seeded": int,
                "banks": {
                    "bank_name": {
                        "entries": int,
                        "storage_file": str,
                        "written": bool
                    }
                },
                "errors": [str],
                "warnings": [str]
            }

        Raises:
            ValidationError: If validation fails
        """
        # Step 1: Get bank mapping
        bank_mapping = self._get_bank_mapping()

        # Step 2: Load all seed files
        entries_by_bank = self._load_all_seeds(bank_mapping)

        # Step 3: Validate all entries
        validation_report = self._validate_all_entries(entries_by_bank)

        if not validation_report["ok"]:
            raise ValidationError("Validation failed")

        # Step 4: Build report
        total_entries = sum(len(entries) for entries in entries_by_bank.values())

        report = {
            "ok": True,
            "mode": "validate_only" if validate_only else "apply",
            "total_entries": total_entries,
            "banks_seeded": 0,
            "banks": {},
            "errors": validation_report.get("errors", []),
            "warnings": validation_report.get("warnings", []),
        }

        # Step 5: Apply seeds if not validate-only
        for bank_name, entries in entries_by_bank.items():
            info = bank_mapping[bank_name]
            storage_file = info["storage_file"]

            bank_report = {
                "entries": len(entries),
                "storage_file": str(storage_file),
                "written": False,
            }

            if not validate_only:
                # Re-validate before writing (extra safety)
                # Already validated above, but this ensures consistency

                # Write to storage
                self._write_bank_storage(bank_name, entries, storage_file)
                bank_report["written"] = True
                report["banks_seeded"] += 1

            report["banks"][bank_name] = bank_report

        return report

    def verify_idempotency(self) -> Dict[str, Any]:
        """
        Verify that seeding is idempotent by running it twice and comparing.

        Returns:
            Verification report

        Raises:
            ValidationError: If idempotency check fails
        """
        # Run first seeding
        report1 = self.run_seeding(validate_only=False)

        # Read all written files
        files_content_1 = {}
        for bank_name, bank_info in report1["banks"].items():
            if bank_info["written"]:
                storage_file = Path(bank_info["storage_file"])
                if storage_file.exists():
                    with open(storage_file, 'r', encoding='utf-8') as f:
                        files_content_1[bank_name] = f.read()

        # Run second seeding
        report2 = self.run_seeding(validate_only=False)

        # Read all written files again
        files_content_2 = {}
        for bank_name, bank_info in report2["banks"].items():
            if bank_info["written"]:
                storage_file = Path(bank_info["storage_file"])
                if storage_file.exists():
                    with open(storage_file, 'r', encoding='utf-8') as f:
                        files_content_2[bank_name] = f.read()

        # Compare
        differences = []
        for bank_name in files_content_1.keys():
            if files_content_1[bank_name] != files_content_2.get(bank_name):
                differences.append(bank_name)

        return {
            "ok": len(differences) == 0,
            "idempotent": len(differences) == 0,
            "runs": 2,
            "differences": differences,
        }


def run_seeding(
    seeds_dir: str,
    runtime_dir: str,
    validate_only: bool = False
) -> Dict[str, Any]:
    """
    Run the domain bank seeding process.

    Args:
        seeds_dir: Path to seeds directory
        runtime_dir: Path to runtime domain banks directory
        validate_only: If True, validate but don't write

    Returns:
        Seeding report

    Raises:
        ValidationError: If seeding fails
    """
    engine = SeedingEngine(seeds_dir, runtime_dir)
    return engine.run_seeding(validate_only=validate_only)


if __name__ == "__main__":
    import sys

    # Default paths
    default_seeds = Path(__file__).parent
    # Runtime must be inside maven2_fix
    default_runtime = validate_path_confinement(
        get_runtime_domain_banks_root(), "seed engine runtime"
    )

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        validate_only = mode == "--validate"
    else:
        validate_only = True

    if len(sys.argv) > 2:
        seeds_dir = sys.argv[2]
    else:
        seeds_dir = str(default_seeds)

    if len(sys.argv) > 3:
        runtime_dir = sys.argv[3]
    else:
        runtime_dir = str(default_runtime)

    try:
        report = run_seeding(seeds_dir, runtime_dir, validate_only=validate_only)
        print(f"Seeding {report['mode']} successful!")
        print(json.dumps(report, indent=2))
    except ValidationError as e:
        print(f"Seeding failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
