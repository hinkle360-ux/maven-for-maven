#!/usr/bin/env python3
"""
Domain Bank Seeding CLI

Command-line interface for validating and applying domain bank seeds.

Usage:
    python seed_domain_banks.py --validate    # Validate seeds without applying
    python seed_domain_banks.py --apply       # Validate and apply seeds
    python seed_domain_banks.py --verify      # Verify idempotency

Examples:
    # Validate seed files
    ./seed_domain_banks.py --validate

    # Apply seeds to runtime domain banks
    ./seed_domain_banks.py --apply

    # Verify idempotent behavior
    ./seed_domain_banks.py --verify
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure Maven root is on the module search path for shared utilities
_MAVEN_ROOT = Path(__file__).resolve().parents[5]
if str(_MAVEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAVEN_ROOT))

from brains.maven_paths import (
    get_runtime_domain_banks_root,
    validate_path_confinement,
)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from seeding_engine import SeedingEngine, run_seeding
from seed_validator import ValidationError


def print_header(title: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print(f"--- {title} ---")
    print()


def print_report_summary(report: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the seeding report.

    Args:
        report: Seeding report dict
    """
    print_section("Seeding Summary")

    print(f"Mode:           {report['mode']}")
    print(f"Total entries:  {report['total_entries']}")

    if report['mode'] == 'apply':
        print(f"Banks seeded:   {report['banks_seeded']}")

    print()
    print("Banks:")
    print()

    # Sort banks by name for consistent output
    sorted_banks = sorted(report['banks'].items())

    for bank_name, bank_info in sorted_banks:
        status = "✓ written" if bank_info.get('written') else "  skipped"
        entries = bank_info['entries']
        print(f"  {status}  {bank_name:30s}  {entries:3d} entries")

    if report.get('errors'):
        print()
        print("ERRORS:")
        for error in report['errors']:
            print(f"  ! {error}")

    if report.get('warnings'):
        print()
        print("WARNINGS:")
        for warning in report['warnings']:
            print(f"  * {warning}")

    print()


def validate_seeds(seeds_dir: str, runtime_dir: str) -> int:
    """
    Validate seed files without applying.

    Args:
        seeds_dir: Path to seeds directory
        runtime_dir: Path to runtime directory

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("Domain Bank Seed Validation")

    print(f"Seeds directory:   {seeds_dir}")
    print(f"Runtime directory: {runtime_dir}")

    try:
        report = run_seeding(seeds_dir, runtime_dir, validate_only=True)

        print_report_summary(report)

        if report['ok']:
            print("✓ Validation successful!")
            print()
            return 0
        else:
            print("✗ Validation failed!")
            print()
            return 1

    except ValidationError as e:
        print()
        print(f"✗ Validation Error: {e}")
        print()
        return 1
    except Exception as e:
        print()
        print(f"✗ Unexpected Error: {e}")
        print()
        return 1


def apply_seeds(seeds_dir: str, runtime_dir: str) -> int:
    """
    Validate and apply seed files.

    Args:
        seeds_dir: Path to seeds directory
        runtime_dir: Path to runtime directory

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("Domain Bank Seed Application")

    print(f"Seeds directory:   {seeds_dir}")
    print(f"Runtime directory: {runtime_dir}")

    try:
        # First validate
        print_section("Phase 1: Validation")
        val_report = run_seeding(seeds_dir, runtime_dir, validate_only=True)

        if not val_report['ok']:
            print("✗ Validation failed! Not applying seeds.")
            print()
            return 1

        print(f"✓ Validated {val_report['total_entries']} entries across "
              f"{len(val_report['banks'])} banks")

        # Then apply
        print_section("Phase 2: Application")
        app_report = run_seeding(seeds_dir, runtime_dir, validate_only=False)

        print_report_summary(app_report)

        if app_report['ok']:
            print("✓ Seeds applied successfully!")
            print()
            return 0
        else:
            print("✗ Application failed!")
            print()
            return 1

    except ValidationError as e:
        print()
        print(f"✗ Validation Error: {e}")
        print()
        return 1
    except Exception as e:
        print()
        print(f"✗ Unexpected Error: {e}")
        print()
        return 1


def verify_idempotency(seeds_dir: str, runtime_dir: str) -> int:
    """
    Verify that seeding is idempotent.

    Args:
        seeds_dir: Path to seeds directory
        runtime_dir: Path to runtime directory

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("Domain Bank Seed Idempotency Verification")

    print(f"Seeds directory:   {seeds_dir}")
    print(f"Runtime directory: {runtime_dir}")

    try:
        engine = SeedingEngine(seeds_dir, runtime_dir)

        print_section("Running seeding twice...")
        result = engine.verify_idempotency()

        print()
        print(f"Idempotent: {result['idempotent']}")
        print(f"Runs:       {result['runs']}")

        if result['differences']:
            print()
            print("Differences found in banks:")
            for bank in result['differences']:
                print(f"  ! {bank}")

        print()

        if result['ok']:
            print("✓ Idempotency verified!")
            print()
            return 0
        else:
            print("✗ Idempotency check failed!")
            print()
            return 1

    except ValidationError as e:
        print()
        print(f"✗ Validation Error: {e}")
        print()
        return 1
    except Exception as e:
        print()
        print(f"✗ Unexpected Error: {e}")
        print()
        return 1


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Domain Bank Seeding Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate    # Validate seeds without applying
  %(prog)s --apply       # Validate and apply seeds
  %(prog)s --verify      # Verify idempotency
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate seed files without applying'
    )
    mode_group.add_argument(
        '--apply',
        action='store_true',
        help='Validate and apply seed files'
    )
    mode_group.add_argument(
        '--verify',
        action='store_true',
        help='Verify idempotency by applying twice'
    )

    # Optional path overrides
    parser.add_argument(
        '--seeds-dir',
        type=str,
        default=None,
        help='Path to seeds directory (default: script directory)'
    )
    parser.add_argument(
        '--runtime-dir',
        type=str,
        default=None,
        help='Path to runtime domain banks directory'
    )

    args = parser.parse_args()

    # Determine paths
    if args.seeds_dir:
        seeds_dir = args.seeds_dir
    else:
        seeds_dir = str(Path(__file__).parent)

    if args.runtime_dir:
        runtime_dir = args.runtime_dir
    else:
        # Default to standard runtime location inside maven2_fix
        runtime_dir = str(
            validate_path_confinement(
                get_runtime_domain_banks_root(), "seed domain banks runtime"
            )
        )

    # Execute mode
    if args.validate:
        return validate_seeds(seeds_dir, runtime_dir)
    elif args.apply:
        return apply_seeds(seeds_dir, runtime_dir)
    elif args.verify:
        return verify_idempotency(seeds_dir, runtime_dir)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
