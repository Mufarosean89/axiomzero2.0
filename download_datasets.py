#!/usr/bin/env python3
"""
Axiom Zero - Dataset Downloader
Downloads external Lean proof datasets for cold-start pre-training.

Downloads four datasets:
  1. LeanDojo     — state→tactic traces from Mathlib proofs (Zenodo)
  2. miniF2F      — formal competition-math problems (GitHub)
  3. ProofNet     — undergraduate theorem statements (GitHub)
  4. Lean Workbook — large-scale (state, tactic) pairs (Hugging Face)

Usage:
    python download_datasets.py                          # Download all
    python download_datasets.py --dataset leandojo       # Single dataset
    python download_datasets.py --dataset minif2f,proofnet
    python download_datasets.py --output-dir ./my_data   # Custom output dir
    python download_datasets.py --list-sources           # Show URLs only
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Fix Windows console encoding for Unicode output ──────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT = Path(__file__).parent / "datasets"
DOWNLOAD_TIMEOUT = 120  # seconds

# Dataset definitions ----------------------------------------------------------
# Each entry:
#   name        : short display name
#   description : human-readable description
#   urls        : list of (url, dest_filename, is_archive)
#   expected    : list of files that should exist after download + extraction
#   post_process: optional function name in this module (called after extract)
#   archive_subdir: if set, GitHub archives create a top-level dir like {repo}-{branch}
#                   that we flatten by moving contents up one level

DATASET_DEFS: Dict[str, dict] = {
    "leandojo": {
        "name": "LeanDojo",
        "description": "State→tactic traces from human-written Mathlib proofs",
        "urls": [
            (
                "https://zenodo.org/records/8040109/files/leandojo_benchmark_v4.tar.gz",
                "leandojo_benchmark.tar.gz",
                True,
            ),
        ],
        "expected": [
            "leandojo_benchmark_v4",  # verified by recursive search of traj.json files
        ],
        "archive_subdir": "leandojo_benchmark_v4",
    },
    "minif2f": {
        "name": "miniF2F",
        "description": "Formal competition-math problems (Lean 4)",
        "urls": [
            (
                "https://github.com/google-deepmind/miniF2F/archive/refs/heads/main.zip",
                "minif2f.zip",
                True,
            ),
        ],
        "expected": [
            "minif2f_valid.json",
            "minif2f_test.json",
        ],
        "archive_subdir": "miniF2F-main",  # GitHub archive pattern
        "post_process": "_postprocess_minif2f",
    },
    "proofnet": {
        "name": "ProofNet",
        "description": "Undergraduate-course theorem statements with reference proofs (Lean 4)",
        "urls": [
            (
                "https://github.com/zhangir-azerbayev/ProofNet/archive/refs/heads/main.zip",
                "proofnet.zip",
                True,
            ),
        ],
        "expected": [
            "proofnet.jsonl",
        ],
        "archive_subdir": "ProofNet-main",
        "post_process": "_postprocess_proofnet",
    },
    "lean_workbook": {
        "name": "Lean Workbook",
        "description": "Large-scale corpus of Lean 4 proof (state, tactic) pairs",
        "urls": [
            (
                "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json",
                "lean_workbook.json",
                False,
            ),
        ],
        "expected": [
            "lean_workbook.json",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _report_progress(block_count: int, block_size: int, total_size: int):
    """Callback for urlretrieve to show download progress."""
    if total_size <= 0:
        return
    downloaded = block_count * block_size
    pct = min(100, int(downloaded * 100 / total_size))
    bar_len = 40
    filled = int(bar_len * downloaded / total_size)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r    [{bar}] {pct:3d}%  {downloaded // 1024:>8} KB / {total_size // 1024:>8} KB")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def download_url(url: str, dest: Path, desc: str = "") -> bool:
    """Download a URL to a local path with progress indication."""
    label = desc or os.path.basename(str(dest))
    print(f"  \u2b07  {label}")

    try:
        # Use urlretrieve with socket timeout
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)
        urllib.request.urlretrieve(url, str(dest), reporthook=_report_progress)
        if not dest.exists():
            print(f"     \u2717 File not created after download")
            return False
        file_size = dest.stat().st_size
        if file_size == 0:
            print(f"     \u2717 Downloaded file is empty")
            dest.unlink(missing_ok=True)
            return False
        size_str = f"{file_size // 1024 // 1024} MB" if file_size > 1024 * 1024 else f"{file_size // 1024} KB"
        print(f"     \u2713 {dest.name} ({size_str})")
        return True
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        print(f"     \u2717 Download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path, subdir: Optional[str] = None) -> bool:
    """Extract a tar.gz or zip archive and optionally flatten a top-level subdirectory."""
    print(f"  \ud83d\udce6 Extracting {archive_path.name}...")
    try:
        if str(archive_path).endswith((".tar.gz", ".tgz")):
            with tarfile.open(str(archive_path), "r:gz") as tar:
                tar.extractall(path=str(output_dir))
        elif str(archive_path).endswith((".zip",)):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(path=str(output_dir))
        else:
            print(f"     \u2717 Unknown archive format: {archive_path.name}")
            return False

        # Flatten GitHub archive subdirectory if present
        if subdir:
            subdir_path = output_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                # Move all contents up one level, then remove the subdir
                for item in list(subdir_path.iterdir()):
                    target = output_dir / item.name
                    if target.exists():
                        # Remove conflicting target before moving
                        if target.is_dir():
                            shutil.rmtree(str(target))
                        else:
                            target.unlink()
                    shutil.move(str(item), str(output_dir))
                shutil.rmtree(str(subdir_path))

        print(f"     \u2713 Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"     \u2717 Extraction error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Post-processors (convert raw data to formats our parsers expect)
# ═══════════════════════════════════════════════════════════════════════════


def _postprocess_minif2f(dataset_dir: Path) -> bool:
    """
    Convert miniF2F Lean files into JSON format expected by parse_minif2f().

    The google-deepmind/miniF2F repo uses two single-file splits:
      - MiniF2F/Valid.lean
      - MiniF2F/Test.lean

    Each .lean file contains many named theorems. This converter splits them
    into individual { id, informal, formal_statement, formal_proof } entries.
    """
    # Find MiniF2F/ directory (capital M)
    src_dir = dataset_dir / "MiniF2F"
    if not src_dir.exists():
        for p in dataset_dir.rglob("*.lean"):
            src_dir = p.parent
            break
        else:
            print("     \u26a0  Could not find MiniF2F .lean files. Skipping conversion.")
            return False

    print("  \ud83d\udd04 Converting miniF2F .lean files to JSON...")
    success = False

    split_map = {
        "Valid.lean": "valid",
        "Test.lean": "test",
        "valid.lean": "valid",
        "test.lean": "test",
    }

    for lean_name, split_name in split_map.items():
        lean_path = src_dir / lean_name
        if not lean_path.exists():
            continue

        content = lean_path.read_text(encoding="utf-8")
        entries = []
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Skip non-theorem declarations
            if any(stripped.startswith(kw) for kw in ("import", "namespace", "open", "set_option", "/-", "-/", "--")):
                i += 1
                continue

            # Detect theorem/lemma/example/def statement
            if any(stripped.startswith(kw) for kw in ("theorem", "lemma", "example", "def")):
                # Collect id from the first statement line
                problem_id = stripped.split()[1] if len(stripped.split()) > 1 else "unknown"
                problem_id = problem_id.split(".")[-1].strip(":")

                # Collect full statement including line-wrapped parts
                stmt_lines = [stripped]
                j = i + 1
                while j < len(lines):
                    lj = lines[j].strip()
                    if ":=" in lj or lj.startswith(":="):
                        stmt_lines.append(lj)
                        j += 1
                        break
                    stmt_lines.append(lj)
                    j += 1

                # Now collect the proof
                proof_lines = []
                brace = 0
                found_by = False
                while j < len(lines):
                    lj = lines[j].strip()

                    # Next theorem/lemma/example/def → stop
                    if any(lj.startswith(kw) for kw in ("theorem", "lemma", "example", "def")):
                        break

                    # Track brace depth for `by { ... }` blocks
                    brace += lj.count("{") - lj.count("}")

                    # `end` closes a `begin ... end` block
                    if lj == "end" or lj.startswith("end "):
                        if brace <= 0:
                            proof_lines.append(lj)
                            j += 1
                            break

                    if lj:
                        proof_lines.append(lj)

                    # If we passed through `by` at brace depth 0, done when brace returns to 0
                    if lj.startswith("by"):
                        found_by = True
                    if found_by and brace == 0 and lj == "sorry":
                        j += 1
                        break

                    j += 1

                i = j

                entries.append({
                    "id": problem_id,
                    "informal": "",
                    "formal_statement": " ".join(stmt_lines),
                    "formal_proof": "\n".join(proof_lines).strip(),
                })
                continue

            i += 1

        if entries:
            out_path = dataset_dir / f"minif2f_{split_name}.json"
            with open(str(out_path), "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            print(f"     \u2713 {out_path.name} ({len(entries)} theorems)")
            success = True
        else:
            print(f"     \u26a0  No theorems found in {lean_name}")

    if success:
        print("     \u2713 miniF2F conversion complete")
    else:
        print("     \u26a0  No miniF2F entries converted")
    return success


def _postprocess_proofnet(dataset_dir: Path) -> bool:
    """
    Convert ProofNet Lean files to JSONL format expected by parse_proofnet().

    The raw ProofNet repo organizes problems by textbook (Rudin.lean, Artin.lean, ...)
    inside benchmark/benchmark_to_publish/formal/. Each .lean file contains multiple
    named theorems. This extracts them into individual JSONL entries.
    """
    print("  \ud83d\udd04 Converting ProofNet .lean files to JSONL...")

    # Find formal/ directory
    formal_dir = dataset_dir / "benchmark" / "benchmark_to_publish" / "formal"
    if not formal_dir.exists():
        for p in dataset_dir.rglob("formal"):
            if p.is_dir() and list(p.glob("*.lean")):
                formal_dir = p
                break
        else:
            print("     \u26a0  Could not find formal/ directory. Skipping ProofNet conversion.")
            return False

    lean_files = sorted(formal_dir.glob("*.lean"))
    if not lean_files:
        print("     \u26a0  No .lean files found in formal/ directory.")
        return False

    print(f"     Found {len(lean_files)} textbook files")

    entries = []
    for lean_file in lean_files:
        category = lean_file.stem  # e.g., "Rudin", "Artin"
        content = lean_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        theorem_count = 0

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Skip blank lines, comments, namespace/import/open at top level
            if not stripped or stripped.startswith(("/-", "--", "import", "namespace", "open", "set_option")):
                i += 1
                continue

            # Detect theorem/lemma/example/def
            if any(stripped.startswith(kw) for kw in ("theorem", "lemma", "example", "def")):
                # Collect full statement including wrapped lines
                stmt_lines = [stripped]
                j = i + 1
                while j < len(lines):
                    lj = lines[j].strip()
                    if ":=" in lj:
                        stmt_lines.append(lj)
                        j += 1
                        break
                    if any(lj.startswith(kw) for kw in ("theorem", "lemma", "example", "def")):
                        break
                    # Don't include blank/comments in statement
                    if lj and not lj.startswith("--"):
                        stmt_lines.append(lj)
                    j += 1

                # Collect proof lines
                proof_lines = []
                brace = 0
                in_begin = False
                in_by = False
                while j < len(lines):
                    lj = lines[j].strip()

                    # Next theorem → stop
                    if any(lj.startswith(kw) for kw in ("theorem", "lemma", "example", "def")):
                        break

                    # Track `begin ... end` blocks
                    if lj == "begin":
                        in_begin = True
                        proof_lines.append(lj)
                        j += 1
                        continue

                    if lj == "end" or lj.startswith("end "):
                        if brace <= 0:
                            proof_lines.append(lj)
                            j += 1
                            break

                    # Track `by { ... }` blocks
                    brace += lj.count("{") - lj.count("}")
                    if lj.startswith("by"):
                        in_by = True

                    if lj and not lj.startswith("--"):
                        proof_lines.append(lj)

                    # `by` with no `{` → check next line to decide if proof continues
                    if in_by and brace == 0:
                        nxt = lines[j + 1].strip() if j + 1 < len(lines) else ""
                        if not (nxt and not nxt.startswith(("/-", "--")) and
                                not any(nxt.startswith(kw) for kw in ("theorem", "lemma", "example", "def"))):
                            break

                    j += 1

                problem_id = f"{category}_{theorem_count:04d}"
                entries.append({
                    "header": {
                        "problem_id": problem_id,
                        "source": f"ProofNet/{category}",
                        "category": category,
                    },
                    "formal_statement": " ".join(stmt_lines),
                    "formal_proof": "\n".join(proof_lines).strip(),
                })
                theorem_count += 1
                i = j
                continue

            i += 1

        print(f"       {category}: {theorem_count} theorems")

    if entries:
        out_path = dataset_dir / "proofnet.jsonl"
        with open(str(out_path), "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"     \u2713 {out_path.name} ({len(entries)} total entries)")
        return True

    print("     \u26a0  No ProofNet entries found")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Main download logic
# ═══════════════════════════════════════════════════════════════════════════


def download_dataset(
    key: str,
    output_dir: Path,
    force: bool = False,
    skip_extract: bool = False,
) -> bool:
    """Download and prepare a single dataset."""
    info = DATASET_DEFS.get(key)
    if info is None:
        print(f"  \u2717 Unknown dataset: {key}")
        return False

    print(f"\n{'='*60}")
    print(f"Dataset: {info['name']}")
    print(f"  {info['description']}")
    print(f"{'='*60}")

    ds_dir = output_dir / key
    ds_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True

    for url, filename, is_archive in info["urls"]:
        dest = ds_dir / filename

        if dest.exists() and not force:
            size = dest.stat().st_size
            size_str = f"{size // 1024 // 1024} MB" if size > 1024 * 1024 else f"{size // 1024} KB"
            print(f"  \u2713 Already cached: {filename} ({size_str})")
            continue

        # Download
        ok = download_url(url, dest, desc=info["name"])
        if not ok:
            all_ok = False
            continue

        # Extract if archive
        if is_archive and not skip_extract:
            subdir = info.get("archive_subdir")
            ok = extract_archive(dest, ds_dir, subdir=subdir)
            if not ok:
                all_ok = False

    # Run post-processor if defined and all downloads/extractions succeeded
    if all_ok:
        post_fn_name = info.get("post_process")
        if post_fn_name:
            post_fn: Optional[Callable[[Path], bool]] = globals().get(post_fn_name)
            if post_fn:
                post_fn(ds_dir)

    # Verify expected files exist, with lenient fallback
    for expected in info["expected"]:
        expected_path = ds_dir / expected
        if not expected_path.exists():
            # Search recursively as fallback
            matches = list(ds_dir.rglob(expected_path.name))
            if matches:
                continue
            # Check flattened (no subdirectory)
            alt = ds_dir / expected_path.name
            if alt.exists():
                continue
            print(f"  \u26a0  Expected file not verified: {expected}")
            all_ok = False
        else:
            size = expected_path.stat().st_size
            print(f"  \u2713 Verified: {expected} ({size // 1024} KB)")

    return all_ok


def list_sources():
    """Print all dataset sources without downloading."""
    print("Axiom Zero - Available Dataset Sources")
    print("=" * 60)
    for key, info in DATASET_DEFS.items():
        print(f"\n{info['name']} ({key})")
        print(f"  {info['description']}")
        for url, filename, _ in info["urls"]:
            print(f"  URL: {url}")
            print(f"  \u2192  {filename}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Download external Lean proof datasets for Axiom Zero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py                          # Download all datasets
  python download_datasets.py --dataset leandojo       # Just LeanDojo
  python download_datasets.py --dataset minif2f,proofnet --force
  python download_datasets.py --list-sources           # Show URLs only
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Comma-separated dataset keys to download (default: all). "
             f"Options: {', '.join(DATASET_DEFS.keys())}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction of archives (just download raw files)",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print dataset URLs and exit",
    )

    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        datasets_to_download = list(DATASET_DEFS.keys())
    else:
        datasets_to_download = [d.strip() for d in args.dataset.split(",")]

    print(f"Axiom Zero - Dataset Downloader")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Datasets: {', '.join(datasets_to_download)}")
    print()

    results: List[Tuple[str, bool]] = []
    for key in datasets_to_download:
        ok = download_dataset(
            key,
            output_dir,
            force=args.force,
            skip_extract=args.skip_extract,
        )
        results.append((key, ok))

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    success_count = 0
    for key, ok in results:
        status = "\u2713" if ok else "\u2717"
        info = DATASET_DEFS[key]
        print(f"  {status} {info['name']} ({key})")
        if ok:
            success_count += 1
    print(f"\n{success_count}/{len(results)} datasets downloaded successfully.")
    print(f"Output: {output_dir.resolve()}")

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
