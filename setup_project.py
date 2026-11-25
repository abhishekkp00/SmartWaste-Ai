"""
setup_project.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — One-shot project scaffold script.

Run this ONCE from the project root to create all required directories
and empty __init__.py files:

    python setup_project.py

Safe to re-run — existing files are never overwritten.
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Directories to create ────────────────────────────────────────────────────
DIRS = [
    # Dataset
    "dataset/raw",
    "dataset/processed/train",
    "dataset/processed/val",
    "dataset/processed/test",
    # ML pipeline
    "ml",
    # Saved model weights
    "models",
    # Training outputs
    "results/graphs",
    "results/confusion_matrix",
    "results/reports",
    # Jupyter notebooks
    "notebooks",
    # Backend
    "backend/app/routes",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/database",
    "backend/app/utils",
    # Frontend
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/hooks",
    "frontend/src/services",
    # Tests
    "tests",
]

# ── Python packages that need __init__.py ────────────────────────────────────
PYTHON_PACKAGES = [
    "ml",
    "backend",
    "backend/app",
    "backend/app/routes",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/database",
    "backend/app/utils",
    "tests",
]

# ── Placeholder .gitkeep for empty non-Python dirs ───────────────────────────
GITKEEP_DIRS = [
    "dataset/raw",
    "dataset/processed/train",
    "dataset/processed/val",
    "dataset/processed/test",
    "models",
    "results/graphs",
    "results/confusion_matrix",
    "results/reports",
    "notebooks",
]


def main() -> None:
    print("=" * 60)
    print(" SmartWaste AI — Project Scaffold Setup")
    print("=" * 60)

    # 1. Create all directories
    for rel_path in DIRS:
        full_path = PROJECT_ROOT / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR]  {rel_path}/")

    print()

    # 2. Create __init__.py for every Python package
    for rel_path in PYTHON_PACKAGES:
        init_file = PROJECT_ROOT / rel_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                f'"""\nSmartWaste AI — {rel_path.replace("/", ".")} package.\n"""\n'
            )
            print(f"  [PKG]  {rel_path}/__init__.py  (created)")
        else:
            print(f"  [PKG]  {rel_path}/__init__.py  (already exists)")

    print()

    # 3. Add .gitkeep to empty directories so git tracks them
    for rel_path in GITKEEP_DIRS:
        gitkeep = PROJECT_ROOT / rel_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"  [GIT]  {rel_path}/.gitkeep  (created)")

    print()
    print("=" * 60)
    print(" Scaffold complete.")
    print(f" Project root: {PROJECT_ROOT}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. pip install -r requirements-dev.txt")
    print("  3. python dataset/download_trashnet.py")
    print("  4. python dataset/verify_dataset.py")
    print()


if __name__ == "__main__":
    main()
