"""Auto-generates API reference pages for every Python module under src/."""
from pathlib import Path
import mkdocs_gen_files

src = Path("src")
nav = mkdocs_gen_files.Nav()

for path in sorted(src.rglob("*.py")):
    if path.name.startswith("_") or "pycache" in str(path):
        continue

    # Skip files in directories that aren't proper Python packages
    rel = path.relative_to(src)
    current = src
    is_valid_package = True
    for part in rel.parent.parts:
        current = current / part
        if not (current / "__init__.py").exists():
            is_valid_package = False
            break
    if not is_valid_package:
        continue

    module_path = rel.with_suffix("")
    doc_path = rel.with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.parts)

    nav[parts] = str(doc_path)

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        ident = ".".join(parts)
        f.write(f"# `{ident}`\n\n::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
