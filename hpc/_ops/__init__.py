# Copyright (C) 2026 Tencent.

from pathlib import Path

_source_root = Path(__file__).resolve().parents[2] / "src"
if _source_root.is_dir():
    __path__.append(str(_source_root))
