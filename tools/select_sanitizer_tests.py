#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental sanitizer test selector for HCF-Private/hpc-ops (symbol-trace).

Pipeline (per Makefile `sanitizer-incremental`):
  1. Resolve fork-point against origin/master and compute the accumulated
     changed-file set D. D is the union of four sources:
        - committed: BASE..HEAD
        - staged:    git diff --cached
        - unstaged:  git diff (worktree vs index)
        - untracked: git ls-files --others --exclude-standard
     Deletions are excluded (cannot symbol-trace a gone file). On any git
     failure -> full fallback.
  2. If D hits the (narrow) GLOBAL_IMPACT list -> full fallback.
     GLOBAL_IMPACT covers: build system (CMakeLists.txt),
     test infra (tests/utils.py), shared C/C++
     (src/C|utils|communicator), and vendored deps (3rd/**).
     setup.py is handled with fine-grained diff analysis: only
     substantive changes (not parallelism/comments) trigger fallback.
     Makefile / .ci/** / tools/** are still considered no-impact.
  3. Classify D into (top_src_dirs, direct_tests, touched_hpc_modules).
  4. For each A in top_src_dirs, run a 4-layer C++ symbol trace:
        U_A  = async launcher decls in A/**.h (names ending in `_async`)
        U_A* = closure expansion of U_A inside A/**
               over {.h, .hpp, .cuh, .inl, .cu}  (explicitly NOT .cc)
        E_A  = (op_name, entry_symbol) pairs from A/**.cc
               via `m.def("<op>", &ns::entry)`              (form alpha)
               or  `m.def("<op>(schema) -> ...")` + `m.impl(...)` (form beta)
        O_A  = ops whose entry_symbol body (in A/**.cc or A/**.cu) calls
               at least one name in U_A*
               (fail-open: unresolvable entries are conservatively added)
     O = union of O_A.
  5. Build INV: op_name -> {hpc function names using `torch.ops.hpc.<op>`}
     via AST over hpc/*.py (excluding __init__.py). This is the only
     C++<->Python bridge; OVERRIDE_SRC_TO_HPC has been removed.
  6. P (impacted hpc function names) =
        (union of INV[op] for op in O)                       # C++ side
        + (all top-level funcs of each touched hpc/<mod>.py) # Python side
  7. T = direct test changes
        + tests where used_funcs(test) intersects P
        + incomplete tests when P is non-empty (conservative)
  8. stdout: sorted T, one test path per line. stderr: all logs.

Constraints:
  - Standard library only (ast/re/subprocess/pathlib/logging/argparse/os/sys).
  - All paths resolved from __file__, never CWD.
  - Logging (with timestamp + level) goes to stderr so `$(shell ...)` stays
    safe.

Usage:
    python3 tools/select_sanitizer_tests.py
    python3 tools/select_sanitizer_tests.py --verbose
    SANITIZER_FULL=1 python3 tools/select_sanitizer_tests.py
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants and paths (resolved from __file__, never CWD)
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
HPC_DIR: Path = REPO_ROOT / "hpc"
SRC_DIR: Path = REPO_ROOT / "src"
TESTS_DIR: Path = REPO_ROOT / "tests"

REMOTE_MASTER_DEFAULT: str = "origin/master"

# Source-file extensions scanned during U_A* closure expansion.
# `.cc` is deliberately excluded: it is owned by the torch-registration step.
CXX_SRC_EXTS_FOR_CLOSURE: Tuple[str, ...] = (
    ".h", ".hpp", ".cuh", ".inl", ".cu",
)

# Patterns whose modification forces a full-sanitizer fallback. Kept verbatim
# against the previous implementation. `.*` = fnmatch `**`, `[^/]*` = fnmatch
# `*`, literal dots escaped.
# NOTE: setup.py is handled separately with fine-grained diff analysis
#       (see _setup_py_has_real_impact) to avoid false full-fallback when
#       only build parallelism or other non-product settings are changed.
GLOBAL_IMPACT_REGEX: List[Tuple[str, re.Pattern]] = [
    (label, re.compile(pat)) for label, pat in [
        ("CMakeLists.txt",       r"^CMakeLists\.txt$"),
        ("tests/utils.py",       r"^tests/utils\.py$"),
        ("src/C/**",             r"^src/C/.*$"),
        ("src/utils/**",         r"^src/utils/.*$"),
        ("3rd/**",               r"^3rd/.*$"),
    ]
]

# Files that require fine-grained diff analysis before triggering full
# fallback. Each entry maps to a checker function that returns True if the
# diff has real (product-affecting) impact.
CONDITIONAL_IMPACT_FILES: Set[str] = {"setup.py"}

# Regex patterns matching setup.py diff lines that are considered harmless
# (do not affect the build product). A changed line (after stripping the
# leading +/- and whitespace) matching ANY of these is safe to ignore.
# If ALL changed lines match, setup.py is deemed non-impactful.
_SETUP_PY_SAFE_PATTERNS: List[re.Pattern] = [
    # Parallelism-related: CMAKE_BUILD_PARALLEL_LEVEL / parallel / -j
    re.compile(r'.*CMAKE_BUILD_PARALLEL_LEVEL.*'),
    re.compile(r'.*parallel.*=.*os\.environ\.get\(.*CMAKE_BUILD_PARALLEL_LEVEL.*'),
    re.compile(r'.*f"-j\{parallel\}".*'),
    re.compile(r'.*-j\d+.*'),
    re.compile(r'^\s*parallel\s*='),
    # Pure comment lines
    re.compile(r'^\s*#'),
    # Blank lines
    re.compile(r'^\s*$'),
]

# Dirs traced at 2nd level: a change under `src/<top>/<sub>/...` maps to
# `src/<top>/<sub>` instead of `src/<top>`. Requires each subdir to be
# self-contained (its entry .cc only calls `*_async` declared in the same
# subdir), else ops are silently dropped. Files directly under `src/<top>/`
# keep 1st-level granularity.
SECOND_LEVEL_SRC_DIRS: Set[str] = {
    "src/attention",
}

# ---------------------------------------------------------------------------
# Logging (stderr only so stdout stays capture-safe)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("sanitizer-incremental")


# ---------------------------------------------------------------------------
# Git helpers (unchanged contract)
# ---------------------------------------------------------------------------

def run_git(args: List[str]) -> Tuple[int, str, str]:
    """Run git with REPO_ROOT as cwd. Returns (rc, stdout, stderr)."""
    proc = subprocess.run(
        ["git"] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _resolve_base(remote_master: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve BASE: try `merge-base --fork-point`, then plain `merge-base`."""
    rc, out, _ = run_git(["merge-base", "--fork-point", remote_master, "HEAD"])
    if rc == 0 and out:
        return out, "merge-base --fork-point"
    rc, out, _ = run_git(["merge-base", remote_master, "HEAD"])
    if rc == 0 and out:
        return out, "merge-base"
    return None, None


def _git_name_only(args: List[str], label: str) -> Optional[List[str]]:
    """Run a `git <args>` that produces a newline-separated file list.

    Returns the list of non-empty lines, or None on failure (signals the
    caller to FULL-fallback).
    """
    rc, out, err = run_git(args)
    if rc != 0:
        log.warning("git %s failed (%s) -> FULL fallback", label, err)
        return None
    return [line for line in out.splitlines() if line.strip()]


def resolve_diff_files() -> Optional[List[str]]:
    """
    Return the accumulated changed-file set D, covering four sources:
      1) committed: BASE..HEAD  (fork-point vs current commit)
      2) staged:    --cached     (index vs HEAD)
      3) unstaged:  (no ref)     (worktree vs index)
      4) untracked: ls-files --others --exclude-standard
    Deletions (filter=D) are excluded: symbol-trace cannot read a file that
    is gone. The result is deduplicated and sorted. Return None to signal
    that the caller must fall back to the full test list.
    """
    remote_master = (os.environ.get("HPC_DIFF_BASE_REF", "").strip()
                     or REMOTE_MASTER_DEFAULT)

    base, strategy = _resolve_base(remote_master)
    if not base:
        log.warning("cannot resolve fork-point against %s -> FULL fallback",
                    remote_master)
        return None

    log.info("diff base resolved: base=%s strategy=%s remote=%s",
             base, strategy, remote_master)

    committed = _git_name_only(
        ["diff", "--name-only", "--diff-filter=ACMRT", f"{base}..HEAD"],
        "diff BASE..HEAD",
    )
    if committed is None:
        return None

    staged = _git_name_only(
        ["diff", "--name-only", "--diff-filter=ACMRT", "--cached"],
        "diff --cached",
    )
    if staged is None:
        return None

    unstaged = _git_name_only(
        ["diff", "--name-only", "--diff-filter=ACMRT"],
        "diff (worktree)",
    )
    if unstaged is None:
        return None

    untracked = _git_name_only(
        ["ls-files", "--others", "--exclude-standard"],
        "ls-files --others",
    )
    if untracked is None:
        return None

    log.info("diff sources: committed=%d staged=%d unstaged=%d untracked=%d",
             len(committed), len(staged), len(unstaged), len(untracked))

    return sorted(set(committed) | set(staged) | set(unstaged) | set(untracked))


# ---------------------------------------------------------------------------
# Global-impact detection
# ---------------------------------------------------------------------------

def is_global_impact(path: str) -> Optional[str]:
    """Return the matched label if `path` hits the global-impact list."""
    for label, regex in GLOBAL_IMPACT_REGEX:
        if regex.match(path):
            return label
    return None


def _setup_py_has_real_impact() -> bool:
    """Check whether setup.py changes affect the build product.

    Runs `git diff` (combined staged + unstaged + committed) on setup.py and
    inspects the changed lines. If every added/removed line matches one of
    the safe patterns (parallelism settings, comments, blank lines), the
    change is deemed non-impactful and returns False.

    Returns True (impactful) on any git failure (fail-open / conservative).
    """
    # Get the full diff of setup.py relative to merge-base
    remote_master = (os.environ.get("HPC_DIFF_BASE_REF", "").strip()
                     or REMOTE_MASTER_DEFAULT)
    base, _ = _resolve_base(remote_master)

    # Collect changed lines from all diff sources
    diff_sources_args = []
    if base:
        diff_sources_args.append(["diff", f"{base}..HEAD", "--", "setup.py"])
    diff_sources_args.append(["diff", "--cached", "--", "setup.py"])
    diff_sources_args.append(["diff", "--", "setup.py"])

    changed_lines: List[str] = []
    for args in diff_sources_args:
        rc, out, _ = run_git(args)
        if rc != 0:
            log.warning("git diff for setup.py fine-grained check failed "
                        "-> conservatively treating as impactful")
            return True
        for line in out.splitlines():
            # Only consider actual changed lines (starting with + or -, excluding diff headers)
            if line.startswith(("+++", "---")):
                continue
            if line.startswith(("+", "-")):
                # Strip the leading +/- character
                content = line[1:]
                changed_lines.append(content)

    if not changed_lines:
        # No actual content lines changed (e.g. permission-only change) -> non-impactful
        log.info("setup.py: no content lines changed -> non-impactful")
        return False

    # Check whether every changed line matches a safe pattern
    for line in changed_lines:
        stripped = line.strip()
        # Blank lines are always safe
        if not stripped:
            continue
        safe = any(pat.match(stripped) for pat in _SETUP_PY_SAFE_PATTERNS)
        if not safe:
            log.info("setup.py: impactful line detected: %r", line)
            return True

    log.info("setup.py: all %d changed lines are non-impactful "
             "(parallelism/comments only)", len(changed_lines))
    return False


def check_conditional_impact_files(D: List[str]) -> Optional[str]:
    """Check files in D that need fine-grained analysis.

    Returns the label/path of the first file with real impact, or None if
    all conditional-impact files pass the fine-grained check.
    """
    for p in D:
        if p in CONDITIONAL_IMPACT_FILES:
            if p == "setup.py":
                if _setup_py_has_real_impact():
                    return "setup.py (substantive change)"
                else:
                    log.info("setup.py changed but only non-impactful "
                             "modifications (e.g. parallelism) -> skipping "
                             "full fallback")
    return None


# ---------------------------------------------------------------------------
# Change-set classifier: D -> (top_src_dirs, direct_tests, touched_hpc_mods)
# ---------------------------------------------------------------------------

def classify_changed_files(
    D: List[str],
) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """
    Partition D into:
      - top_src_dirs:        {'src/<A>', ...}    (trace units under src/;
                             1st-level by default, 2nd-level for dirs in
                             SECOND_LEVEL_SRC_DIRS, e.g. 'src/attention/mla')
      - direct_tests:        {'tests/test_*.py', ...}
      - touched_hpc_modules: {'<mod>', ...}       (hpc/<mod>.py, no __init__)
      - ignored:             list of paths recorded for INFO logging

    hpc/__init__.py is silently ignored (see module docstring).
    """
    top_src_dirs: Set[str] = set()
    direct_tests: Set[str] = set()
    touched_hpc_modules: Set[str] = set()
    ignored: List[str] = []

    for p in D:
        if p.startswith("src/"):
            rest = p[len("src/"):]
            parts = rest.split("/")
            top = parts[0]
            if not top:
                ignored.append(p)
                continue
            top_dir = f"src/{top}"
            # Whitelisted dirs: descend to 2nd level when the file is in a
            # subdir; files directly under src/<top>/ keep 1st-level.
            if top_dir in SECOND_LEVEL_SRC_DIRS and len(parts) >= 3 and parts[1]:
                top_src_dirs.add(f"{top_dir}/{parts[1]}")
            else:
                top_src_dirs.add(top_dir)
            continue

        if p.startswith("tests/") and Path(p).name.startswith("test_") \
                and p.endswith(".py"):
            direct_tests.add(p)
            continue

        if p.startswith("hpc/") and p.endswith(".py"):
            name = p[len("hpc/"):-len(".py")]
            # __init__.py is intentionally ignored: collection re-imports
            # `hpc` at test start, so breakage surfaces regardless. Nested
            # paths (hpc/foo/bar.py) do not exist in this project.
            if "/" in name or name == "__init__":
                ignored.append(p)
                continue
            touched_hpc_modules.add(name)
            continue

        ignored.append(p)

    return top_src_dirs, direct_tests, touched_hpc_modules, ignored


# ---------------------------------------------------------------------------
# Lightweight C++ source preprocessing (comment stripping only)
# ---------------------------------------------------------------------------

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _preprocess_cxx(text: str) -> str:
    """Strip /* ... */ and // ... comments. Best-effort, not string-aware.

    `#if 0` blocks are NOT stripped: at worst they leak a few extra async
    names into U*, which is fail-open and harmless under the conservative
    selection semantics.
    """
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    return text


def _read_text(path: Path) -> str:
    """Read a text file as utf-8 with replacement on errors.

    The callers always iterate files under REPO_ROOT that we just rglob'ed,
    so raising on actual I/O failure is the desired behaviour (makes disk
    problems loud instead of silently dropping them).
    """
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Function-body iteration: (fn_name, body_text) over a preprocessed source.
# ---------------------------------------------------------------------------

# Matches a C/C++ function-definition header: identifier followed by a
# parenthesised parameter list, then a `{`. The parameter list itself can
# span multiple lines and contain nested parentheses, so we use a manual
# depth scan rather than a single regex.
_IDENT = re.compile(r"[A-Za-z_]\w*")
# Keywords that can appear as `<kw> (` and would otherwise be mis-read as a
# function name by the scanner. Declarator/type keywords (`class`, `struct`,
# `using`, `extern`, `friend`, `typedef`, `template`, `namespace`, ...) never
# appear in that shape and don't need to be listed here.
_NON_FN_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "return",
    "sizeof", "alignof", "decltype", "typeid",
}


def iter_function_bodies(text: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (fn_name, body_text) for every C++ function definition found in
    `text`. Expects `text` to have been preprocessed (comments + `#if 0`
    stripped). The scan is conservative: the body includes all nested
    braces, so lambdas / inner classes / template instantiations are
    collapsed into the enclosing function (good enough for closure and
    entry tracing per requirements 3.4 and 4.3).

    Algorithm: walk the string; at every position, check if the current
    identifier looks like a function name, then skip whitespace, require
    `(`, balance parens, skip whitespace/ref-qualifiers/noexcept, require
    `{`, then balance braces to capture the body.
    """
    n = len(text)
    i = 0
    while i < n:
        m = _IDENT.match(text, i)
        if not m:
            i += 1
            continue
        name = m.group()
        end = m.end()

        if name in _NON_FN_KEYWORDS:
            i = end
            continue

        # Skip whitespace and require '('
        j = end
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "(":
            i = end
            continue

        # Balance parentheses
        depth = 0
        k = j
        while k < n:
            c = text[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        if depth != 0:
            i = end
            continue

        # Between the parameter-list `)` and the body `{` we may see CV/ref
        # qualifiers, `noexcept`, trailing-return `-> T`, attributes, etc.
        # Scan forward until we find `{` (definition) or `;` (declaration);
        # treat any `(` we encounter (e.g. `noexcept(...)`) as a nested
        # paren group that must be balanced. Anything else is simply
        # skipped -- being over-permissive here only risks over-collecting
        # function bodies, which is safe for our fail-open selector.
        p = k
        found_brace = False
        while p < n:
            c = text[p]
            if c == "{":
                found_brace = True
                break
            if c == ";":
                break
            if c == "(":
                d = 0
                while p < n:
                    cc = text[p]
                    if cc == "(":
                        d += 1
                    elif cc == ")":
                        d -= 1
                        if d == 0:
                            p += 1
                            break
                    p += 1
                continue
            p += 1

        if not found_brace:
            i = end
            continue

        # Balance braces for the body.
        body_start = p + 1
        depth = 1
        q = body_start
        while q < n and depth > 0:
            c = text[q]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            q += 1
        if depth != 0:
            i = end
            continue

        body = text[body_start:q - 1]
        yield name, body
        i = q


# ---------------------------------------------------------------------------
# Call-name extraction from a function body
# ---------------------------------------------------------------------------

# Match `<optional ns::...::>name(`: we grab the trailing identifier, which
# is the unqualified call name.
_CALL_RE = re.compile(r"(?:[A-Za-z_]\w*\s*::\s*)*([A-Za-z_]\w*)\s*\(")


def _calls_in_body(body: str) -> Set[str]:
    """Return the set of unqualified call names appearing in `body`.

    Namespace qualifiers like `hpc::gemm::` are stripped; only the final
    identifier is kept. Control-flow keywords are filtered out.
    """
    names: Set[str] = set()
    for m in _CALL_RE.finditer(body):
        name = m.group(1)
        if name in _NON_FN_KEYWORDS:
            continue
        names.add(name)
    return names


# ---------------------------------------------------------------------------
# Step 1: U_A -- async decls in A/**.h
# ---------------------------------------------------------------------------

# Match function declarations whose name ends in `_async` and are followed
# by `(`. We don't validate the return type; the preprocessor pass has
# already dropped comments and `#if 0` bodies, and requirement 2.1 accepts
# over-collection as a safe behaviour.
_ASYNC_DECL_RE = re.compile(r"\b([A-Za-z_]\w*_async)\s*\(")

def extract_async_decls_from_headers(a_dir: Path) -> Set[str]:
    """Scan A/**.h and return the set of `*_async` declared names.

    A name is kept even if it also has an inline definition; the closure
    step can recognise it either way.
    """
    U: Set[str] = set()
    for h in a_dir.rglob("*.h"):
        text = _preprocess_cxx(_read_text(h))
        for m in _ASYNC_DECL_RE.finditer(text):
            U.add(m.group(1))
    return U


# ---------------------------------------------------------------------------
# Step 2: U_A* -- closure expansion over {.h, .hpp, .cuh, .inl, .cu}
# ---------------------------------------------------------------------------

def expand_async_closure(a_dir: Path, U_A: Set[str]) -> Set[str]:
    """Compute U_A* by iterating until no new `*_async` function is added.

    For each file matching CXX_SRC_EXTS_FOR_CLOSURE under `a_dir`, collect
    every function body whose name ends in `_async`, along with the set of
    unqualified call names in that body. Then iterate:

        while changed:
            for each (fn, calls) with fn ending in `_async`:
                if fn not in U and calls intersects U:
                    U.add(fn)
    """
    U = set(U_A)
    async_defs: Dict[str, Set[str]] = {}

    for ext in CXX_SRC_EXTS_FOR_CLOSURE:
        for src in a_dir.rglob(f"*{ext}"):
            text = _preprocess_cxx(_read_text(src))
            for fn_name, body in iter_function_bodies(text):
                if not fn_name.endswith("_async"):
                    continue
                # Merge if the same symbol has multiple definitions across
                # files (e.g. template specialisations).
                async_defs.setdefault(fn_name, set()).update(
                    _calls_in_body(body),
                )

    # Fixed-point closure.
    while True:
        added = False
        for fn, calls in async_defs.items():
            if fn in U:
                continue
            if calls & U:
                U.add(fn)
                added = True
        if not added:
            break

    return U


# ---------------------------------------------------------------------------
# Step 3: torch-registration extraction from A/**.cc
# ---------------------------------------------------------------------------

# Form alpha (one-shot):
#   m.def("<op>", &<ns::...::entry>);
# Form beta step 1:
#   m.def("<op>(schema) -> ...");         (the schema contains a `(` or space)
# Form beta step 2:
#   m.impl("<op>", <dispatch-key>, &<ns::...::entry>);

_DEF_ALPHA_RE = re.compile(
    r"""m\.def\s*\(\s*            # m.def(
        "(?P<op>[A-Za-z_]\w*)"    # first argument is a bare op name
        \s*,\s*                   # ,
        &?\s*                     # optional &
        (?P<entry>(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)   # ns::...::name
        \s*(?:,[^)]*)?            # allow optional further args (e.g. tags)
        \s*\)\s*;                 # );
    """,
    re.VERBOSE | re.DOTALL,
)

_DEF_BETA_RE = re.compile(
    r"""m\.def\s*\(\s*
        "(?P<op>[A-Za-z_]\w*)    # op name...
        \s*\(                    # ...followed by a schema paren (beta marker)
    """,
    re.VERBOSE,
)

_IMPL_RE = re.compile(
    r"""m\.impl\s*\(\s*
        "(?P<op>[A-Za-z_]\w*)"
        \s*,\s*[^,]+,\s*          # dispatch key (kCUDA / DispatchKey::CUDA)
        &?\s*
        (?P<entry>(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)
        \s*\)\s*;
    """,
    re.VERBOSE | re.DOTALL,
)


def extract_torch_registrations(
    a_dir: Path,
) -> List[Tuple[str, Optional[str], Path]]:
    """Return a list of (op_name, entry_symbol_or_None, cc_path).

    Supports form alpha (`m.def("op", &entry);`) and form beta
    (`m.def("op(schema) -> ...");` + `m.impl("op", ..., &entry);`).
    When a beta `m.def` has no matching `m.impl` in the same A, we emit a
    WARNING and return entry_symbol=None so the caller can conservatively
    add the op to O_A (requirement 4.2).
    """
    results: List[Tuple[str, Optional[str], Path]] = []
    # Beta-only ops that still need an m.impl to pair up.
    beta_pending: Dict[str, Path] = {}
    # Collected m.impl entries, keyed by op.
    impl_map: Dict[str, Tuple[str, Path]] = {}

    for cc in a_dir.rglob("*.cc"):
        text = _preprocess_cxx(_read_text(cc))

        # Form alpha.
        for m in _DEF_ALPHA_RE.finditer(text):
            op = m.group("op")
            entry = m.group("entry").rsplit("::", 1)[-1]
            results.append((op, entry, cc))

        # Form beta step 1 (m.def with schema).
        for m in _DEF_BETA_RE.finditer(text):
            op = m.group("op")
            # If a matching alpha already fired for this op in this file it
            # would have emitted its own record; beta is only recorded when
            # the op is not already covered.
            beta_pending.setdefault(op, cc)

        # Form beta step 2 (m.impl).
        for m in _IMPL_RE.finditer(text):
            op = m.group("op")
            entry = m.group("entry").rsplit("::", 1)[-1]
            impl_map[op] = (entry, cc)

    # Pair beta m.def with m.impl.
    alpha_ops = {op for (op, _, _) in results}
    for op, def_path in beta_pending.items():
        if op in alpha_ops:
            # Already recorded via alpha; skip.
            continue
        if op in impl_map:
            entry, impl_path = impl_map[op]
            results.append((op, entry, impl_path))
        else:
            log.warning(
                "op '%s' has m.def in %s but no m.impl in %s -> conservatively "
                "marking impacted",
                op, def_path.relative_to(REPO_ROOT), a_dir.relative_to(REPO_ROOT),
            )
            results.append((op, None, def_path))

    return results


# ---------------------------------------------------------------------------
# Step 4: O_A -- scan entry bodies inside A/**.{cc,cu}
# ---------------------------------------------------------------------------

def _collect_entry_bodies(
    a_dir: Path,
) -> Dict[str, List[Set[str]]]:
    """Pre-compute entry-name -> [set of call-names, ...] for all .cc/.cu
    files under `a_dir`. A function name may have multiple definitions
    (e.g. overloads); we keep them as a list of per-body call sets so the
    caller takes the union on lookup.
    """
    out: Dict[str, List[Set[str]]] = {}
    files: List[Path] = list(a_dir.rglob("*.cc")) + list(a_dir.rglob("*.cu"))
    for f in files:
        text = _preprocess_cxx(_read_text(f))
        for fn_name, body in iter_function_bodies(text):
            out.setdefault(fn_name, []).append(_calls_in_body(body))
    return out


def resolve_entry_body_and_build_OA(
    a_dir: Path,
    U_A_star: Set[str],
    registrations: List[Tuple[str, Optional[str], Path]],
) -> Set[str]:
    """Compute O_A from the registrations list and U_A*."""
    O_A: Set[str] = set()
    if not registrations:
        return O_A

    entry_bodies = _collect_entry_bodies(a_dir)

    for op, entry, _where in registrations:
        if entry is None:
            # Already WARNING-ed in extract_torch_registrations.
            O_A.add(op)
            continue

        bodies = entry_bodies.get(entry)
        if not bodies:
            log.warning(
                "entry symbol '%s' for op '%s' not found in %s -> "
                "conservatively marking impacted",
                entry, op, a_dir.relative_to(REPO_ROOT),
            )
            O_A.add(op)
            continue

        all_calls: Set[str] = set()
        for s in bodies:
            all_calls |= s

        if all_calls & U_A_star:
            O_A.add(op)

    return O_A


# ---------------------------------------------------------------------------
# Step 5: hpc/*.py inverted index  INV: op_name -> {fn_name, ...}
# ---------------------------------------------------------------------------

def _extract_torch_ops_hpc_op(node: ast.AST) -> Optional[str]:
    """If `node` is `torch.ops.hpc.<op>` as an Attribute chain, return <op>.

    Matches only the exact 4-level chain Name('torch') . ops . hpc . <op>.
    """
    if not isinstance(node, ast.Attribute):
        return None
    op = node.attr
    n2 = node.value
    if not isinstance(n2, ast.Attribute) or n2.attr != "hpc":
        return None
    n3 = n2.value
    if not isinstance(n3, ast.Attribute) or n3.attr != "ops":
        return None
    n4 = n3.value
    if not isinstance(n4, ast.Name) or n4.id != "torch":
        return None
    return op


def build_hpc_op_inverted_index() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return:
        INV: op_name -> {hpc_py_func_name, ...}
        F_per_module: {module_name: {top_level_func_name, ...}}
    """
    INV: Dict[str, Set[str]] = {}
    F_per_module: Dict[str, Set[str]] = {}

    for py_path in sorted(HPC_DIR.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        module_name = py_path.stem
        tree = ast.parse(py_path.read_text(encoding="utf-8"),
                         filename=str(py_path))

        funcs: Set[str] = set()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            funcs.add(node.name)
            for sub in ast.walk(node):
                op = _extract_torch_ops_hpc_op(sub)
                if op is not None:
                    INV.setdefault(op, set()).add(node.name)
        F_per_module[module_name] = funcs

    return INV, F_per_module


def compute_P(
    O: Set[str],
    INV: Dict[str, Set[str]],
    touched_hpc_modules: Set[str],
    F_per_module: Dict[str, Set[str]],
) -> Set[str]:
    """Combine the C++-side and Python-side contributions into P."""
    P: Set[str] = set()

    # C++ side contribution.
    unresolved_ops: List[str] = []
    for op in sorted(O):
        fns = INV.get(op)
        if not fns:
            unresolved_ops.append(op)
            continue
        P.update(fns)
    if unresolved_ops:
        log.info(
            "ops with no hpc/*.py caller (likely framework-internal): %s",
            ", ".join(unresolved_ops),
        )

    # Python side contribution: any directly touched hpc/<mod>.py promotes
    # all its top-level function names (requirement 6.2).
    for mod in sorted(touched_hpc_modules):
        P.update(F_per_module.get(mod, set()))

    return P


# ---------------------------------------------------------------------------
# tests/test_*.py call-relation index (unchanged)
# ---------------------------------------------------------------------------

class _TestCallVisitor(ast.NodeVisitor):
    """
    Detect calls to hpc functions of these shapes:
        import hpc                    -> hpc.<name>(...)
        import hpc as X               -> X.<name>(...)
        from hpc import a [as A]      -> a(...) / A(...) (recorded as `a`)
        from hpc.<mod> import c [as C]-> c(...) / C(...) (recorded as `c`)
    Dynamic forms (`getattr(hpc, ...)`, `from hpc import *`) mark the
    test as incomplete so it can be included conservatively.
    """

    def __init__(self) -> None:
        self.hpc_module_aliases: Set[str] = {"hpc"}
        self.from_hpc_names: Dict[str, str] = {}  # local_name -> real_name
        self.called_funcs: Set[str] = set()
        self.incomplete: bool = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "hpc":
                self.hpc_module_aliases.add(alias.asname or "hpc")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod == "hpc" or mod.startswith("hpc."):
            for alias in node.names:
                if alias.name == "*":
                    self.incomplete = True
                    continue
                local = alias.asname or alias.name
                self.from_hpc_names[local] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Shape A: hpc.<name>(...) / X.<name>(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in self.hpc_module_aliases:
                self.called_funcs.add(func.attr)
        # Shape B: bare name(...)
        elif isinstance(func, ast.Name):
            local = func.id
            if local in self.from_hpc_names:
                self.called_funcs.add(self.from_hpc_names[local])
        self.generic_visit(node)


def build_test_call_index() -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Return U: {test_relpath: {func_name, ...}} and incomplete_tests."""
    U: Dict[str, Set[str]] = {}
    incomplete: Set[str] = set()

    for py_path in sorted(TESTS_DIR.glob("test_*.py")):
        rel = py_path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(py_path.read_text(encoding="utf-8"),
                         filename=str(py_path))

        v = _TestCallVisitor()
        v.visit(tree)
        U[rel] = v.called_funcs
        if v.incomplete:
            log.warning("test %s has `from hpc import *` -> incomplete", rel)
            incomplete.add(rel)

    return U, incomplete


# ---------------------------------------------------------------------------
# Final selector: direct_tests + (used ∩ P) + incomplete-when-P -> T
# ---------------------------------------------------------------------------

def select_affected_tests(
    direct_tests: Set[str],
    P: Set[str],
    U: Dict[str, Set[str]],
    incomplete_tests: Set[str],
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Return (sorted_T, reasons)."""
    T: Set[str] = set()
    reasons: Dict[str, Set[str]] = {}

    for t in direct_tests:
        T.add(t)
        reasons.setdefault(t, set()).add("<direct test change>")

    for test_rel, used in U.items():
        hits = used & P
        if hits:
            T.add(test_rel)
            reasons.setdefault(test_rel, set()).update(hits)
        elif test_rel in incomplete_tests and P:
            T.add(test_rel)
            reasons.setdefault(test_rel, set()).add("<incomplete:conservative>")

    return sorted(T), reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _all_tests() -> List[str]:
    # Exclude cutedsl tests (need nvidia-cutlass-dsl, absent on CI runner).
    return sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in TESTS_DIR.glob("test_*.py")
        if "cutedsl" not in p.name
    )


def _emit_full(reason: str) -> int:
    log.info("%s -> emitting full test list", reason)
    for t in _all_tests():
        print(t)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select sanitizer tests affected by the current branch's "
                    "diff (symbol-trace). Selected paths go to stdout; "
                    "logs go to stderr.",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logs on stderr.")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("REPO_ROOT=%s", REPO_ROOT)

    # Switch: SANITIZER_FULL=1 forces the full list.
    if os.environ.get("SANITIZER_FULL", "").strip() == "1":
        return _emit_full("SANITIZER_FULL=1")

    # Step 1: diff
    D = resolve_diff_files()
    if D is None:
        return _emit_full("diff resolution failed")

    log.info("changed files (%d):", len(D))
    for p in D:
        log.info("  - %s", p)
    if not D:
        log.info("no changed files; emitting empty list")
        return 0

    # Step 2: global-impact check
    hit_global = [(p, is_global_impact(p)) for p in D if is_global_impact(p)]
    if hit_global:
        log.warning("GLOBAL IMPACT files detected -> FULL sanitizer:")
        for p, label in hit_global:
            log.warning("  * %s (pattern: %s)", p, label)
        return _emit_full("global-impact hit")

    # Step 2b: conditional-impact check (fine-grained diff analysis)
    conditional_hit = check_conditional_impact_files(D)
    if conditional_hit:
        log.warning("CONDITIONAL IMPACT file has real changes: %s",
                    conditional_hit)
        return _emit_full(f"conditional-impact: {conditional_hit}")

    # Step 3: classify D
    top_src_dirs, direct_tests, touched_hpc_modules, ignored = \
        classify_changed_files(D)
    if ignored:
        log.info("ignored (outside src/ | hpc/*.py | tests/test_*.py): %d",
                 len(ignored))
        for p in ignored:
            log.info("    . %s", p)
    log.info("direct test changes: %d", len(direct_tests))
    for p in sorted(direct_tests):
        log.info("    ! %s", p)
    log.info("touched hpc modules: %s",
             sorted(touched_hpc_modules) or "<none>")
    log.info("src top-level dirs:  %s",
             sorted(top_src_dirs) or "<none>")

    # Step 4: per-A symbol trace  ->  O
    O: Set[str] = set()
    for a_rel in sorted(top_src_dirs):
        a_dir = REPO_ROOT / a_rel
        if not a_dir.is_dir():
            log.warning("src top-dir %s does not exist; skipped", a_rel)
            continue

        U_A = extract_async_decls_from_headers(a_dir)
        if not U_A:
            log.info("[%s] U_A = empty (no .h with *_async decls); "
                     "skip closure/entry for this dir", a_rel)
            continue

        U_A_star = expand_async_closure(a_dir, U_A)
        delta = U_A_star - U_A
        log.info("[%s] |U_A|=%d, |U_A*|=%d, delta=%d",
                 a_rel, len(U_A), len(U_A_star), len(delta))
        if delta:
            log.info("[%s] U_A* \\ U_A: %s", a_rel, sorted(delta))

        registrations = extract_torch_registrations(a_dir)
        if not registrations:
            log.info("[%s] no m.def/m.impl registrations found; "
                     "this dir contributes no ops", a_rel)
            continue

        O_A = resolve_entry_body_and_build_OA(a_dir, U_A_star, registrations)
        log.info("[%s] O_A (%d): %s", a_rel, len(O_A), sorted(O_A))

        O |= O_A

    log.info("global O (%d): %s", len(O), sorted(O))

    # Step 5 + 6: INV  and  P
    INV, F_per_module = build_hpc_op_inverted_index()
    log.info("hpc inverted index: %d ops mapped across %d hpc modules",
             len(INV), len(F_per_module))
    if args.verbose:
        for op in sorted(INV):
            log.debug("  INV[%s] = %s", op, sorted(INV[op]))

    P = compute_P(O, INV, touched_hpc_modules, F_per_module)
    log.info("|P| = %d", len(P))
    if args.verbose:
        log.debug("P = %s", sorted(P))

    # Step 7: tests
    U, incomplete_tests = build_test_call_index()
    log.info("scanned %d test files, %d incomplete",
             len(U), len(incomplete_tests))

    T, reasons = select_affected_tests(
        direct_tests, P, U, incomplete_tests,
    )
    # Exclude cutedsl tests (need nvidia-cutlass-dsl, absent on CI runner).
    dropped = sorted(t for t in T if "cutedsl" in os.path.basename(t))
    if dropped:
        T = {t for t in T if "cutedsl" not in os.path.basename(t)}
        for t in dropped:
            log.info("  - %s   [excluded: cutedsl not in CI]", t)
    log.info("|T| = %d", len(T))
    if not T:
        log.info("no affected tests; emitting empty list")
        return 0

    log.info("selected %d affected test(s):", len(T))
    for t in T:
        log.info("  + %s   [hit: %s]", t,
                 ", ".join(sorted(reasons.get(t, set()))) or "<unknown>")
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
