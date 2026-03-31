import ast
import re


def sanitize(completion: str, entry_point: str, language: str) -> tuple:
    """Return (code, use_prompt).

    use_prompt=True  → code is the function body inner content; the assembler
                        prepends the prompt (which ends with the opening '{' or
                        'def name(args)') and appends the closing token + test.
    use_prompt=False → code is a self-contained function definition; assembler
                        uses empty prompt and appends the test directly.
    """
    if language == "python":
        return _python_body(completion, entry_point), True
    if language in ("ruby", "perl"):
        return _keyword_body(completion, entry_point, language)
    return _brace_body(completion, entry_point), True


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _python_body(code: str, entry_point: str) -> str:
    """Return the function body with original indentation preserved so it
    stays valid inside the function signature supplied by the prompt."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == entry_point:
                lines = code.splitlines(keepends=True)
                body_lines = lines[node.body[0].lineno - 1 : node.end_lineno]
                return "".join(body_lines)
    except SyntaxError:
        pass
    return code  # fallback: return as-is


# ---------------------------------------------------------------------------
# Brace languages (Go, JS, TS, Java, Kotlin, Swift, Scala, PHP, C#)
# ---------------------------------------------------------------------------

def _brace_body(code: str, entry_point: str) -> str:
    """Extract the INNER content of the entry-point function (no outer braces).

    Three cases:
    1. Full function with signature detected → find balanced {…}, strip braces.
    2. Code starts with { (body-with-braces, no signature) → strip outer braces.
    3. Neither → already inner content, return as-is.

    Stripping outer braces means the caller's assembler can do:
        prompt_ending_with_{ + inner + "\\n}\\n" + test
    """
    # Build a pattern that catches various signature styles:
    #   - traditional:  entry_point(  or  entry_point<
    #   - JS/TS const:  const/let/var entry_point =
    #   - function kw:  function entry_point(
    ep = re.escape(entry_point)
    pattern = re.compile(
        rf'(?:'
        rf'function\s+{ep}\s*[(<]'
        rf'|(?:const|let|var)\s+{ep}\s*='
        rf'|\b{ep}\s*[(<]'
        rf')',
        re.MULTILINE,
    )

    match = pattern.search(code)
    if match:
        brace_pos = code.find('{', match.end())
        if brace_pos != -1:
            inner = _extract_inner(code, brace_pos)
            return inner

    # No signature found — check if the body itself is wrapped in { }
    stripped = code.lstrip()
    if stripped.startswith('{'):
        brace_pos = len(code) - len(stripped)
        inner = _extract_inner(code, brace_pos)
        return inner

    # Already inner content
    return code


def _extract_inner(code: str, brace_pos: int) -> str:
    """Walk forward from brace_pos counting braces; return the content
    BETWEEN the outer { and its matching }, preserving internal structure."""
    depth = 0
    for i, ch in enumerate(code[brace_pos:], start=brace_pos):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return code[brace_pos + 1 : i]  # strip outer { and }
    # Unbalanced — return everything after the opening brace
    return code[brace_pos + 1:]


# ---------------------------------------------------------------------------
# Keyword-delimited languages (Ruby uses def/end, Perl uses sub/{})
# ---------------------------------------------------------------------------

def _keyword_body(code: str, entry_point: str, language: str) -> tuple:
    """If the model returned the full function definition, signal use_prompt=False
    so the assembler passes an empty prompt and appends the test directly.
    Otherwise (body-only continuation) return use_prompt=True."""
    if language == "ruby":
        if re.search(rf'\bdef\s+{re.escape(entry_point)}\b', code):
            return code, False
    elif language == "perl":
        if re.search(rf'\bsub\s+{re.escape(entry_point)}\b', code):
            return code, False
    return code, True
