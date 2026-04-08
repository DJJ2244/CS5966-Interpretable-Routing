import ast
import re
import textwrap


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

    # Fallback: normalize indentation for bare body snippets.
    # Strip top-level comment placeholders (e.g. "# Your code here") then
    # add 4-space indent to any lines that sit at column 0 while other lines
    # are already indented — a common model artifact.
    lines = code.splitlines(keepends=True)
    lines = [l for l in lines if not (not l.startswith(' ') and l.lstrip().startswith('#'))]
    # Truncate at any top-level `def` or `class` after the first non-blank
    # line — the model sometimes appends helper functions that would become
    # doubly-indented and cause SyntaxErrors.
    body_started = False
    truncated = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            body_started = True
        elif not body_started:
            truncated.append(line)
            continue
        if body_started and not line.startswith(' ') and re.match(r'(def|class)\s', line):
            break
        truncated.append(line)
    code = ''.join(truncated)
    dedented = textwrap.dedent(code)
    content_lines = [l for l in dedented.splitlines() if l.strip()]
    if content_lines:
        indents = [len(l) - len(l.lstrip()) for l in content_lines]
        if min(indents) == 0 and max(indents) > 0:
            code = textwrap.indent(dedented, '    ')
    return code


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

    # Already inner content — but may have a trailing unmatched } (and any
    # extra code after it, e.g. "func main()" or "console.log(...)") from the
    # model including its own closing brace.  Truncate at the first } that
    # would bring the depth below zero.
    depth = 0
    for i, ch in enumerate(code):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                return code[:i].rstrip()
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

# Ruby keywords that open a block requiring a matching `end`
_RUBY_BLOCK_OPEN = re.compile(
    r'^(if|unless|while|until|for|begin|case|def|class|module)\b'
)
_RUBY_BLOCK_DO = re.compile(r'\bdo\s*(\|[^|]*\|)?\s*$')
_RUBY_BLOCK_END = re.compile(r'^end\b')


def _ruby_extract_function(code: str, entry_point: str) -> str:
    """Return just the def…end block for entry_point, dropping trailing code."""
    match = re.search(rf'\bdef\s+{re.escape(entry_point)}\b', code)
    if not match:
        return code
    start = code.rfind('\n', 0, match.start()) + 1
    lines = code[start:].splitlines(keepends=True)
    depth = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if _RUBY_BLOCK_OPEN.match(s):
            depth += 1
        elif _RUBY_BLOCK_DO.search(s):
            depth += 1
        elif _RUBY_BLOCK_END.match(s):
            depth -= 1
            if depth == 0:
                return ''.join(lines[:i + 1]).rstrip()
    return ''.join(lines).rstrip()


def _ruby_strip_closing_end(code: str) -> str:
    """For body-continuation completions: strip the model's closing `end`
    (and any trailing code) so the assembler can add its own."""
    depth = 0
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        s = line.strip()
        if _RUBY_BLOCK_OPEN.match(s):
            depth += 1
        elif _RUBY_BLOCK_DO.search(s):
            depth += 1
        elif _RUBY_BLOCK_END.match(s):
            if depth == 0:
                return ''.join(lines[:i]).rstrip()
            depth -= 1
    return code.rstrip()


def _keyword_body(code: str, entry_point: str, language: str) -> tuple:
    """If the model returned the full function definition, signal use_prompt=False
    so the assembler passes an empty prompt and appends the test directly.
    Otherwise (body-only continuation) return use_prompt=True."""
    if language == "ruby":
        if re.search(rf'\bdef\s+{re.escape(entry_point)}\b', code):
            return _ruby_extract_function(code, entry_point), False
        # Body continuation — strip the model's closing `end` so assembler
        # can add its own, and drop any extra code that follows.
        return _ruby_strip_closing_end(code), True
    elif language == "perl":
        match = re.search(rf'\bsub\s+{re.escape(entry_point)}\b', code)
        if match:
            # Extract just the matching sub, dropping any extra subs after it
            # (extra subs may require uninstalled modules and cause failures).
            brace_pos = code.find('{', match.end())
            if brace_pos != -1:
                line_start = code.rfind('\n', 0, match.start()) + 1
                depth = 0
                for i in range(brace_pos, len(code)):
                    if code[i] == '{':
                        depth += 1
                    elif code[i] == '}':
                        depth -= 1
                        if depth == 0:
                            return code[line_start:i + 1], False
            return code[match.start():], False
        # Body continuation — truncate at the first unmatched `}` (the
        # model's closing brace) and drop any extra sub definitions after it.
        depth = 0
        for i, ch in enumerate(code):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth < 0:
                    return code[:i].rstrip(), True
    return code, True
