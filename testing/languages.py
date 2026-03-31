"""
languages.py - Per-language Docker image and code assembly config.

setup_cmd : runs once when the container starts (install deps, etc.)
run_cmd   : runs per test case
mem_limit : Docker memory cap (default 256m; increase for heavy runtimes)

All assemblers receive (prompt, inner_body, test, entry_point).

For brace languages the prompt ends with the opening '{', inner_body is the
function body without outer braces, and the assembler closes with '\\n}\\n'
before appending the test.

For Ruby/Perl, prompt="" signals standalone mode (model returned the full
function); prompt non-empty signals body-continuation mode.
"""


# ---------------------------------------------------------------------------
# Assembler functions
# ---------------------------------------------------------------------------

def _python(prompt, inner, test, entry_point):
    return prompt + inner + "\n" + test + f"\ncheck({entry_point})"


def _brace(prompt, inner, test, entry_point):
    """Generic brace-language assembler.
    prompt ends with opening '{'; we add inner body, close '}', then test."""
    if prompt:
        return prompt + inner + "\n}\n" + test
    # standalone (use_prompt=False): model returned a self-contained function
    return inner + "\n" + test


def _java(prompt, inner, test, entry_point):
    # prompt ends with method '{'; close method '}' then class '}' then test
    return prompt + inner + "\n    }\n}\n" + test


def _scala(prompt, inner, test, entry_point):
    # prompt is 'object Main extends App { ... def name(...) = {'
    # test already ends with '\n}\n' closing the object
    return prompt + inner + "\n    }\n" + test


def _ruby(prompt, inner, test, entry_point):
    if prompt:
        # body-continuation: prompt ends with 'def name(args)', no 'end' yet
        return prompt + "\n" + inner + "\nend\n" + test
    # standalone: model returned full 'def...end' block
    return inner + "\n" + test


def _perl(prompt, inner, test, entry_point):
    if prompt:
        # body-continuation: prompt ends with 'sub name\n{\n  my...;\n'
        return prompt + inner + "\n}\n" + test
    # standalone: model returned full 'sub name { ... }' block
    return inner + "\n" + test


def _php(prompt, inner, test, entry_point):
    # prompt already starts with '<?php' — do NOT prepend it again
    return prompt + inner + "\n}\n" + test


def _csharp(prompt, inner, test, entry_point):
    # prompt already has using directives + namespace + class + method signature
    # test has Main() + closes class + closes namespace
    return prompt + inner + "\n        }\n" + test


# ---------------------------------------------------------------------------
# Language configs
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, dict] = {
    "python": {
        "image":     "python:3.11-alpine",
        "filename":  "solution.py",
        "setup_cmd": None,
        "run_cmd":   "python solution.py",
        "assemble":  _python,
    },
    "java": {
        "image":     "eclipse-temurin:21-jdk-alpine",
        "filename":  "Main.java",
        "setup_cmd": None,
        "run_cmd":   "javac Main.java && java Main",
        "assemble":  _java,
    },
    "javascript": {
        "image":     "node:20-alpine",
        "filename":  "solution.js",
        "setup_cmd": "npm install --silent lodash",
        "run_cmd":   "node solution.js",
        "assemble":  _brace,
    },
    "typescript": {
        "image":     "node:20-alpine",
        "filename":  "solution.ts",
        "setup_cmd": "npm install --silent ts-node typescript @types/node @types/assert",
        "run_cmd":   "npx ts-node solution.ts",
        "assemble":  _brace,
    },
    "go": {
        "image":     "golang:1.21-alpine",
        "filename":  "solution.go",
        "setup_cmd": None,
        "run_cmd":   "go run solution.go",
        "assemble":  _brace,
    },
    "kotlin": {
        "image":     "eclipse-temurin:21-jdk-jammy",
        "filename":  "solution.kt",
        "setup_cmd": "apt-get update -qq && apt-get install -yqq kotlin",
        "run_cmd":   "kotlinc solution.kt -include-runtime -d solution.jar && java -jar solution.jar",
        "assemble":  _brace,
        "mem_limit": "1g",
    },
    "scala": {
        "image":     "eclipse-temurin:21-jdk-jammy",
        "filename":  "solution.scala",
        "setup_cmd": "apt-get update -qq && apt-get install -yqq scala",
        "run_cmd":   "scala solution.scala",
        "assemble":  _scala,
        "mem_limit": "1g",
    },
    "ruby": {
        "image":     "ruby:3.2-slim",
        "filename":  "solution.rb",
        "setup_cmd": None,
        "run_cmd":   "ruby solution.rb",
        "assemble":  _ruby,
    },
    "php": {
        "image":     "php:8.2-cli",
        "filename":  "solution.php",
        "setup_cmd": None,
        "run_cmd":   "php solution.php",
        "assemble":  _php,
    },
    "swift": {
        "image":     "swift:5.9",
        "filename":  "solution.swift",
        "setup_cmd": None,
        "run_cmd":   "swift solution.swift",
        "assemble":  _brace,
    },
    "perl": {
        "image":     "perl:5.38-slim",
        "filename":  "solution.pl",
        "setup_cmd": "cpanm --notest Data::Compare",
        "run_cmd":   "perl solution.pl",
        "assemble":  _perl,
    },
    "csharp": {
        "image":     "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
        "filename":  "solution.cs",
        "setup_cmd": (
            "dotnet new console -o /tmp/csproject -f net8.0 --force "
            "&& cd /tmp/csproject "
            "&& dotnet add package CompareNETObjects "
            "&& dotnet build -c Release -o /tmp/csproject/bin"
        ),
        "run_cmd":   "cp /tmp/solution.cs /tmp/csproject/Program.cs && dotnet run --project /tmp/csproject -c Release",
        "assemble":  _csharp,
    },
}
