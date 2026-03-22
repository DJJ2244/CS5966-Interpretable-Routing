"""
languages.py - Per-language Docker image and code assembly config.

setup_cmd : runs once when the container starts (install deps, etc.)
run_cmd   : runs per test case
"""


def _default(prompt, completion, test, entry_point):
    return prompt + completion + "\n" + test

def _python(prompt, completion, test, entry_point):
    return prompt + completion + "\n" + test + f"\ncheck({entry_point})"

def _java(prompt, completion, test, entry_point):
    return prompt + completion + "\n}\n" + test

def _scala(prompt, completion, test, entry_point):
    return (
        prompt + completion
        + "\nobject Main {\n  def main(args: Array[String]): Unit = {\n"
        + test + "\n  }\n}"
    )

def _php(prompt, completion, test, entry_point):
    return "<?php\n" + prompt + completion + "\n" + test

def _csharp(prompt, completion, test, entry_point):
    return (
        "using System;\nusing System.Collections.Generic;\nusing KellermanSoftware.CompareNetObjects;\n"
        + prompt + completion + "\n    }\n" + test + "\n    }\n}"
    )


LANGUAGES: dict[str, dict] = {
    "python": {
        "image":     "python:3.11-slim",
        "filename":  "solution.py",
        "setup_cmd": None,
        "run_cmd":   "python solution.py",
        "assemble":  _python,
    },
    "java": {
        "image":     "openjdk:21-slim",
        "filename":  "Main.java",
        "setup_cmd": None,
        "run_cmd":   "javac Main.java && java Main",
        "assemble":  _java,
    },
    "javascript": {
        "image":     "node:20-slim",
        "filename":  "solution.js",
        "setup_cmd": "npm install --silent lodash",
        "run_cmd":   "node solution.js",
        "assemble":  _default,
    },
    "typescript": {
        "image":     "node:20-slim",
        "filename":  "solution.ts",
        "setup_cmd": "npm install --silent ts-node typescript @types/node",
        "run_cmd":   "npx ts-node solution.ts",
        "assemble":  _default,
    },
    "go": {
        "image":     "golang:1.21-alpine",
        "filename":  "solution.go",
        "setup_cmd": None,
        "run_cmd":   "go run solution.go",
        "assemble":  _default,
    },
    "kotlin": {
        "image":     "zenika/kotlin",
        "filename":  "solution.kt",
        "setup_cmd": None,
        "run_cmd":   "kotlinc solution.kt -include-runtime -d solution.jar && java -jar solution.jar",
        "assemble":  _default,
    },
    "scala": {
        "image":     "sbtscala/scala-sbt:eclipse-temurin-21.0.2_13_1.9.9_3.4.0",
        "filename":  "solution.scala",
        "setup_cmd": None,
        "run_cmd":   "scala solution.scala",
        "assemble":  _scala,
    },
    "ruby": {
        "image":     "ruby:3.2-slim",
        "filename":  "solution.rb",
        "setup_cmd": None,
        "run_cmd":   "ruby solution.rb",
        "assemble":  _default,
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
        "assemble":  _default,
    },
    "perl": {
        "image":     "perl:5.38-slim",
        "filename":  "solution.pl",
        "setup_cmd": "cpanm --quiet Data::Compare",
        "run_cmd":   "perl solution.pl",
        "assemble":  _default,
    },
    "csharp": {
        "image":     "mcr.microsoft.com/dotnet/sdk:8.0",
        "filename":  "solution.cs",
        "setup_cmd": None,
        "run_cmd":   "dotnet script solution.cs",
        "assemble":  _csharp,
    },
}
