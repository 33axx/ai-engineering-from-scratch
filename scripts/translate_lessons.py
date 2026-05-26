#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
PHASE_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
TOKEN_RE = re.compile(r"@@CODE_BLOCK_(\d+)@@")

PROMPT = """You are translating AI engineering course material from English to Simplified Chinese.

Rules:
- Preserve Markdown structure exactly.
- Keep fenced code blocks exactly as-is.
- Keep inline code, URLs, package names, API identifiers, model names, and paper titles in English.
- Translate explanatory prose, table labels, list items, and headings into natural Simplified Chinese.
- Do not add commentary.
"""


@dataclass(frozen=True)
class LessonDoc:
    lesson_dir: Path
    en_path: Path
    zh_path: Path


@dataclass(frozen=True)
class Finding:
    kind: str
    lesson: str
    file: str
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.file} - {self.message}"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_runtime_env() -> dict[str, str]:
    merged: dict[str, str] = {}
    for env_path in (ROOT / ".env", ROOT / "scripts" / ".env"):
        merged.update(load_env_file(env_path))
    merged.update({key: value for key, value in os.environ.items() if value})
    return merged


def parse_phase_filter(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = int(raw, 10)
    if value < 0:
        raise ValueError("phase must be >= 0")
    return value


def replace_fenced_code_blocks(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def repl(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"@@CODE_BLOCK_{len(blocks) - 1}@@"

    return FENCE_RE.sub(repl, text), blocks


def restore_fenced_code_blocks(text: str, blocks: list[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return blocks[int(match.group(1))]

    return TOKEN_RE.sub(repl, text)


def count_fenced_code_blocks(text: str) -> int:
    return len(FENCE_RE.findall(text))


def collect_lessons(
    root: Path,
    phase_filter: int | None,
    lesson_filter: str | None,
    missing_only: bool,
) -> list[LessonDoc]:
    lessons: list[LessonDoc] = []
    lesson_filter_path = (root / lesson_filter).resolve() if lesson_filter else None
    for phase_dir in sorted((root / "phases").iterdir()):
        if not phase_dir.is_dir() or not PHASE_DIR_RE.match(phase_dir.name):
            continue
        phase_num = int(phase_dir.name.split("-", 1)[0])
        if phase_filter is not None and phase_num != phase_filter:
            continue
        for lesson_dir in sorted(phase_dir.iterdir()):
            if not lesson_dir.is_dir() or not LESSON_DIR_RE.match(lesson_dir.name):
                continue
            if lesson_filter_path is not None and lesson_dir.resolve() != lesson_filter_path:
                continue
            en_path = lesson_dir / "docs" / "en.md"
            zh_path = lesson_dir / "docs" / "zh.md"
            if not en_path.is_file():
                continue
            if missing_only and zh_path.exists():
                continue
            lessons.append(LessonDoc(lesson_dir=lesson_dir, en_path=en_path, zh_path=zh_path))
    return lessons


def find_validation_issues(
    root: Path,
    phase_filter: int | None,
    lesson_filter: str | None,
) -> list[Finding]:
    issues: list[Finding] = []
    for lesson in collect_lessons(root, phase_filter, lesson_filter, missing_only=False):
        rel_lesson = lesson.lesson_dir.relative_to(root).as_posix()
        rel_zh = lesson.zh_path.relative_to(root).as_posix()
        if not lesson.zh_path.exists():
            issues.append(Finding("missing_zh", rel_lesson, rel_zh, "missing docs/zh.md"))
            continue

        zh_text = lesson.zh_path.read_text(encoding="utf-8")
        en_text = lesson.en_path.read_text(encoding="utf-8")
        if not zh_text.strip():
            issues.append(Finding("empty_zh", rel_lesson, rel_zh, "docs/zh.md is empty"))
        if not H1_RE.search(zh_text):
            issues.append(Finding("missing_h1", rel_lesson, rel_zh, "docs/zh.md missing top-level H1"))
        if count_fenced_code_blocks(en_text) != count_fenced_code_blocks(zh_text):
            issues.append(
                Finding(
                    "code_block_mismatch",
                    rel_lesson,
                    rel_zh,
                    "fenced code block count differs between docs/en.md and docs/zh.md",
                )
            )
    return issues


def build_client() -> "OpenAI":
    env = load_runtime_env()
    api_key = env.get("DEEPSEEK_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required for translation mode")
    base_url = (
        env.get("DEEPSEEK_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def translate_markdown(text: str, client: "OpenAI", model: str) -> str:
    masked, blocks = replace_fenced_code_blocks(text)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": masked},
        ],
    )
    translated = response.choices[0].message.content or ""
    return restore_fenced_code_blocks(translated, blocks)


def write_translations(root: Path, lessons: list[LessonDoc], model: str) -> int:
    client = build_client()
    written = 0
    for lesson in lessons:
        source = lesson.en_path.read_text(encoding="utf-8")
        translated = translate_markdown(source, client, model)
        lesson.zh_path.write_text(translated.rstrip() + "\n", encoding="utf-8")
        written += 1
        print(f"WROTE {lesson.zh_path.relative_to(root).as_posix()}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate lesson docs/en.md into docs/zh.md")
    parser.add_argument("--phase", help="Phase number, e.g. 00 or 7")
    parser.add_argument("--lesson", help="Lesson directory path relative to repo root")
    parser.add_argument("--missing-only", action="store_true", help="Only translate lessons without docs/zh.md")
    parser.add_argument("--check", action="store_true", help="Validate docs/zh.md coverage and structure")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for translation mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        phase_filter = parse_phase_filter(args.phase)
    except ValueError as exc:
        parser.error(str(exc))

    if args.check:
        issues = find_validation_issues(ROOT, phase_filter, args.lesson)
        if issues:
            for issue in issues:
                print(issue)
            return 1
        print("OK: docs/zh.md coverage and structure look good")
        return 0

    lessons = collect_lessons(ROOT, phase_filter, args.lesson, args.missing_only)
    if not lessons:
        print("No lessons matched the requested scope")
        return 0
    write_translations(ROOT, lessons, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
