# 课程文档中文翻译实施计划

> **给执行型 agent 的要求：** 实施本计划时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项推进。步骤使用复选框 `- [ ]` 语法跟踪。

**目标：** 构建一套可重复执行的流程，从 lesson 的 `docs/en.md` 生成 `docs/zh.md`，校验中文文档结构，并产出一套经过检查的 Phase 0 中文样本，且不改动仓库现有的英文优先工具链。

**架构：** 新增一个 Python 命令行脚本 `scripts/translate_lessons.py`，负责发现 lesson 文档、在保留 Markdown 结构的前提下调用 OpenAI SDK 翻译正文片段，并在只读模式下校验 `zh.md`。使用聚焦的小型 `unittest` 测试覆盖工作流，再运行脚本生成 Phase 0 样本并验证输出结构。

**技术栈：** Python 3.10+、标准库 `argparse` / `pathlib` / `re` / `unittest`、`requirements.txt` 中已存在的 `openai`

---

## 文件结构

- 新建：`scripts/translate_lessons.py`
  - 负责 lesson 发现、范围过滤、Markdown 分段、OpenAI 翻译、写入模式与校验模式
- 新建：`tests/test_translate_lessons.py`
  - 回归测试：lesson 发现、结构保留、范围过滤、校验结果
- 新建：`phases/00-setup-and-tooling/*/docs/zh.md`
  - 使用新脚本生成的 Phase 0 中文样本
- 修改：`docs/superpowers/plans/2026-05-26-lesson-docs-chinese-translation.md`
  - 如果使用内联执行，可在此文件里勾选进度

### 任务 1：补齐翻译工作流测试

**文件：**
- 新建：`tests/test_translate_lessons.py`
- 测试：`tests/test_translate_lessons.py`

- [ ] **步骤 1：先写失败测试**

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.translate_lessons import (
    Finding,
    collect_lessons,
    count_fenced_code_blocks,
    find_validation_issues,
    parse_phase_filter,
    replace_fenced_code_blocks,
    restore_fenced_code_blocks,
)


class TranslateLessonsTests(unittest.TestCase):
    def test_parse_phase_filter_accepts_zero_padded_value(self) -> None:
        self.assertEqual(parse_phase_filter("00"), 0)
        self.assertEqual(parse_phase_filter("7"), 7)

    def test_replace_and_restore_fenced_code_blocks_round_trip(self) -> None:
        source = "# Title\n\nText.\n\n```python\nprint('hi')\n```\n\nMore text.\n"
        replaced, blocks = replace_fenced_code_blocks(source)
        self.assertNotIn("print('hi')", replaced)
        restored = restore_fenced_code_blocks(replaced, blocks)
        self.assertEqual(restored, source)
        self.assertEqual(count_fenced_code_blocks(restored), 1)

    def test_collect_lessons_respects_phase_and_missing_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson_a = root / "phases" / "00-setup-and-tooling" / "01-dev-environment" / "docs"
            lesson_b = root / "phases" / "01-math-foundations" / "01-linear-algebra-intuition" / "docs"
            lesson_a.mkdir(parents=True)
            lesson_b.mkdir(parents=True)
            (lesson_a / "en.md").write_text("# A\n\nEnglish text.\n", encoding="utf-8")
            (lesson_b / "en.md").write_text("# B\n\nEnglish text.\n", encoding="utf-8")
            (lesson_b / "zh.md").write_text("# B\n\n中文。\n", encoding="utf-8")

            phase_zero = collect_lessons(root, phase_filter=0, lesson_filter=None, missing_only=False)
            self.assertEqual([item.lesson_dir.name for item in phase_zero], ["01-dev-environment"])

            missing_only = collect_lessons(root, phase_filter=None, lesson_filter=None, missing_only=True)
            self.assertEqual([item.lesson_dir.name for item in missing_only], ["01-dev-environment"])

    def test_find_validation_issues_reports_missing_h1_empty_file_and_code_block_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson_docs = root / "phases" / "00-setup-and-tooling" / "01-dev-environment" / "docs"
            lesson_docs.mkdir(parents=True)
            (lesson_docs / "en.md").write_text("# Title\n\n```bash\necho hi\n```\n", encoding="utf-8")
            issues = find_validation_issues(root, phase_filter=0, lesson_filter=None)
            self.assertEqual([issue.kind for issue in issues], ["missing_zh"])

            (lesson_docs / "zh.md").write_text("", encoding="utf-8")
            issues = find_validation_issues(root, phase_filter=0, lesson_filter=None)
            self.assertEqual(
                [issue.kind for issue in issues],
                ["empty_zh", "missing_h1", "code_block_mismatch"],
            )

    def test_validation_finding_string_contains_relative_path(self) -> None:
        finding = Finding(
            kind="missing_zh",
            lesson="phases/00-setup-and-tooling/01-dev-environment",
            file="phases/00-setup-and-tooling/01-dev-environment/docs/zh.md",
            message="missing docs/zh.md",
        )
        rendered = str(finding)
        self.assertIn("missing_zh", rendered)
        self.assertIn("docs/zh.md", rendered)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试，确认它先失败**

运行：`python -m unittest tests.test_translate_lessons -v`
预期：失败，报错 `ModuleNotFoundError: No module named 'scripts.translate_lessons'`

- [ ] **步骤 3：创建 tests 包标记文件**

```python
# tests/__init__.py
```

用 `apply_patch` 新增一个空的 `tests/__init__.py`，保证 `python -m unittest` 能正确导入 `tests.test_translate_lessons`

- [ ] **步骤 4：再次运行测试，确认仍因实现缺失而失败**

运行：`python -m unittest tests.test_translate_lessons -v`
预期：失败，报错 `ModuleNotFoundError: No module named 'scripts.translate_lessons'`

- [ ] **步骤 5：提交**

```bash
git add tests/__init__.py tests/test_translate_lessons.py
git commit -m "test: add translate_lessons workflow coverage"
```

### 任务 2：实现翻译与校验 CLI

**文件：**
- 新建：`scripts/translate_lessons.py`
- 测试：`tests/test_translate_lessons.py`

- [ ] **步骤 1：先实现最小可用代码，让测试能导入核心 helper**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
PHASE_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*[a-z0-9]$")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
H1_RE = re.compile(r"^#\\s+\\S", re.MULTILINE)
TOKEN_RE = re.compile(r"@@CODE_BLOCK_(\\d+)@@")


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
    phases_dir = root / "phases"
    lessons: list[LessonDoc] = []
    lesson_filter_path = (root / lesson_filter).resolve() if lesson_filter else None
    for phase_dir in sorted(phases_dir.iterdir()):
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
```

- [ ] **步骤 2：运行测试，确认 helper 层全部通过**

运行：`python -m unittest tests.test_translate_lessons -v`
预期：当前测试全部通过

- [ ] **步骤 3：补全 OpenAI 翻译逻辑和 CLI 入口**

```python
PROMPT = """You are translating AI engineering course material from English to Simplified Chinese.

Rules:
- Preserve Markdown structure exactly.
- Keep fenced code blocks exactly as-is.
- Keep inline code, URLs, package names, API identifiers, model names, and paper titles in English.
- Translate explanatory prose, table labels, list items, and headings into natural Simplified Chinese.
- Do not add commentary.
"""


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for translation mode")
    return OpenAI(api_key=api_key)


def translate_markdown(text: str, client: OpenAI, model: str) -> str:
    masked, blocks = replace_fenced_code_blocks(text)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": masked},
        ],
    )
    translated = response.output_text
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
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for translation mode")
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
```

- [ ] **步骤 4：补一条 CLI 校验模式回归测试**

```python
    def test_main_check_mode_returns_non_zero_when_zh_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lesson_docs = root / "phases" / "00-setup-and-tooling" / "01-dev-environment" / "docs"
            lesson_docs.mkdir(parents=True)
            (lesson_docs / "en.md").write_text("# Title\n\nBody\n", encoding="utf-8")
            with mock.patch("scripts.translate_lessons.ROOT", root):
                from scripts.translate_lessons import main

                self.assertEqual(main(["--phase", "00", "--check"]), 1)
```

- [ ] **步骤 5：运行完整测试**

运行：`python -m unittest tests.test_translate_lessons -v`
预期：全部通过

- [ ] **步骤 6：提交**

```bash
git add scripts/translate_lessons.py tests/test_translate_lessons.py
git commit -m "feat: add lesson translation workflow"
```

### 任务 3：生成并校验 Phase 0 中文样本

**文件：**
- 新建：`phases/00-setup-and-tooling/01-dev-environment/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/02-git-and-collaboration/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/03-gpu-setup-and-cloud/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/04-apis-and-keys/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/05-jupyter-notebooks/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/06-python-environments/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/07-docker-for-ai/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/08-editor-setup/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/09-data-management/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/10-terminal-and-shell/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/11-linux-for-ai/docs/zh.md`
- 新建：`phases/00-setup-and-tooling/12-debugging-and-profiling/docs/zh.md`
- 测试：`scripts/translate_lessons.py`

- [ ] **步骤 1：执行 Phase 0 批量翻译**

运行：`python scripts/translate_lessons.py --phase 00 --model gpt-4.1-mini`
预期：输出 12 行 `WROTE phases/00-setup-and-tooling/.../docs/zh.md`，每个 Phase 0 lesson 一行

- [ ] **步骤 2：校验生成出的样本**

运行：`python scripts/translate_lessons.py --phase 00 --check`
预期：输出 `OK: docs/zh.md coverage and structure look good`

- [ ] **步骤 3：抽查一个翻译结果，确认结构保留**

运行：`python -c "from pathlib import Path; p = Path(r'phases/00-setup-and-tooling/01-dev-environment/docs/zh.md'); text = p.read_text(encoding='utf-8'); print(text.splitlines()[0]); print(text.count('```'))"`
预期：

```text
# 开发环境
6
```

- [ ] **步骤 4：查看工作区改动**

运行：`git status --short`
预期：看到 `phases/00-setup-and-tooling/*/docs/zh.md` 下新增的中文文档

- [ ] **步骤 5：提交**

```bash
git add phases/00-setup-and-tooling/*/docs/zh.md
git commit -m "feat: add phase 0 chinese lesson docs"
```

## 自检

### 规格覆盖

- 增量生成 `docs/zh.md`：由任务 2 和任务 3 覆盖
- 可重复执行的工作流：由任务 2 的 CLI 模式覆盖
- 翻译结构与覆盖率校验：由任务 1 测试与任务 2 `--check` 覆盖
- 先样本后全量：由任务 3 覆盖
- 第一阶段不改动英文主工具链：由所有任务的文件范围约束保证

### 占位符检查

- 没有残留 `TODO`、`TBD` 或“以后再补”式描述
- 每条命令都给了明确预期结果
- 每个代码编辑步骤都给了具体代码或明确文件内容

### 类型一致性

- `LessonDoc` 作为 lesson 记录类型，在发现、写入流程中统一使用
- `Finding` 作为校验结果类型，在测试和 CLI 输出中统一使用
- `parse_phase_filter`、`collect_lessons`、`find_validation_issues`、`main` 在各任务中命名一致
