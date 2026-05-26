import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
from uuid import uuid4

from scripts.translate_lessons import (
    Finding,
    build_parser,
    collect_lessons,
    count_fenced_code_blocks,
    find_validation_issues,
    load_env_file,
    parse_phase_filter,
    replace_fenced_code_blocks,
    restore_fenced_code_blocks,
)


@contextmanager
def workspace_tempdir() -> Path:
    root = Path.cwd() / ".tmp_test_translate_lessons" / uuid4().hex
    root.parent.mkdir(exist_ok=True)
    root.mkdir()
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


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
        with workspace_tempdir() as tmp:
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
        with workspace_tempdir() as tmp:
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

    def test_load_env_file_reads_key_value_pairs(self) -> None:
        with workspace_tempdir() as tmp:
            env_path = tmp / ".env"
            env_path.write_text(
                "DEEPSEEK_API_KEY=test-key\nDEEPSEEK_BASE_URL=https://api.deepseek.com\n",
                encoding="utf-8",
            )
            loaded = load_env_file(env_path)
            self.assertEqual(loaded["DEEPSEEK_API_KEY"], "test-key")
            self.assertEqual(loaded["DEEPSEEK_BASE_URL"], "https://api.deepseek.com")

    def test_build_parser_uses_deepseek_flash_as_default_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.model, "deepseek-v4-flash")

    def test_main_check_mode_returns_non_zero_when_zh_missing(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            lesson_docs = root / "phases" / "00-setup-and-tooling" / "01-dev-environment" / "docs"
            lesson_docs.mkdir(parents=True)
            (lesson_docs / "en.md").write_text("# Title\n\nBody\n", encoding="utf-8")
            with mock.patch("scripts.translate_lessons.ROOT", root):
                from scripts.translate_lessons import main

                self.assertEqual(main(["--phase", "00", "--check"]), 1)


if __name__ == "__main__":
    unittest.main()
