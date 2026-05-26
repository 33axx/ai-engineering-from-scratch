# Lesson Docs Chinese Translation Design

## Goal

Add Chinese lesson documents to the curriculum without disrupting the existing
English-first tooling. Each lesson that already has `docs/en.md` should gain a
peer `docs/zh.md`, with English remaining the canonical source for current
site, catalog, and audit behavior.

## Scope

This design covers:

- Batch generation of `docs/zh.md` for lesson teaching documents under
  `phases/**/docs/en.md`
- A repeatable translation workflow that can be re-run for newly added lessons
- Validation for translated document structure and coverage
- A sample-first rollout that validates translation style before full-scale
  generation

This design does not cover:

- Replacing `docs/en.md` as the canonical source for existing scripts
- Changing the website to default to Chinese
- Translating non-lesson files such as phase README files, repo-level docs,
  quizzes, outputs, or code comments

## Current State

The repository currently treats `docs/en.md` as the lesson source of truth.
Examples:

- `scripts/audit_lessons.py` validates `docs/en.md`
- `scripts/build_catalog.py` reads lesson titles from `docs/en.md`
- Curriculum skills reference `docs/en.md` when generating quizzes or guidance

There is no established `docs/zh.md` convention in the repo today. That means
the safest first step is additive: introduce Chinese files without changing the
English-based build and audit paths.

## Requirements

### Functional Requirements

1. Every lesson with `docs/en.md` can receive a sibling `docs/zh.md`.
2. Translation output must preserve Markdown structure:
   - headings
   - lists
   - blockquotes
   - tables
   - fenced code blocks
   - links and image references
3. Code blocks, commands, API names, model names, paper titles, and URLs must
   remain untranslated unless they are part of explanatory prose outside code.
4. The translation workflow must support partial execution:
   - by phase
   - by lesson path
   - by missing-files-only mode
5. The workflow must support validation without writing files.
6. The workflow must be usable for an initial sample run on Phase 0 before
   full-repo rollout.

### Quality Requirements

1. Chinese should be readable teaching prose, not keyword substitution.
2. Technical terminology should prioritize correctness over literal wording.
3. Long English sentences may be split into natural Chinese sentences where
   helpful.
4. H1 titles and section headings must exist in translated files.
5. The number of fenced code blocks in `zh.md` should match `en.md`.

## Recommended Approach

Use a dedicated translation script that scans lesson docs, generates peer
`zh.md` files, and validates structure. Roll out in two stages:

1. Generate and review a Phase 0 sample.
2. After style approval, run the same workflow across the rest of the repo.

This balances speed and safety. It avoids manual one-off work while still
creating a checkpoint before hundreds of files are generated.

## Alternatives Considered

### Manual per-file translation

Pros:

- Highest per-document editorial control

Cons:

- Not scalable across the curriculum
- Hard to keep up with newly added lessons
- No reusable workflow for future maintenance

### Full-repo translation with no workflow or validation

Pros:

- Fastest initial output

Cons:

- High risk of malformed Markdown
- Hard to re-run safely
- Hard to audit missing or stale translations later

## Proposed File Changes

### New Files

- `scripts/translate_lessons.py`
  - Batch translation entry point for lesson docs
- `docs/superpowers/specs/2026-05-26-lesson-docs-chinese-translation-design.md`
  - This design document

### Potential Updates to Existing Files

- `scripts/audit_lessons.py`
  - Optional later enhancement to validate `docs/zh.md`
- `scripts/build_catalog.py`
  - Optional later enhancement to expose `has_zh_docs`
- `README.md`
  - Optional later note describing bilingual lesson availability

Only `scripts/translate_lessons.py` is required for the first implementation
phase. Existing build and audit scripts should remain English-canonical until
the translation layer is stable.

## Translation Workflow Design

### Inputs

- Source files: `phases/**/docs/en.md`
- Scope filters:
  - all lessons
  - specific phase
  - specific lesson directory
  - missing translations only

### Outputs

- Target files: `phases/**/docs/zh.md`

### CLI Shape

The implementation must support this command shape:

```bash
python scripts/translate_lessons.py --phase 00
python scripts/translate_lessons.py --missing-only
python scripts/translate_lessons.py --lesson phases/00-setup-and-tooling/01-dev-environment
python scripts/translate_lessons.py --check
```

### Translation Rules

The script should preserve:

- fenced code blocks verbatim
- inline code spans verbatim where possible
- raw URLs verbatim
- Markdown table layout
- image and link targets

The script should translate:

- titles
- explanatory prose
- list items
- table labels and descriptions
- blockquote prose
- exercise text

The script should not attempt to rewrite:

- code
- shell commands
- package names
- framework names
- API identifiers

## Validation Design

The translation workflow should provide a validation mode that reports:

- missing `docs/zh.md`
- missing H1 in `docs/zh.md`
- code block count mismatch between `en.md` and `zh.md`
- obvious unreadable output such as empty files

Validation should be read-only and return a non-zero exit code when problems
are found.

## Rollout Plan

### Stage 1: Sample

- Implement the translation script and validation mode.
- Generate `docs/zh.md` only for Phase 0.
- Review the Chinese output for tone, terminology, and structure.
- Adjust translation rules if the sample reveals weak spots.

### Stage 2: Full Translation

- Run translation for all remaining lesson docs.
- Run validation across the full curriculum.
- Spot-check multiple phases for consistency.

### Stage 3: Optional Follow-Up

- Extend catalog metadata to note Chinese availability.
- Extend audit tooling to treat `zh.md` as a supported translation layer.
- Add website language switching if desired later.

## Risks and Mitigations

### Risk: Low-quality or awkward translation

Mitigation:

- Sample-first rollout on Phase 0
- Preserve structure so manual cleanup remains easy
- Keep English canonical during the first pass

### Risk: Breaking existing repo tooling

Mitigation:

- Do not change existing consumers of `docs/en.md` in phase one
- Add Chinese files additively

### Risk: Future lessons fall out of sync

Mitigation:

- Make translation re-runnable
- Add `--missing-only` and `--check` modes

## Testing Strategy

Implementation should be verified with:

1. A targeted run against a single lesson
2. A Phase 0 batch run
3. Validation over the generated Phase 0 translations
4. A full-curriculum missing-file check before and after full rollout

Automated verification should focus on:

- lesson discovery
- scope filtering
- Markdown block preservation
- validation error detection

## Success Criteria

This project is successful when:

- each lesson can have a valid `docs/zh.md`
- the translation workflow can be re-run safely
- Phase 0 is translated and reviewed successfully
- full-curriculum translation can be executed without changing English-based
  tooling

## Implementation Boundary

The first implementation cycle should stop after:

- adding the translation script
- generating sample Chinese docs for Phase 0
- validating the sample

Full-curriculum generation can proceed immediately after sample approval using
the same workflow, but it should remain a distinct execution step so the user
can inspect the sample quality first.
