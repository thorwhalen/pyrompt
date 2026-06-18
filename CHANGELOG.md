# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.0.5] - 2026-05-16

- test: add tests/test_smoke.py and exclude pyrompt source from doctest collection
- ci: migrate Modern -> uv (wads-migrate ci-to-uv)
- Refactor pyproject.toml for improved readability and consistency
- Incorporate changes from claude/implement-pyrompt-feature-011CV4f55gVnR9UdEppvejXQ
- License
- new CI and pyproject
- Implement feature with tests and demo notebook ([#2](https://github.com/thorwhalen/pyrompt/pull/2))
- dev docs
- dev docs
- 0.0.4:
- Initial commit

### Fixed

- fix(metadata): set real Homepage URL (PyPI rejects empty)
- fix(doctest): mark mall.PromptMall.search example as +SKIP
- fix(deps): add dol to core dependencies
