# Contributing

Please open issues; PRs may be closed and applied manually to keep a single-author history.

## Development flow

1. Create or update a local branch.
2. Install project in editable mode:
   - `python -m pip install -e .`
3. Run validation:
   - `h2t --help`
   - `PYTHONPATH=. pytest -q`
4. Keep commits small and focused (one concern per commit).
5. Use direct, descriptive commit messages.

## Reporting problems

When opening an issue, include:

- Command run
- Relevant config file or overrides
- `results/run.log` excerpt
- Environment details from `results/env.txt` if available
