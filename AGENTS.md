This Python project should only use `uv`, nothing else.

Always use a local `uv` cache dir to avoid sandbox restrictions.

For example, run tests like so:

```bash
UV_CACHE_DIR=./.uv-cache uv run python -m unittest discover -s tests
```
