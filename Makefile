
.PHONY: test
test:
	UV_CACHE_DIR=./.uv-cache uv run python -m unittest discover -s ./tests
