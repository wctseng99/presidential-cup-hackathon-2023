PY = python
PYTHON = PYTHONPATH=. $(PY)

test:
	@$(PYTHON) -m pytest
