# Automates your development workflows

# ----------Environment Variables---------

# set environment variables to be used in the commands here

# ----------------Commands----------------

# change the 20 value in printf to adjust width
# Use ' ## some comment' behind a command and it will be added to the help message automatically
help: ## Shows this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

format: ## Formats code and sorts imports
	poetry run isort src
	poetry run black src notebooks

check: ## Runs all static checks (flake8, mypy, formatting checks)
	@echo "\n=== isort ======================="
	poetry run isort --check-only src
	@echo "\n=== black ======================="
	poetry run black --check src
	@echo "\n=== flake8 ======================"
	poetry run flake8 src --ignore=E203,E266,E501,W503 --max-line-length=88 --max-complexity=15 --select=B,C,E,F,W,T4,B9
	@echo "\n=== mypy ========================"
	poetry run mypy src notebooks


clean:
	-find . -type f -name "*.py[co]" -delete
	-find . -type d -name "__pycache__" -delete
	-find . -type d -name ".pytest_cache" -exec rm -r "{}" \;


test: ## Runs all tests with pytest
	@echo "\n=== pytest ========================"
	make clean
	poetry run pytest .



# --------------Configuration-------------

.NOTPARALLEL: ; # wait for this target to finish
.EXPORT_ALL_VARIABLES: ; # send all vars to shell

.PHONY: docs all # All targets are accessible for user
.DEFAULT: help # Running Make will run the help target

MAKEFLAGS += --no-print-directory # dont add message about entering and leaving the working directory
