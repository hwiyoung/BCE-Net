.PHONY: setup verify env shell

setup:
	./scripts/setup_env.sh

verify:
	./scripts/run_in_env.sh python scripts/verify_env.py

env:
	@cat .venv-bcenet-geo/.bcenet-environment

shell:
	./scripts/run_in_env.sh
