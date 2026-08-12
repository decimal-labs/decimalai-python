.PHONY: setup demo clean

setup:
	pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "⚠️  Edit .env with your DECIMAL_API_KEY"; fi

demo: setup
	@echo "═══════════════════════════════════════════"
	@echo "  SupportBot Demo — Full DecimalAI Workflow"
	@echo "═══════════════════════════════════════════"
	@echo ""
	@echo "Step 1/4: Generating traces with Agent v1..."
	python scenarios/01_generate_traces.py
	@echo ""
	@echo "Step 2/4: Updating agent to v2..."
	python scenarios/02_update_agent.py
	@echo ""
	@echo "Step 3/4: Running evaluations..."
	python scenarios/03_run_evals.py
	@echo ""
	@echo "Step 4/4: Building dataset..."
	python scenarios/04_build_dataset.py
	@echo ""
	@echo "✅ Demo complete! Open https://app.decimal.ai to explore."

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
