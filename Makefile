# ParkSentinel — Makefile
# Usage: make <target>

.PHONY: setup clean run lint notebook

setup:
	@echo "Setting up environment..."
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Environment ready. Activate with: source venv/bin/activate"

run:
	@echo "Launching Streamlit dashboard..."
	streamlit run app/streamlit_app.py

notebook:
	@echo "Starting Jupyter..."
	jupyter notebook notebooks/

lint:
	@echo "Running black formatter..."
	black src/ app/ notebooks/

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean."

help:
	@echo "Available targets:"
	@echo "  make setup    - Create venv and install dependencies"
	@echo "  make run      - Launch Streamlit dashboard"
	@echo "  make notebook - Start Jupyter notebook server"
	@echo "  make lint     - Format code with black"
	@echo "  make clean    - Remove cache and temp files"
