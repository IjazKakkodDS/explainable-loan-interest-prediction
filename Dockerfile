# Explainable Credit Pricing Intelligence System
# FastAPI serving image
#
# Build:
#   docker build -t explainable-credit-pricing .
#
# Run (API only):
#   docker run -p 8000:8000 explainable-credit-pricing
#
# The container starts the FastAPI serving boundary.
# This is not a production lending system. Not financial advice.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so layer is cached on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY src/ ./src/
COPY app/model.pkl ./app/model.pkl
COPY app/features_list.pkl ./app/features_list.pkl
COPY app/shap_explainer.pkl ./app/shap_explainer.pkl

# Copy evidence scripts (optional; used for benchmark runs inside container)
COPY scripts/ ./scripts/
COPY docs/ ./docs/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
