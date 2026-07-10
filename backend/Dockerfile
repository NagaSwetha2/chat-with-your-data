FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_engine.py .
COPY backend/main.py ./backend/main.py
COPY backend/charts.py ./backend/charts.py
COPY backend/sf_knowledge.py ./backend/sf_knowledge.py
COPY backend/ml_engine.py ./backend/ml_engine.py
COPY backend/hypothesis_engine.py ./backend/hypothesis_engine.py
COPY backend/eda_engine.py ./backend/eda_engine.py
COPY backend/proactive_engine.py ./backend/proactive_engine.py
COPY backend/extractor.py ./backend/extractor.py
COPY backend/model_router.py ./backend/model_router.py
COPY backend/business_context_engine.py ./backend/business_context_engine.py
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
