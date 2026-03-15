FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Railway will automatically assign a PORT environment variable, but we default to 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]