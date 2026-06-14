FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
COPY app/ /app/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
