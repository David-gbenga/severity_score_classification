FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY train.py predict.py ./
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
ENTRYPOINT ["python", "predict.py"]
