FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY coursework/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt
COPY coursework /app/coursework
COPY lectures /app/lectures
ENV COURSE_ROOT=/app COURSE_STATE=/state HF_HOME=/models
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "coursework.api:app", "--host", "0.0.0.0", "--port", "8000"]
