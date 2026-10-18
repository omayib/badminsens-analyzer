FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["streamlit", "run", "app.py"]
