FROM python:3.11.13

ENV PYTHONDONTWRITEBYTECODE=1  \ 
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./
COPY pyproject.toml ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8501        

CMD ["streamlit", "run", "app.py", "--server.address=localhost", "--server.port=8501"]