FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE $PORT
CMD streamlit run app.py --server.port $PORT --server.addresspip install -r requirements.txt
