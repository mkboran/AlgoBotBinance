# Dockerfile

# Temel imaj olarak Python 3.10 slim kullanıyoruz
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY ./strategies /app/strategies/
COPY ./utils /app/utils/
COPY ./main.py /app/main.py
COPY ./backtest_runner.py /app/backtest_runner.py
# Gerekliyse diğer .py dosyalarını da buraya ekleyin (örn: data_downloader.py)
# COPY ./data_downloader.py /app/data_downloader.py

# .env.example dosyasını kopyala (opsiyonel ama iyi bir pratik)
COPY ./.env.example /app/.env.example

# Ortam değişkenleri
ENV RUNNING_IN_DOCKER=true
ENV PYTHONUNBUFFERED=1

# Botu çalıştırmak için varsayılan komut
CMD ["python", "main.py"]