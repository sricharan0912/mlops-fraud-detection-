FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow[extras]>=2.13" psycopg2-binary boto3

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
