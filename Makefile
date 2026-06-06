.PHONY: scaffold install download-data eda train-baseline train-xgb train-lgbm \
        docker-up docker-down ci-local drift-check airflow-init test lint

BASE_DATA := data/raw/PS_20174392719_1491204439457_log.csv

scaffold:
	mkdir -p data/raw data/processed data/sample models notebooks \
	         api/app/routers api/app/schemas api/app/services api/tests \
	         training/features training/models drift infra/terraform \
	         airflow/dags mlflow scripts .github/workflows

install:
	pip install -e ".[dev]"

download-data:
	bash scripts/download_data.sh

eda:
	jupyter nbconvert --to notebook --execute notebooks/00_eda.ipynb --output notebooks/00_eda_executed.ipynb

train-baseline:
	python training/train.py --model baseline --data "$(BASE_DATA)"

train-xgb:
	python training/train.py --model xgb --data "$(BASE_DATA)"

train-lgbm:
	python training/train.py --model lgbm --data "$(BASE_DATA)"

train-xgb-smote:
	python training/train.py --model xgb --data "$(BASE_DATA)" --smote

train-sample:
	python training/train.py --model baseline --data data/sample/paysim_sample_1k.csv

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

lint:
	ruff check api/ training/ drift/
	mypy api/ training/ --ignore-missing-imports

test:
	pytest api/tests/ -v

ci-local: lint test

drift-check:
	python drift/monitor.py

airflow-init:
	docker compose run --rm airflow airflow db init
	docker compose run --rm airflow airflow users create \
		--username admin --password admin --firstname Admin \
		--lastname User --role Admin --email admin@local.dev
