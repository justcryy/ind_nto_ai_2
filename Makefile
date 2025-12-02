.PHONY: install train predict validate run clean download-data prepare-data

install:
	@echo "Assuming dependencies are managed by poetry. Use 'poetry install'."

download-data:
	@echo "Downloading data from Google Drive..."
	@mkdir -p data/raw
	@poetry run gdown --folder https://drive.google.com/drive/folders/1ioqrB9146B9DLcEVD7V3LIl46IYAyBfm -O data/raw/
	@if [ -d "data/raw/public" ]; then \
		mv data/raw/public/*.csv data/raw/ && \
		rmdir data/raw/public && \
		echo "Files moved from data/raw/public/ to data/raw/"; \
	fi
	@echo "Data downloaded successfully to data/raw/"

prepare-data:
	@echo "Preparing and processing data..."
	poetry run python -m src.baseline.prepare_data

train:
	@echo "Running training script..."
	poetry run python -m src.baseline.train

predict:
	@echo "Running prediction script..."
	poetry run python -m src.baseline.predict

validate:
	@echo "Running validation script..."
	poetry run python -m src.baseline.validate

run: prepare-data train predict validate
	@echo "Full pipeline executed successfully."

clean:
	@echo "Cleaning output directories..."
	rm -f output/models/*
	rm -f output/submissions/*
	rm -f data/processed/*
	@echo "Done."
