.PHONY: test

setup:
	pip install -r requirements.txt
	pip install .

test:
	pytest tests -m "not pybox2d" --tb=line --disable-pytest-warnings

test-with-pybox2d:
	pytest tests --tb=line --disable-pytest-warnings

tensorboard:
	tensorboard --logdir experiments

monitor:
	jupyter lab
