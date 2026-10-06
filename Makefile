.PHONY: help setup profile step3 step4 step4-enhancements step5 final-eval eda test

help:
	@echo "Available targets:"
	@echo "  setup  Install project dependencies"
	@echo "  profile  Regenerate the full raw-data profile"
	@echo "  step3  Regenerate Step 3 EDA, features, figures, and tables"
	@echo "  step4  Train and compare Step 4 classification models"
	@echo "  step4-enhancements  Run chronological tuning, representation, and calibration checks"
	@echo "  step5  Run explainability, robustness, and fairness audits"
	@echo "  final-eval  Refit the frozen winner and evaluate the supplied test once"
	@echo "  eda    Execute notebooks/EDA.ipynb in place"
	@echo "  test   Run automated tests"

setup:
	python -m pip install -r requirements.txt

profile:
	python -m src.data_profile --project-root .

step3:
	python -m src.step3_analysis --project-root .

step4:
	python -m src.step4_modeling --project-root .

step4-enhancements:
	python -m src.step4_enhancements --project-root .

step5:
	python -m src.step5_fairness --project-root .

final-eval:
	python -m src.final_evaluation --project-root .

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb --ExecutePreprocessor.timeout=600

test:
	python -m unittest discover -s tests -v
