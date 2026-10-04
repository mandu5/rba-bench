# Reproduce the analysis, the manuscript numbers, and the manuscript.
#
#   make            regenerate artifacts, numbers and the PDF
#   make analysis   rerun analyses, model calibration and training-setup audit
#   make numbers    regenerate paper/numbers.tex from the artifacts
#   make paper      build paper/main.pdf
#   make verify     prove nothing drifted

PYTHON ?= python3
export SOURCE_DATE_EPOCH ?= 1700000000

.PHONY: all analysis numbers figures paper verify clean

all: paper

analysis:
	$(PYTHON) scripts/triage_rank_resolvability.py
	$(PYTHON) scripts/metric_disagreement_null.py --power --floor \
	  --drop-metric improvement_per_1k_updates
	$(PYTHON) scripts/validate_inference.py
	$(PYTHON) scripts/export_training_setup.py

numbers: analysis
	$(PYTHON) scripts/export_paper_numbers.py
	$(PYTHON) scripts/export_abstract_text.py

figures: analysis
	$(PYTHON) scripts/make_figures.py

paper: numbers figures
	cd paper && tectonic -X compile main.tex

# Regenerating must not change anything that is committed. The artifact check
# is the one that matters: it asserts the documented command reproduces the
# committed JSON, not merely that the paper agrees with whatever JSON is there.
verify:
	$(PYTHON) -m pytest -q
	$(MAKE) analysis
	git diff --exit-code -- results
	$(PYTHON) scripts/export_paper_numbers.py
	$(PYTHON) scripts/export_abstract_text.py
	git diff --exit-code -- paper/numbers.tex paper/table_cells.tex paper/table_calibration.tex paper/table_training.tex docs/abstract.txt docs/tldr.txt
	@echo "verify: artifacts and manuscript numbers reproduce from a fresh run"

clean:
	rm -f paper/*.aux paper/*.log paper/*.blg paper/*.bbl paper/*.out
