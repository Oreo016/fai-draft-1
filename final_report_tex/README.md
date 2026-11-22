# Group 27 final paper

The paper is [main.pdf](main.pdf), with [LaTeX source](main.tex) and [BibTeX references](references.bib). The original repository instructions are in the [project README](../README.md). This supplement documents the simulator, archived inputs, and paper build.

The configured Git remote is [Oreo016/fai-draft-1](https://github.com/Oreo016/fai-draft-1). The manuscript and newer result archives were assembled from the local working tree. Before submission, verify that the link is accessible to the instructor and that the submitted revision or accompanying archive contains the exact source and data listed below.

## Run the simulator and experiments

Run these commands from the repository root. A full main evaluation launches 120 episodes with six workers, followed by 100 oracle contexts. The recorded run took about 32 minutes on the original host and wrote a new timestamped directory.

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m voc_sar --config configs/default.yaml --validate-only
python -m voc_sar --config configs/default.yaml
~~~

The installed package requires PyYAML and Pillow. The default evaluates Never, Random, Full, Fixed-frequency, aware myopic and five-step policies, and the five-step agnostic comparator. It also executes the prospective two-message actionable-fact arm. Each run records an exact configuration, source hashes, per-step episode traces, oracle context traces and a summary.

~~~powershell
python scripts/run_research_experiment.py E04
python scripts/run_research_experiment.py E09
python scripts/run_research_experiment.py E10
python scripts/run_research_experiment.py E12
python scripts/run_all_research_experiments.py
python scripts/sweep_gamma_lambda.py
~~~

The E-series and sweep use the separate study configuration in `configs/experiments/study_base.yaml` and per-study YAML files. New runs write fresh timestamps. The legacy fixed-budget and dedicated H1/H4 outputs reside under `project/faiproj/results/` with their own configuration and archived implementation.

## Regenerate results and figures from the archives

The primary source is `results/run_20261122_144033_944398/`. Its `episodes.json` stores 120 episode records, and `contexts.json` stores 100 held-out oracle records. The previous fixed-step run is `results/run_20261122_135906_541004/`. The exploratory results and nested archives are listed with exclusion and ownership notes in the [inventory](data_summary/experiment_inventory.csv) and [audit notes](AUDIT_NOTES.md).

~~~powershell
python -m pip install matplotlib
python final_report_tex/scripts/reproduce.py
python final_report_tex/scripts/check_legacy.py
~~~

The figure script checks recorded values against raw traces, applies the seed-level analysis, and writes the manuscript tables, vector plots and numerical summaries. The legacy script recomputes the secondary archived H1, H2 and H4 comparisons from their separate raw data. The PNG result images are plots generated from these data for the screenshot requirement.

## Compile

From `final_report_tex/`, use a LaTeX distribution with `latexmk`, or build with Tectonic. Both run the BibTeX and cross-reference passes.

~~~powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# Alternative
tectonic main.tex
~~~

You can also upload the report folder to Overleaf, retaining `main.tex`, `PRIMEarxiv.sty`, `references.bib`, `figures/`, and `tables/`, and select `main.tex` as the main document. Figure regeneration requires the local archives and Python. LaTeX compilation uses the generated assets.

The `PRIMEarxiv.sty` layout is from Moulay A. Akhloufi's [Arxiv & PRIME AI Style Template](https://www.overleaf.com/latex/templates/arxiv-and-prime-ai-style-template/qdnhqytdqzsc), published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The manuscript replaces the sample content and retains the template's page geometry and body typography.
