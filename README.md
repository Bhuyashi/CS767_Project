# CS767 Project

Analyses on MIMIC-CXR and linked MIMIC-IV where applicable.

| Study | Topic | Features |
|-------|--------|--------|
| **Study 1** | Circadian language drift in radiology reports | data → features → proxy radiologist clusters → mixed models, figures, inference |
| **Study 2** | VLM lead-time cohort | |

---

## Study 1: Circadian language drift

**Question:** Do report-level language features (length, hedging, diversity, certainty proxy, etc.) vary with time-of-day, after accounting for illness severity and approximate writer style?

**Dataset**

- **MIMIC-CXR:** `mimic-cxr-2.0.0-metadata.csv` (`StudyDate`, `StudyTime` → `study_datetime`), `mimic-cxr-reports` text files, `mimic-cxr-2.0.0-chexpert.csv` for per-study **severity** (count of positive labels among the 14 CheXpert columns).
- **MIMIC-IV:** `patients.csv` for **gender** and **anchor_age** (demographics; severity comes from CheXpert, not from this table).

**Pipeline**

1. **Phase 1 - Data extraction:** Parse report **FINDINGS** / **IMPRESSION** when possible; otherwise use full text. Join reports to study times on `(study_id, subject_id)`. Infer whether acquisition times are roughly **minute-level** or **coarse 3-hour** metadata; if coarse, circadian labels fall back to **day vs night** (6:00–18:00 vs otherwise); if finer, use **night / morning / afternoon / evening** (0–6, 6–12, 12–18, 18–24).
2. **Phase 2 - Language features:** Per report: word count, hedge phrase rate (custom radiology-style lexicon, case-insensitive), mean words per sentence, type-token ratio (lowercased word tokens), lexicon-based **certainty score** (certain vs uncertain word counts / length). *ClinicalBERT-style scores are not in the current pipeline.*
3. **Phase 3 - Proxy “radiologist”:** TF-IDF bag-of-words on report text + **k-means** cluster id (`radiologist_cluster`). This is a **deliberately rough** writer-style proxy (not true reader identity).
4. **Phase 4:** Mixed-effects models, Cohen’s *d*, Bonferroni correction, ICC for cluster random effects.
5. **Phase 5:** Tables and circadian bin figures for the final writeup.

Runs write **timestamped logs** under `code/logs/` (e.g. `study1_YYMMDD_HHMMSS.log`) in addition to console output.

### Repository layout

```text
code/
  study_logging.py          # shared console + file logging
  logs/                     # per-run log files (gitignored if you prefer)
  study1/
    core/
      constants.py          # CheXpert label list, hedge phrases, certainty lexicons
      data_io.py            # load metadata, reports, CheXpert, MIMIC-IV patients
      text_processing.py    # section parsing, circadian bins, time granularity
      features.py           # feature extraction language features
      clustering.py         # feature extraction TF-IDF + k-means
      pipeline.py           # wires feature extraction steps
    scripts/
      feature_extractor.py  # CLI: build feature extraction dataset
      read_dcm.py           # optional: dump DICOM header fields to .txt
```

### Data layout

Place credentialed copies of the datasets under `data/` (paths are defaults in the CLI):

- `data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv`
- `data/MIMIC-CXR/csv/mimic-cxr-2.0.0-chexpert.csv`
- `data/MIMIC-CXR/mimic-cxr-reports/files/` (tree of `p<subject_id>/s<study_id>.txt`)
- `data/MIMIC-IV/csv/patients.csv`

The extractor **fails fast** with a clear error if any required path is missing.

**Dependencies:** Python 3.10+ recommended; install `pandas`, `numpy`, `scikit-learn`, `statsmodels`, and `matplotlib`. Optional for ICC fallback/alternate calculation: `pingouin` (no `requirements.txt` is checked in yet—add one when you pin versions for submission).

### Run

```bash
python code/study1/scripts/feature_extractor.py --k-clusters 30
```

Smaller dry run:

```bash
python code/study1/scripts/feature_extractor.py --max-reports 500 --k-clusters 20
```

Useful flags: `--metadata-csv`, `--chexpert-csv`, `--reports-root`, `--mimic-iv-patients-csv`, `--output-csv`, `--qc-csv`, `--random-state`, `--log-level`.

**Outputs**

- `data/MIMIC-CXR/csv/study1_feature_extraction_features.csv` - one row per report with `study_id`, `subject_id`, `study_datetime`, `circadian_bin`, `report_text`, language features, `severity`, `radiologist_cluster`, `gender`, `anchor_age`.
- `data/MIMIC-CXR/csv/study1_feature_extraction_qc.csv` - single-row summary (row counts, detected time granularity, circadian mode, bin counts).

### Run stats modeling and results analysis

```bash
python code/study1/scripts/stats_modelling.py
```

Useful flags: `--input-csv`, `--out-dir`, `--alpha`, `--force-three-comparisons`, `--log-level`.

**Stats modeling and results outputs**

- `data/MIMIC-CXR/csv/study1_stats_results/study1_table1_descriptives.csv` - mean, SD, and mean ± SD by circadian bin for all five features (+ bin sample size).
- `data/MIMIC-CXR/csv/study1_stats_results/study1_table2_mixedlm_effects.csv` - mixed-effects circadian contrasts, p-values, Bonferroni-adjusted p-values, Cohen's *d*, and test counts.
- `data/MIMIC-CXR/csv/study1_stats_results/study1_icc_summary.csv` - ICC per feature (pingouin ICC2 when available; otherwise random-intercept variance ratio).
- `data/MIMIC-CXR/csv/study1_stats_results/study1_feature_distributions.png` - multi-panel violin+box plot by circadian bin (one panel per feature).

### Optional: DICOM metadata to text

If you need raw header fields from on-disk DICOMs (separate from the metadata CSV pipeline):

```bash
python code/study1/scripts/read_dcm.py data/MIMIC-CXR/files
```

Adjust the path to your local `files/` root.

---

## Study 2: VLM lead-time cohort (planned)

When you add the second study, a practical convention is:

- `code/study2/` for code, mirroring `study1` (e.g. `core/`, `scripts/`).
- `configure_study_logging(study_name="study2", …)` so logs stay separated.
- This README: extend the table at the top and add a **Study 2** section with data sources, run commands, and outputs.

---

## Ethics and data use

MIMIC-CXR and MIMIC-IV require **appropriate credentialing and DUA compliance**. Do not commit patient identifiers, raw restricted exports, or credentials to the repository.
