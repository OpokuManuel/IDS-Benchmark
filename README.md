# Leakage-safe ML intrusion-detection benchmark (UNSW-NB15)

**Author:** Opoku Emmanuel Atta · **Status:** results reproduced from the code in this repo · **Python 3.12**

**Question.** How much do standard evaluation practices overstate the performance of ML intrusion detectors,
and which of the errors that remain are fixable by better models versus caused by the data itself?

**Short answer.** On the official UNSW-NB15 files, 20% of test flows have a near-identical twin in the training set,
42% of training rows are exact duplicates, and thousands of identical feature vectors carry contradictory labels.
When near-duplicate groups are kept entirely on one side of the split, the best baseline (gradient boosting) drops from
macro-F1 **0.70** (random split on cleaned data) to **0.43**, and the false-positive rate at 95% detection rises from **12%** to **18%**.

---

## 1. Data

| | |
|---|---|
| Dataset | UNSW-NB15, official *training-set* (175,341 rows) and *testing-set* (82,332 rows) CSVs, 10 classes (Normal + 9 attack families) |
| Origin | Moustafa & Slay, UNSW Canberra (2015). Primary source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. Check the page for usage terms and **cite the paper** if you reuse the data. |
| What I used | GitHub copies, because the UNSW server was not reachable from my build environment. Both files are pinned by SHA-256 in `data/raw/SHA256SUMS`; the training file was byte-identical across two independent mirrors; row counts match UNSW's documentation. I could not compare against UNSW's server directly, so run `sha256sum -c` if you can reach it. |
| Download | `python scripts/download_data.py` (verifies checksums) |

**Why not CIC-IDS2017?** It is >1 GB and was not reachable from my environment. Adding it (with its real timestamps) is the natural extension; see Limitations.

### Data problems found and how they were handled (`results/data_quality.csv`, `results/audit.json`)

* **Exact duplicates:** 42% of official training rows and 34% of test rows. After pooling both files, 103,989 duplicate rows are removed (257,673 -> 153,684 unique feature vectors).
* **Cross-split leakage in the official split:** 8,541 test rows (10.4%) have an exact twin in the training set; 20.4% have a near-twin (nearest-neighbour distance < 0.1 in standardised log-feature space).
* **Contradictory labels:** 414 feature vectors carry both benign and malicious labels (1,758 rows) -> **dropped**. A further 2,031 vectors carry several attack categories -> **kept once, with the most frequent category** (ties alphabetical). This shrinks the rare classes; see `data_quality.csv`. 77% of Analysis, 85% of Backdoor and 77% of DoS rows sit in such conflicting vectors, which puts a ceiling on what any flow-feature model can do for those classes.
* **Cleaned pool:** 153,270 rows.

## 2. Evaluation protocols

| Protocol | Data | Split | Purpose |
|---|---|---|---|
| **P0 official** | raw official files | as published | what most papers report |
| **P1 random** | cleaned pool | stratified random 5-fold | removes exact duplicates only |
| **P2 grouped (primary)** | cleaned pool | stratified **group** 5-fold | near-duplicate groups never straddle train/test |

**Groups (the "session proxy").** The official CSVs contain **no timestamps and no IP addresses**, so a genuine time- or session-based split is
*not possible with these files*. Instead, flows are grouped by protocol, service, state, both TTLs and a coarse log2 size profile
(packets and bytes in each direction), using no labels (`src/idsbench/splits.py`). Each group sits wholly in train or wholly in test.
Measured effect (`results/leakage_audit.csv`, `fig2`): near-twin leakage falls from 15.7% (P1) to 0.7% (P2), with zero shared groups (enforced by `tests/`).
P2 is a **stress test** (it also holds out whole traffic "archetypes"); P1 is optimistic; a true temporal split would lie somewhere between, and I cannot measure it here.
Sensitivity to the grouping granularity (1 / 2 / 4 size-bins per octave) is reported below.

**Models** (fixed after a small sweep on an *inner validation split of one training fold*, never test data: `scripts/00_tune_on_validation.py`, `results/tuning_inner_validation.json`):
logistic regression (C=10), random forest (100 trees), histogram gradient boosting (300 rounds, lr 0.05, 15 leaves, L2=1).
Features: all numeric columns (log1p) + one-hot proto/service/state. All seeds fixed (42).

**Metrics.** Per-class precision/recall, macro-F1, binary ROC/PR-AUC (score = 1 - P(Normal)), FPR at 95% and 99% detection, and
an *honest operating point*: the threshold for 95% detection is chosen on a validation split carved from the **training** fold, then measured on test.

## 3. Results (pooled out-of-fold predictions; every flow is tested once)

| Protocol | Model | Accuracy | Macro-F1 | ROC-AUC | FPR @ 95% det. | FPR @ 99% det. |
|---|---|---|---|---|---|---|
| P0 official | Logistic reg. | 0.730 | 0.373 | 0.969 | 0.229 | 0.380 |
| P0 official | Random forest | 0.758 | 0.473 | 0.980 | 0.132 | 0.305 |
| P0 official | Grad. boosting | 0.767 | 0.511 | 0.983 | 0.134 | 0.317 |
| P1 random | Logistic reg. | 0.809 | 0.484 | 0.966 | 0.191 | 0.264 |
| P1 random | Random forest | 0.872 | 0.663 | 0.981 | 0.127 | 0.229 |
| P1 random | Grad. boosting | 0.871 | 0.697 | 0.982 | 0.124 | 0.224 |
| **P2 grouped** | Logistic reg. | 0.633 | 0.256 | 0.856 | 0.371 | 0.431 |
| **P2 grouped** | Random forest | 0.745 | 0.377 | 0.955 | 0.223 | 0.351 |
| **P2 grouped** | Grad. boosting | 0.757 | 0.429 | 0.962 | 0.179 | 0.261 |

Full table with PR-AUC and validation-chosen thresholds: `results/summary_by_protocol.csv`. Fold-to-fold spread under P2 (macro-F1 sd about 0.025;
AUC ranges 0.91-0.99 for the tree models and **0.59-0.96 for logistic regression**) is large: treat differences smaller than about 0.05 macro-F1 as noise.

**Reading it honestly.**
* The official split (P0) does **not** simply "inflate" scores: it scores *below* the cleaned random split because the official partitions also have their own distribution shift. Its problem is that it mixes leakage (duplicates) with shift, so you cannot tell which is driving the number.
* Moving from a random to a grouped split costs gradient boosting 0.27 macro-F1 and 0.06 FPR at 95% detection. The ranking of models is stable (GB > RF > LR) across all three protocols and all grouping granularities (`fig6`).
* The honest operating point at a nominal 95% detection target gave 95.3% detection at **19.4%** FPR for gradient boosting under P2 (`val_thr_*` columns): a false-alarm rate that is not deployable on its own.

**Per-class (P2, gradient boosting; `results/per_class_metrics.csv`).** Strong: Normal (R 0.90, P 0.88), Exploits (R 0.91, P 0.73). Weak: **Generic recall 0.07**,
Reconnaissance 0.15, DoS 0.25, Analysis P 0.12 / R 0.32. Under P0 the same classes score far higher (Generic R 0.97, Reconnaissance R 0.81), which is the leakage-and-duplication effect.

**Ablation (random forest, P2; `fig7`).** Removing TTL features (`sttl`, `dttl`, `ct_state_ttl`) raises FPR@95% from 0.223 to **0.366**: the model leans heavily on them. Removing TCP sequence numbers (0.221) or proto/service/state (0.239) changes little.

**Group-strictness sensitivity (gradient boosting macro-F1 / FPR@95%):** 4 bins 0.461 / 0.162, **2 bins 0.429 / 0.179**, 1 bin 0.404 / 0.182. Stricter grouping -> lower scores, as expected.

## 4. Error analysis (gradient boosting, P2; `results/error_analysis.json`, `fig4`)

* **Which classes get confused:** 81% of Generic is predicted Fuzzers; 64% of DoS is predicted Exploits; 35% of Fuzzers and 30% of Reconnaissance are predicted Normal.
  The DoS->Exploits confusion matches the label-conflict finding (flow features alone often cannot separate them), so it is largely a data ceiling, not a model defect.
* **Missed attacks** (at an oracle 95%-detection threshold): almost all misses are Fuzzers (15.8% of Fuzzers missed; every other class under 3%).
* **False alarms:** the false-alarm rate on benign flows is 37% for HTTP, 30% for FTP and 20% for flows with no identified service, versus 0.2% for DNS; 74% of all false alarms come from benign flows with no identified service.
* **Shortcut evidence (verified for logistic regression, fold 0):** 93% of UDP/DNS flows in the official training file are "Generic" attacks, and when a 10,255-flow benign DNS cluster was held out, logistic regression labelled 10,250 of them Exploits. The random forest ablation shows that removing the categorical columns alone does not fix tree models, so the shortcut they use is carried by other features (the TTL ablation points there). I have not run a deeper attribution analysis.
* The low Generic recall under P2 is *consistent with* Generic being 88% duplicates (its distinct patterns are few, so holding out a group removes whole patterns), but I did not test that explanation directly.

## 5. Limitations (please read)

1. **No true temporal or session split** (the files have no timestamps/IPs). P2 is a near-duplicate-group proxy. Next step: repeat with CIC-IDS2017, which has timestamps, and split by day/time.
2. **One grouping seed.** All P2 numbers use fold assignment seed 42; I did not complete repeats with other seeds (`scripts/02_run_experiments.py --stage seeds` is provided but its results are not in this report). Fold-level spread is reported instead.
3. **Group definition is a heuristic**, chosen by measured leakage (`fig2`), not an oracle for "same attack session".
4. **Label resolution** (majority category for conflicting vectors) is a judgement call that shrinks rare classes.
5. **Hyper-parameters** were tuned on one inner validation split only; baselines are deliberately simple.
6. **FPR@TPR columns use an oracle threshold**; the `val_thr_*` columns give the honest, transferred operating point.
7. Pooled out-of-fold metrics mix folds whose score scales differ; per-fold values are in `results/raw/*.json`.

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # verifies SHA-256
python scripts/01_audit_data.py          # data-quality + leakage audit        (~5 min)
python scripts/00_tune_on_validation.py  # optional: re-derive hyper-parameters (~10 min)
python scripts/02_run_experiments.py --stage main   # P0/P1/P2               (~35 min, 1 CPU)
python scripts/02_run_experiments.py --stage ablation
python scripts/02_run_experiments.py --stage sensitivity
python scripts/03_make_report_assets.py  # tables + figures
python report/build_report.py --author "Your Name"   # 2-page PDF
pytest -q                                # 11 tests guarding the leakage claims
```
Runs are resumable and deterministic given the pinned versions (`requirements.txt`).

## 7. Repository layout
`src/idsbench/` library · `scripts/` pipeline · `tests/` · `results/` (tables, `raw/` per-fold metrics + predictions, `figures/`) · `report/` PDF · `data/raw/` data + checksums

## References
* Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS.
* Arp, D. et al. (2022). *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security.
* Engelen, G., Rimmer, V., Joosen, W. (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE S&P Workshops.
* Zoghi, Z. & Serpen, G. (2021). *UNSW-NB15 computer security dataset: analysis through visualization.* arXiv:2101.05067.
