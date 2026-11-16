# Marketing Risk Research

Master's thesis: predicting next-year marketing performance deviation risk (MPDR) for A-share consumer-facing firms, 2013-2025, with BO-TabNet. Read this file first. It is the only guide in the project. Every other file is data, a thesis document or a reference.

## 1. Folder map

| Folder | Contents |
|---|---|
| `01_thesis/` | `research_proposal.docx` (current proposal) and `mpdr_technical_roadmap.pptx` |
| `02_data/final/` | The data the experiments use |
| `02_data/raw/` | Original sources, read-only |
| `03_literature/` | `cited/` (43 papers the thesis cites), `candidate_pool/` (67 papers collected but not cited), `abstracts_digest.docx` |
| `04_experiments/` | Modeling code and results: package `mpdr/` (features, labels, split, metrics, `stages.py` for the persistence stage, `learners.py` for Lasso / LightGBM / TabNet and their Bayesian optimization, `modules.py` for the predictor modules and the text block, `attribution.py` for group Shapley values), `run_lasso.py` (single-stage Lasso baseline, `results/`, `results_lagged/`), `run_two_stage.py` (main two-stage models, module ablations and tests of predictive accuracy, `results_two_stage/`), `run_evaluation.py` (proposal Table 13 metrics with bootstrap CIs and research question 3 accuracy by direction, from the saved predictions), `run_rq3_diagnosis.py` (research question 3 drivers and the firm-level diagnosis `firm_diagnosis.csv`, run after the other two), `build_channel_features.py` (customer and supplier HHI in `model_features_X.csv`, from FN_FN010), `run_cumulative_ablation.py` (proposal Table 12 cumulative ablation with per-specification Bayesian search and a Shapley decomposition of the modules, `results_cumulative/`), `results_two_stage_text/` and `results_two_stage_no_text/` (the 2026-11-16 rerun of `run_two_stage.py` with and without the text module, plus `paired_text_vs_control.csv`; `results_two_stage/` holds the 2026-10-22 run without text), `run_robustness.py` (tail-cutoff and EWAL sensitivity from the saved predictions; expanding-window evaluation, `results_expanding/`; the lagged label reruns the pipeline through `MPDR_LABEL_PATH` and `MPDR_TWO_STAGE_DIR`) and `tests/` (`python3 -m pytest`). Requirements in `requirements.txt`; the device shell lacks them, so runs so far used the cloud container. |
| `04_experiments/text_scoring/` | Seven-dimension annual-report text scoring: `rubric.md` (the instrument and the validation steps), `anchors.csv` (100 labelled anchors), `system_prompt.md` built from both, `output_schema.json`, the MD&A text sent to the model (`mdna/`, `mdna_index.csv`, built by `extract_mdna.py`), `manual_spans.csv` (hand-set page ranges) and `manual_texts/` (texts recovered by OCR from the PDF for reports whose stored text layer is unreadable), `score_tools.py`, `run_cli_scoring.py`, `repair_quotes.py`, `finalize_year.py`, `check_scores.py` (scoring, quote repair, wrap-up and checks), per-year results in `results/YEAR/` (`scores.jsonl`, `retest.jsonl` = independent 5% re-score for repeat reliability, `anchor_test.jsonl`), and `build_final_table.py`, which writes `02_data/final/annual_report_text_scores.csv` |
| `99_archive/` | Superseded drafts, build scripts, logs, verification records and backups. `MANIFEST.tsv` maps each item to its original path. Do not read it unless something has to be recovered. |

## 2. Data

Paths stored inside data files are relative to `02_data/`. Read stock codes as 6-digit text (`dtype={'股票代码': str}`). CSVs are UTF-8 with BOM, and column names are in Chinese.

### final/

| File | Rows | Contents |
|---|---|---|
| `model_features_X.csv` | 12,004 | Features at report year t (2013-2024) for predicting year t+1. Key: 股票代码 + 年度. Includes the four sample flags and the path to the annual report available at the prediction date (`时点年报PDF`, `时点压缩全文`). |
| `model_target_Y.csv` | 12,004 | `下一年实际营销绩效` (log gross profit in t+1), keyed by 股票代码 + 报告年度 + 预测年度. |
| `data_dictionary.csv` | 129 | Every column of X and Y: role, whether it is a model candidate (32 numeric, 2 categorical, 4 flags, the target), definition, unit and source. Also lists the seven text features, now filled from `annual_report_text_scores.csv`. |
| `sample_screening.csv` | 13,592 | Keep or exclude, with the reason, for every firm-year. |
| `text_extraction_tasks.csv` | 11,572 | Firm-years whose annual report was published by 30 April, with the selected PDF and page text, as input to the seven-dimension LLM scoring. |
| `annual_report_text_scores.csv` | 12,004 | The seven text scores (0-4: 市场感知, 品牌管理, 客户关系管理, 渠道管理, 产品开发与新品, 营销沟通, 数字营销 `叙事强度`), their sum `叙事强度合计` and the text length `年报文本字数`, per firm-year. Key: 股票代码 + 报告年度. 披露晚于4月30日 flags the 431 reports disclosed after 30 April of the prediction year (not available at the prediction point). Every row of X has a score. Scored from the MD&A text only, repeat reliability by year from `04_experiments/text_scoring/results/YEAR/retest.jsonl`. |
| `mpdr_label.csv`, `mpdr_benchmark_fit.csv` | 12,004 / 12 | MPDR label from the proposal's benchmark, fitted year by year as a cross-section on non-ST firms (industry effects, 1/99 winsorized), plus each year's fit statistics. Built by `04_experiments/mpdr/labels.py`. |
| `mpdr_label_lagged.csv`, `mpdr_benchmark_fit_lagged.csv` | 12,004 / 12 | The same label with ln GP(t) added to the benchmark, so it measures the surprise component. |
| `restatement_lookahead_flags.csv` | 1,310 | Firm-years with an error correction disclosed after the information cutoff. `疑似使用更正后数值` marks the 489 rows whose CSMAR revenue or total assets do not match the report available at the cutoff. |
| `firm_list.csv`, `target_industries_48.csv`, `macro_annual_2013-2025.csv` | | 1,619 firms, the 48 SW industries, and the four macro indicators with bulletin dates and links. |

MPDR (actual t+1 performance minus the expected level) is estimated year by year (section 3, item 5). Missing-value handling, winsorizing and scaling of features are fitted on the training fold only.

### raw/

- `csmar/`: CSMAR downloads. Each table has its original zip (licence note in the file name) and an unzipped folder with the data CSV and its field description (`[DES][csv].txt`): `income_statement/`, `balance_sheet/`, `cash_flow/`, `company_info_annual/`, `industry_class_annual/`, `top5_customers_suppliers/`, `listing_status_changes/`.
- `nbs_bulletins/`: National Bureau of Statistics annual bulletins 2013-2025 (text) and `来源索引.json` with links and publication dates.
- `annual_reports/`: 14,219 annual-report PDFs (`pdf/`, 38 GB) and page-level text (`text/`, one `.jsonl.gz` per report, one JSON line per PDF page with `PDF页序号` and `正文`). `index.csv` has one row per firm-year (13,600, including the 8 never published) with disclosure date, revision flag, file paths and source links. `page_quality.csv` lists 1,594 OCR pages to double-check.

## 3. Current sample decisions

The data follows these decisions. Where the proposal's section 4.1 says otherwise, the data is current and the proposal text still needs updating.

1. The 8 firm-years whose annual report was never published are dropped (13,600 → 13,592).
2. ST or *ST at the prediction date, non-positive gross profit, an uncomputable log sales expense and an uncomputable capex intensity are kept and flagged, not excluded (decided 2026-10-22). The proposal still says these are excluded.
3. Industry membership uses the SW version in force each year (SW2011 for 2013, SW2014 for 2014-2020, SW2021 for 2021-2025), mapped to the 48 target industries. 1,425 firms match SW2021 directly and 194 enter only through the historical mapping. The proposal says SW2021 in the prediction year.
4. Text features are scored for every report that exists, whenever it was disclosed, from the mandated sections 公司业务概要 / 经营情况讨论与分析 / 管理层讨论与分析 / 董事会报告 only (letters to shareholders excluded), with the reviewed rubric and anchors in `04_experiments/text_scoring/` (decided 2026-10-22; the rule limiting scoring to reports disclosed by 30 April was dropped on 2026-10-26). `披露晚于4月30日` in `annual_report_text_scores.csv` marks the 431 reports that were not public at the prediction point, so the model can treat them as missing or compare with and without them. A few reports need a page range set by hand, a re-read of the PDF or a recovered text; they are listed in `04_experiments/text_scoring/manual_spans.csv` and `manual_texts/` (600941 for 2022-2024 uses the business-discussion pages, decided 2026-10-27).
5. The MPDR benchmark is estimated as a separate cross-section for each prediction year (not on the training window), using non-ST firms; ST firms are then scored with those coefficients (decided 2026-10-22). Because the label is an ex-post outcome, this involves no look-ahead.
6. Split by report year: train 2013-2020, validation 2021-2022, test 2023-2024. Every preprocessing statistic is fitted on the training fold.
7. Restated financials: 489 firm-years may carry values corrected after the prediction date. They stay in the sample, and the test metrics are also reported without them.
8. MPDR stays the proposal's level label. Models are two-stage: stage 1 is OLS of MPDR on ln GP(t), the GP ≤ 0 flag and the benchmark regressors, fitted on the training fold; Lasso, LightGBM, TabNet and BO-TabNet then model its residual, and feature importance and module ablations are read on stage 2. Nested comparisons with stage 1 use the Clark-West test, non-nested ones Diebold-Mariano; tail metrics include firms newly entering the bottom 10%. LightGBM is tuned with the same 40-trial Bayesian search as TabNet. The text scores join the design and the ablation automatically once `02_data/final/annual_report_text_scores.csv` exists (decided 2026-10-22); the text module also carries `text_missing` and `text_late` (the 披露晚于4月30日 flag, a feature and not a filter, decided 2026-10-27).
9. Customer and supplier concentration (客户集中度, 供应商集中度) are HHIs, the sum of squared rank 1–5 shares in FN_FN010, replacing the squared top-5 share. About 29% of firm-years disclose only the top-5 total; their HHI stays missing (口径 仅披露合计) and the model uses the top-5 share, now a feature, with a missing flag. Partial disclosures spread the undisclosed remainder evenly over the missing ranks, and the 50 firm-years whose ranks contradict the stated total by more than one point get no HHI. The channel module follows proposal Table 9: top-5 shares, largest-counterparty shares, HHIs and inventory turnover (decided 2026-10-22).

## 4. Literature

`cited/` holds one file per paper, named `Author-Year-Title.pdf`. One cited paper has no PDF yet: Wedel & Kannan (2016), Marketing Analytics for Data-Rich Environments. In `candidate_pool/`, files are named `Year-Title.pdf`. A `-非正式` suffix marks a working-paper or preprint version rather than the published article. When a candidate gets cited, rename it to `Author-Year-Title.pdf` and move it to `cited/`.

## 5. Rules for new files

**Naming.** English, lowercase, words joined by underscores (`feature_importance_by_module.csv`, `train_bo_tabnet.py`). Name the content and not the process, so no `final`, `new`, `revised`, `v2`, `修订版` or `优化版`. A date (`YYYY-MM-DD`) appears only when the date is the content, as in a snapshot or meeting notes.

**Where things go.** Thesis documents go in `01_thesis/`, data in `02_data/` (new model inputs in `final/`, new sources in `raw/`) and papers in `03_literature/`. Modeling code and results go in `04_experiments/` (package `mpdr/`, runners `run_*.py`, outputs in `results*/`). Do not create any other top-level folder.

**Versions.** Keep one current file per document under its plain name and overwrite it as you work. Keep an older version only at a milestone, such as a version sent to the supervisor or committee. Move that copy to `99_archive/YYYY-MM-DD/` and add a row to `MANIFEST.tsv`. Experiment code goes under git, where history replaces `_v2` copies.

## 6. Cleanup after every task

Before a task is reported done, whoever did it (Claude or Adrain) checks the tree.

Delete right away: caches (`__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/`), lock files (`~$*`, `*.lock`, `*.pid`), render previews and screenshots, inspection dumps, logs of finished runs, empty folders, and scratch files from one-off checks.

Archive to `99_archive/YYYY-MM-DD/` with a `MANIFEST.tsv` row: a file replaced by a newer version, pilot runs once the full run exists, backups, and build or verification scripts once their output is in `final/`.

Never write into the project: progress logs, handoff notes, explanatory notes about a step, or dated backup copies. A decision that changes the data goes in section 3 of this README. Scratch work stays outside the project folder.

Ask Adrain before deleting anything in `01_thesis/`, `02_data/raw/` or `02_data/final/`, anything that cost money to obtain, and anything another file refers to.

## 7. Reading the project efficiently (for Claude)

Start from this README. Read `data_dictionary.csv` before opening X or Y, and load only the columns you need. Do not bulk-read `raw/annual_reports/` or `99_archive/`. If this README and a file disagree, check the file, fix whichever is wrong, and keep this README current.

*Last updated 2026-11-16.*
