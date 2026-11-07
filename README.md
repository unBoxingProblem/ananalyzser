# DataPilot data analysis

DataPilot provides file ingestion, row cleaning, a dataset-agnostic dashboard, and interactive visual exploration. The dashboard profiles each selected DataFrame in Python and adapts its KPIs, filters, chart recommendations, and observations to the detected schema. It does not assume a particular industry or send dataset data to an AI service.

The **Explore Data** section builds interactive Plotly visualizations from the selected dataset's current working DataFrame. Its chart builders do not call an AI service.

## Supported formats

- CSV: UTF-8/BOM and common Windows-1252 text; comma, semicolon, or tab delimiter (automatic detection by default); malformed rows are rejected.
- TSV: tab-separated text.
- XLSX: choose one worksheet at a time; sheets are never merged.
- JSON: arrays of flat objects, record-oriented objects, and JSON Lines. Nested record fields are rejected rather than flattened.
- Parquet: Pandas with PyArrow.

Other formats, including legacy XLS, PDF, DOCX, images, and pickle, are unsupported.

## Data cleaning

After loading a dataset, the **Data Cleaning** section previews and counts rows that would be removed for:

- Rows with any missing values, or only rows where every value is missing.
- Completely duplicated rows, keeping the first occurrence.
- Fully empty rows.

Nothing changes until **Apply cleaning operation** is clicked. Cleaning updates a working DataFrame while preserving the original. The active dataset can undo the most recent cleaning or reset to its original data. Datasets remain isolated in the session registry.

## Adaptive dashboard

The **Dashboard Builder** appears after **Explore Data** and starts with no charts. Add charts explicitly, preview configurations reactively, then save them to the selected dashboard. Chart cards can be edited, duplicated, deleted, and reordered. Dashboard definitions and chart settings persist in a local SQLite database under `.datapilot/` (override with `DATAPILOT_DASHBOARD_DB`); uploaded DataFrames are not stored there. Saved charts retain their dataset IDs and show an unavailable-dataset message rather than selecting a substitute.

## AI Analyst

The **AI Analyst** appears immediately after the dataset preview. Manual cleaning, exploration, and dashboard controls are hidden until **Open manual data tools** is selected. Conversation history is kept separately for each active dataset in the current Streamlit session. With Gemini configured, the analyst uses native function calls: `inspect_dataset`, `execute_dataframe_analysis`, `recommend_dashboard`, and `build_dashboard`. For a dashboard request, Pandas measures numeric distributions, numeric relationships, date trends, category-to-metric effects, and text-length patterns; DataPilot ranks chart candidates and explains why each is useful before Gemini chooses a dashboard that matches the user's requested subject and chart style. Gemini can add a supported chart beyond the ranked recommendations when the user requests it; every chart still passes column/type validation and a Plotly build before it is saved. Dashboard requests populate a saved dashboard, display its charts in the chat response, and allow the user to open the live editor. The tool loop is limited to five rounds, ten tool calls, sixteen total operations, and a 75-second execution budget. The backend never executes model-generated Python.

Registered operations include dataset profiling, numeric and categorical summaries, missing-value and duplicate checks, correlations, grouped aggregation, time trends, outlier detection, text lengths, keyword frequency and TF-IDF distinctiveness by group, row filtering, and sorting. Keyword tools normalize common column aliases and infer a missing text or grouping column only when the dataset makes the choice unambiguous. Conflicting aliases are rejected. If several group columns are plausible, the user is asked to choose. Gemini receives the actual calculated tables for explanation; grouped keyword results include counts, per-1,000-token frequency, and distinctiveness scores.

This is a safe registered analysis set, not unrestricted access to every function in Pandas. Unsupported requests are reported instead of executed as arbitrary code. Set `GEMINI_API_KEY` in `.env` or Streamlit secrets to enable Gemini. `GEMINI_MODEL` can select another supported model; the default is `gemini-3.5-flash-lite`. If the Gemini API itself fails, DataPilot runs its local Pandas planner and states which operations ran and whether it completed the requested analysis. If a selected backend operation fails, DataPilot returns the validation error to Gemini for one correction attempt, then reports a backend failure if it still cannot execute.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to enable Gemini. Do not commit a real API key.

The default upload ceiling is 100 MB per file. Set `DATAPILOT_MAX_UPLOAD_MB` to a positive integer to lower or raise the application-level limit. If raising it above 100 MB, also update `server.maxUploadSize` in `.streamlit/config.toml` so Streamlit allows the browser upload to reach the app.

Uploaded DataFrames and metadata remain in Streamlit session memory only. They are not saved permanently. The preview shows at most 100 rows; file-level parsing still loads the selected file into Pandas memory.

## Tests

```bash
pytest -q
```

Tests create temporary in-memory CSV, TSV, XLSX, JSON, and Parquet fixtures and do not require network access or API keys.

## Current limits

- One Excel worksheet is ingested per selected dataset.
- Duplicate column names are rejected so downstream selection remains unambiguous; names are never silently rewritten.
- JSON nested values are rejected rather than flattened.
- The registry is scoped to the current Streamlit session and is not persistent.
- This is a moderate-sized Pandas ingestion path, not an arbitrary-size or streaming data platform.

Next milestone: add column-level cleaning actions such as filling missing values and standardizing text, using the same preview, apply, undo, and reset flow.
