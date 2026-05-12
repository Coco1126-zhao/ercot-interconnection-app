# ERCOT Interconnection Predictor
# Interface: https://ercot-interconnection-prediction.streamlit.app/

Two-layer interconnection viability tool for ERCOT energy developers:

- **Layer 1** — your existing LBNL-trained XGBoost classifier + regressor with per-instance SHAP attribution.
- **Layer 2** — KNN-style similarity match against the **live ERCOT GIS Report queue**, returning real reference projects with their actual milestone progress.

A Streamlit interface ties them together: a dark, monospace, terminal-aesthetic UI with a Folium map of the ERCOT queue, a scrollable project feed, an input form, and side-by-side prediction panels.

---

## Project structure

```
ercot-interconnection-app/
├── README.md
├── requirements.txt
├── .gitignore
├── .vscode/                       # VS Code launch + settings
├── .github/workflows/
│   └── monthly_refresh.yml        # cron: pull new ERCOT GIS xlsx each month
├── app/
│   └── app_streamlit.py           # the UI
├── data/
│   ├── raw/
│   │   └── LBNL_Ix_Queue_Data_File_thru2024.xlsx   # PRESERVED (your file)
│   ├── ercot/
│   │   ├── latest.xlsx            # always points to newest snapshot
│   │   └── snapshots/             # one xlsx per month
│   ├── geo/                       # auto-populated TX county centroids
│   └── artifacts/                 # pickled trained models (regenerated)
└── src/
    ├── data_loader.py             # PRESERVED — your existing code
    ├── features.py                # PRESERVED
    ├── models.py                  # PRESERVED
    ├── shap_analysis.py           # PRESERVED
    └── ercot/                     # NEW — built on top of your model
        ├── loader.py              # parse monthly GIS xlsx → tidy DF
        ├── fetcher.py             # scrape MIS for newest .xlsx
        ├── geo_utils.py           # substation / county → lat/lon
        ├── similarity.py          # Layer 2 KNN engine
        ├── predictor.py           # Layer 1 wrapper + SHAP per-instance
        └── train_and_save.py      # one-time training → pickled artifacts
```

> **Nothing in your existing model code was modified.** The four PRESERVED files (`data_loader.py`, `features.py`, `models.py`, `shap_analysis.py`) are byte-identical copies from your `Interconnection Predictor/Code/`.

### What's intentionally new (and why)

| New module | Why |
|---|---|
| `train_and_save.py` | The original code re-trains every Colab run. The Streamlit app needs a one-time pickle so it can serve predictions in milliseconds. |
| `predictor.py` | Wraps the trained model for **single-instance** predictions and computes per-instance SHAP. Translates ERCOT-flavoured user input into the LBNL feature schema. |
| `loader.py` | Parses the monthly ERCOT GIS xlsx, derives milestone-progress columns. |
| `fetcher.py` | Pulls the newest GIS report from ERCOT's public MIS endpoint. |
| `geo_utils.py` | Adds lat/lon (county-centroid by default; HIFLD substation upgrade documented below). |
| `similarity.py` | Layer 2: matches a user input to the most-similar live queue projects. |

### Two assumptions baked into the ERCOT → LBNL mapping (surfaced in the UI)

1. `service_type = "NRIS+ERIS"` (LBNL's modal best-case category) since ERCOT does not use NRIS/ERIS.
2. `log_queue_backlog` uses the **current ERCOT queue size** rather than the LBNL 3-yr rolling backlog, so Layer 1 reflects today's congestion, not the historical average.

These are documented in `predictor.py` and exposed in the UI's "Mapping assumptions" expander.

---

## How to run (terminal)

> **Python 3.10+ required.** Your existing model code uses `str | Path` union syntax that was introduced in Python 3.10. macOS ships with Python 3.9, so you need to install a newer one first.

### One-time: install Python 3.11

Pick one:

**A. With Homebrew** (cleanest if you don't have brew yet):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

**B. Or from python.org** — download the macOS 64-bit universal2 installer for Python 3.11 from <https://www.python.org/downloads/macos/> and run it.

Verify:
```bash
python3.11 --version       # → Python 3.11.x
```

### Run the app

```bash
cd ~/Documents/ercot-interconnection-app
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# one-time: train the LBNL model and pickle artifacts (~2–4 min)
python -m src.ercot.train_and_save

# launch the interface
streamlit run app/app_streamlit.py
```

Streamlit prints `http://localhost:8501` and opens it in your browser. `Ctrl+C` to stop.

To pull a fresh ERCOT GIS report manually (otherwise the GitHub Action runs monthly):

```bash
python -m src.ercot.fetcher
```

---

## VS Code workflow

1. **Push to GitHub** (terminal):
   ```bash
   git init
   git add .
   git commit -m "initial scaffold"
   gh repo create ercot-interconnection-app --public --source=. --push
   ```
2. **Open the folder in VS Code**: `File → Open Folder…` → pick `ercot-interconnection-app`. The included `.vscode/settings.json` auto-points to your `.venv`.
3. **Run the app inside VS Code** — three options:
   - **Integrated terminal**: press `` Cmd+` ``, run `streamlit run app/app_streamlit.py`, click the printed `localhost` link.
   - **Simple Browser**: `Cmd+Shift+P` → `Simple Browser: Show` → `http://localhost:8501` — renders inside a VS Code tab.
   - **Debug panel**: open the Run-and-Debug pane, select "Streamlit: app_streamlit.py" (configured in `.vscode/launch.json`), press F5.
4. **Iterate** on visuals in `app/app_streamlit.py` — Streamlit hot-reloads on save.

---

## Monthly data refresh

The `.github/workflows/monthly_refresh.yml` GitHub Action runs at 14:00 UTC on the 5th of every month:

1. Calls `python -m src.ercot.fetcher`
2. Downloads the newest GIS xlsx
3. Saves it to `data/ercot/snapshots/YYYY-MM.xlsx`
4. Updates the `data/ercot/latest.xlsx` pointer
5. Commits and pushes if the file is new

You can trigger it manually from the **Actions** tab of your GitHub repo.

---

## Optional: substation-level coordinates

The default geocoder uses Texas county centroids (auto-downloaded from US Census on first run). For substation-precision pins:

1. Download the [HIFLD Electric Substations CSV](https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-substations).
2. Save as `data/geo/hifld_substations.csv` with columns `NAME, LATITUDE, LONGITUDE, STATE`.
3. The next app launch will auto-detect and use it.

---

## Limitations to flag for users

- LBNL training data ends 2024; ERCOT post-2019 dynamics are underrepresented (your README documents this). Layer 2 exists specifically to reality-check Layer 1 against the live queue.
- ERCOT GIS Report excludes Inactive (INA) projects; withdrawal information is in the separate `Inactive Projects` and `Cancellation Update` sheets (not yet wired into Layer 2 — feature for later).
- `service_type` in the LBNL model is the strongest completion signal but doesn't apply to ERCOT — defaulted as documented above.
- Capacity factor is a Layer-2 input only; it is not a feature in the trained LBNL model.
