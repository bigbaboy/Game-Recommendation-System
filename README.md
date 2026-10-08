# Content-Based Game Recommendation System

A Streamlit academic prototype that recommends games similar to a selected title using **game attributes and cosine nearest neighbors**.

## Recommendation task

A user selects a game from the dataset and receives up to five similar titles. Recommendations use item characteristics rather than a user–item ratings matrix, so this implementation is **content-based**, not collaborative filtering.

## Implementation

1. Load `R7data.csv`.
2. Aggregate repeated rows into game-level records and combine genres, languages and platforms.
3. Multi-hot encode these categorical attributes.
4. Standardize the game price.
5. Fit `NearestNeighbors` with cosine distance.
6. Retrieve neighboring games and display names, review text and publishers in Streamlit.

The implementation requests one extra neighbor and drops the first returned result to exclude the selected game. This assumes the selected game is first; identical feature vectors are an edge case to validate.

## Data fields

The source includes `ID_GAME`, `TEN_GAME`, `NGAY_PHAT_HANH`, `GIA`, `DANH_GIA`, `TY_LE_DANH_GIA_TICH_CUC`, `SO_LUONG_DANH_GIA`, `NHA_PHAT_HANH`, `TEN_THE_LOAI`, `TEN_NGON_NGU` and `TEN_NEN_TANG`.

The price and encoded genre/language/platform features drive similarity. Publisher and review text are displayed but excluded from the fitted feature matrix. A crawler is not included in this repository.

## Run locally

```bash
git clone https://github.com/bigbaboy/Game-Recommendation-System.git
cd Game-Recommendation-System
python -m venv .venv
```

Activate `.venv`, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run R7_Group8.py
```

Keep `R7data.csv` in the working directory. Python dependencies are pandas, NumPy, scikit-learn and Streamlit, with fsspec also listed in the repository. A fresh runtime installation was not tested during the documentation update.

## Demo walkthrough

Select a game, click the recommendation button and inspect the returned titles. Try games with different genres or platform support to compare the neighborhoods.

## Limitations

- No held-out relevance labels or recommendation-quality metric is published here.
- The method does not learn personal preferences, clicks or ratings.
- Price scaling and categorical feature weights influence the nearest neighbors.
- Duplicate feature vectors, missing data and datasets with too few games need explicit checks.

## Next steps

Add deterministic self-exclusion by game identifier, expose similarity explanations and compare against a simple popularity baseline. Document data provenance and team contributions before presenting this as an individually completed project.
