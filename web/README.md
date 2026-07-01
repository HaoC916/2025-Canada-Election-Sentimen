# Election Sentiment Explorer

**Live:** https://haoc916.github.io/2025-Canada-Election-Sentimen/

A static React app that visualizes this project's real output: weekly/daily Reddit
sentiment per federal party (VADER + RoBERTa transformer), plotted against real
polling averages through the 2025-04-27 election, plus the Pearson correlation
between them at increasing lag.

All data in `src/data/*.json` is generated from the pipeline's actual output
(`data/sentiment_*_updated_key`, `data/polling_averages.txt`) — nothing here is
simulated. See the repo root README for how that data was produced.

## Run locally

```bash
npm install
npm run dev
```

## Build / deploy

`npm run build` outputs `dist/`. Pushed to `main` under `web/**`, this deploys
automatically to GitHub Pages via `.github/workflows/deploy-web.yml`.
