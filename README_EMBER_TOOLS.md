# EMBER Tool System

Alberta Wildfire AI Assistant with real tool calling support.

## Files

| File | Purpose |
|------|---------|
| `ember_tools_chat.html` | Frontend chat interface — upload to GitHub Pages |
| `tool_server.py` | FastAPI tool server — runs on your GPU server |
| `requirements.txt` | Python dependencies for the tool server |

---

## Setup

### 1. Install dependencies on your GPU server

```bash
pip install -r requirements.txt
```

### 2. Start the tool server

```bash
python tool_server.py
```

This starts the tool server on port 5001.
Keep it running alongside your Ollama instance.

### 3. Upload the frontend to GitHub

Upload `ember_tools_chat.html` to your GitHub repo.
It will be live at:
`https://emmagrace0130.github.io/EMBER-Alberta-Wildfires/ember_tools_chat.html`

---

## How to use

1. Open the GitHub Pages URL
2. Enter your Ollama server URL and credentials
3. Enter your tool server URL: `http://your-server.com:5001`
4. Click Connect

If you leave the tool server URL blank, the LLM will still work
but will answer from context only without calling real tools.

---

## Available Tools

| Tool | What it does |
|------|-------------|
| `assess_fire_risk` | Calculates escalation risk from fire conditions using SHAP weights |
| `get_regional_profile` | Returns historical risk data for any Alberta forest region |
| `get_model_performance` | Returns CV metrics for RF, LR, XGBoost, Isolation Forest |
| `get_shap_importance` | Returns all 12 SHAP feature importance values |
| `get_season_stats` | Returns statistics for any fire season 2006-2024 |

---

## How tool calling works

1. You send a message to the frontend
2. Frontend sends message + tool definitions to Ollama
3. Ollama decides whether to call a tool
4. If yes — frontend calls the tool server directly
5. Tool result is sent back to Ollama
6. Ollama generates a final response incorporating the result

---

## Research context

- Dataset: 26,551 Alberta FPA fires, 2006-2024
- Target: Large fire (Class D/E, >40 ha) — 3.3% of all fires
- LR AUPRC: 0.584 ± 0.041 (best precision)
- RF AUROC: 0.957 ± 0.008 (best discrimination)
- XGBoost AUPRC: 0.567 ± 0.042
- Isolation Forest AUPRC: 0.264 (unsupervised, 8x above baseline)
- Random baseline AUPRC: 0.033

IE 565 — University of Tennessee, Knoxville
Emma Wilhoit, Dr. Maryam Mofrad (PhD), Dr. Anahita Khojandi (PhD)
