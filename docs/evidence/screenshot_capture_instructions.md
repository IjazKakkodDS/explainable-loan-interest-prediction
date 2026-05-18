# Screenshot Capture Instructions
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-18
Phase: L1

Screenshots of the Streamlit interface could not be captured automatically during this elevation pass because the Streamlit app requires a live browser session. This document provides exact instructions for capturing them manually.

---

## Target Screenshots

| Filename | What to capture |
|---|---|
| docs/evidence/screenshots/streamlit-interface.png | Full Streamlit app on first load: sidebar visible, default KPI metrics visible |
| docs/evidence/screenshots/prediction-result.png | After entering custom inputs: predicted rate KPI + loan purpose + annual income visible |
| docs/evidence/screenshots/shap-review-flow.png | Scroll to show SHAP waterfall and bar plots side by side |

---

## Step 1: Start the Streamlit App

From the project root:

```
pip install -r requirements.txt
streamlit run app/app.py
```

The app will open at http://localhost:8501 in your default browser.

---

## Step 2: Capture streamlit-interface.png

1. Allow the app to load fully (sidebar appears, KPIs render)
2. Leave sidebar inputs at default values
3. Capture the full browser window (not just the viewport) at 1280 x 900 or wider
4. Save to: docs/evidence/screenshots/streamlit-interface.png

Recommended tool (Windows): Win + Shift + S (Snipping Tool), or use browser Dev Tools > Device Toolbar

---

## Step 3: Capture prediction-result.png

1. In the sidebar, set:
   - Loan Amount: 25,000
   - Installment: 600
   - Annual Income: 75,000
   - Revolving Credit Utilization: 65%
   - Total Received Interest: 2,000
   - Inquiries Last 6 Months: 3
   - Loan Term: 60 months
   - Purpose: Debt Consolidation
   - Verification Status: Verified
2. The KPI section should update showing the predicted rate
3. Capture the top section: title, timestamp, and three KPI metric cards
4. Save to: docs/evidence/screenshots/prediction-result.png

---

## Step 4: Capture shap-review-flow.png

1. With the same inputs as Step 3 active
2. Scroll down to the "Feature Contribution Visuals" section
3. Both SHAP Waterfall and SHAP Bar plots should be visible side by side
4. Capture this two-column section
5. Save to: docs/evidence/screenshots/shap-review-flow.png

---

## Step 5: Create the Screenshots Directory

Before saving, create the directory:

```
mkdir docs\evidence\screenshots
```

---

## Optional: Playwright Automated Capture

If Playwright is available, the following script template can automate capture:

```python
from playwright.sync_api import sync_playwright
import time, os

os.makedirs("docs/evidence/screenshots", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://localhost:8501")
    time.sleep(5)  # wait for Streamlit to render
    page.screenshot(path="docs/evidence/screenshots/streamlit-interface.png", full_page=False)
    browser.close()
```

Playwright must be installed separately:
```
pip install playwright
playwright install chromium
```

The Streamlit app must be running at localhost:8501 before this script is executed.

---

## Note on Deployment Screenshots

If a live Streamlit Cloud or Render deployment is available, screenshots can be captured from the public URL instead of localhost. The URL must be confirmed live before capturing.

At the time of this evidence audit, live deployment status was not independently verified.
