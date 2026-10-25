
# CryoNex — Quantum Materials Intelligence

### Discover the Critical Frontier.

CryoNex is a machine-learning-powered web application for exploring superconductivity data and estimating the critical temperature (\(T_c\)) of superconducting materials using a trained XGBoost regression model.

<p align="center">
  <a href="https://cyro-nex-rqob.vercel.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-CryoNex-00A86B?style=for-the-badge" alt="Live Demo" />
  </a>
  <a href="https://github.com/bagadevaishnavi760-collab/CyroNex">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository" />
  </a>
</p>

> **Disclaimer:** Predictions are computational estimates. They are not experimental measurements, proof of superconductivity, or confirmation that a candidate can be synthesized.

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Dataset and Features](#dataset-and-features)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Evaluation](#model-evaluation)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)
- [Testing and Validation](#testing-and-validation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The critical temperature (\(T_c\)) is the temperature below which a material exhibits superconducting behavior under specified conditions. Estimating this property from material descriptors can help researchers prioritize candidates for further investigation.

CryoNex combines a React and TypeScript frontend, a Flask REST API, a superconductivity dataset, and an existing trained XGBoost regression model.

The main goal is to support **material discovery**, not simply predict the target value of a dataset record when that value is already known.

Users can use existing records to understand known materials or as starting templates for candidate profiles. By modifying feature values or entering a complete set of descriptors, users can explore model estimates for candidate profiles whose actual critical temperatures are unknown.

### Live Application

- **Frontend:** https://cyro-nex-rqob.vercel.app/
- **Backend API:** https://cyronex.onrender.com
- **GitHub Repository:** https://github.com/bagadevaishnavi760-collab/CyroNex

---

## Project Goals

CryoNex aims to:

1. Explore superconductivity data through an interactive interface.
2. Estimate critical temperatures for candidate feature profiles.
3. Compare candidates using model-generated estimates.
4. Explain model behavior using genuine model-derived feature importance or attribution methods, where supported.
5. Identify candidate inputs outside observed feature ranges.
6. Distinguish recorded dataset values from predicted values.
7. Preserve the existing trained model during normal inference.

---

## Key Features

The availability of individual features depends on the implementation in the current repository.

- **Material Discovery:** Create candidate profiles using editable dataset templates or manually entered feature values.
- **Candidate Comparison:** Compare candidate profiles by predicted critical temperature, where implemented.
- **81 Numerical Features:** Use the complete feature representation expected by the trained model.
- **Dataset Explorer:** Inspect source records and recorded critical-temperature values.
- **Model Insights:** View global feature importance when available through the backend.
- **Candidate Explanations:** View SHAP or TreeSHAP explanations if supported by the installed model and backend.
- **Input Range Checks:** Identify feature values outside the observed dataset ranges, where implemented.
- **Sample Records:** Load existing records as starting points.
- **Prediction History:** Retain predictions in browser local storage if this functionality is enabled.
- **REST API:** Access model prediction and dataset operations through Flask.
- **Independent Deployment:** Host the frontend on Vercel and backend on Render.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, Flask |
| Machine Learning | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| Model Artifacts | Existing XGBoost JSON model and `model.pkl` |
| Communication | REST API, JSON |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
| Version Control | Git, GitHub |

---

## System Architecture

CryoNex uses a client-server architecture. The frontend sends requests to the Flask backend, which validates the inputs and uses the saved model to generate predictions.

```text
+----------------------------------+
|          User / Browser          |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|       React + TypeScript         |
|          Vite Frontend           |
|                                  |
| Material Discovery               |
| Candidate Comparison             |
| Dataset Explorer                 |
| Model Insights                   |
| Prediction History (if enabled) |
+----------------+-----------------+
                 |
                 | REST API / JSON
                 v
+----------------------------------+
|             Flask API            |
|                                  |
| Input Validation                 |
| Dataset Operations               |
| Prediction Endpoint              |
| Model Insights / Explanations    |
+----------------+-----------------+
                 |
          +------+------+
          |             |
          v             v
+----------------+  +----------------+
| Saved XGBoost  |  | Superconduct.  |
| Model Artifact |  | Dataset / CSV  |
+----------------+  +----------------+
```

### Prediction Flow

1. The user creates a candidate profile or starts from an existing dataset row.
2. Required numerical features are validated.
3. The input values are ordered according to the model's canonical feature schema.
4. Only model features are sent to the backend. The recorded target is excluded.
5. The existing XGBoost model estimates the candidate's critical temperature.
6. The frontend displays the predicted value separately from any recorded target.
7. Where implemented, users can compare candidates and inspect model-derived explanations or input-range warnings.

Normal predictions do not require retraining the model.

---

## Project Structure

The following is an illustrative structure. Verify actual filenames and locations in the repository.

```text
CyroNex/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── .env.example
│   └── ...
├── superconductor_model.json
├── model.pkl
├── Super-Conductor-Critical-Temperature-Prediction-main/
│   ├── CriticalTemperature.csv
│   └── Regression_CriticalTemperature .ipynb
├── superconductivty+data/
├── test_model.py
├── README.md
└── .gitignore
```

Some files or directories may differ between repository versions.

---

## Dataset and Features

CryoNex uses the supplied superconductivity dataset containing engineered numerical descriptors and recorded critical-temperature values.

Examples of material-property descriptor categories include:

- Number of elements
- Atomic mass
- First ionization energy
- Atomic radius
- Density
- Electron affinity
- Heat of fusion
- Thermal conductivity
- Valence

Statistical descriptors may include means, weighted means, geometric means, entropy, ranges, and standard deviations. The exact feature schema returned by the backend is authoritative.

### Input Requirements

The trained model expects **81 numerical features** in the exact names and order required by the backend.

- A chemical formula alone is insufficient unless a validated feature-generation pipeline is available.
- Missing values must not be replaced with arbitrary defaults.
- Input values must be numeric and finite.
- The recorded target (`critical_temp` or equivalent) must never be passed as a model input.
- Use the backend feature endpoint to retrieve the canonical feature schema when available.
- When a dataset record is edited, label it as a modified candidate, not as a new experimentally verified material.

### Recorded vs. Predicted Critical Temperature

| Value | Meaning |
|---|---|
| Recorded \(T_c\) | Target value stored in a dataset record |
| Predicted \(T_c\) | Estimate generated by the trained model |
| Experimental confirmation | Evidence from an experiment; not provided by a model prediction |

---

## Machine Learning Approach

CryoNex uses an existing trained XGBoost regression model to estimate critical temperature.

### Inference Workflow

1. Load the required numerical descriptors.
2. Validate the input values.
3. Preserve the model's expected feature names and ordering.
4. Use the saved model to generate an estimate.
5. Return the result through the Flask API.
6. Display the estimate and any supported diagnostics in the frontend.

### Model Preservation

The supplied model artifacts are used for inference. Normal prediction requests do not retrain the model.

Do not replace, overwrite, or retrain the model as part of a frontend-only change. Any future model training should be handled separately with documented preprocessing, validation, and evaluation.

### Explainability

Global feature importance summarizes features the model relies on across a dataset. SHAP or TreeSHAP values can provide an attribution for an individual prediction relative to a baseline.

- Global feature importance is not an explanation of every individual prediction.
- Feature importance does not prove causation.
- Explanations must be calculated from the actual model and correctly ordered inputs.
- Do not fabricate feature contributions if explainability is unavailable.

### Candidate Comparison

Candidate ranking can help prioritize profiles for further study. It does not establish that a candidate is chemically stable, synthesizable, physically realizable, or experimentally superconducting.

All candidates should be evaluated using the same model version and preprocessing rules.

---

## Model Evaluation

The original notebook reports the following holdout evaluation results for an XGBoost regression model using an 80/20 train-test split with `random_state=42`.

| Metric | Notebook-reported result |
|---|---:|
| R² Score | 0.9264 |
| Mean Absolute Error (MAE) | 5.5566 K |
| Root Mean Squared Error (RMSE) | 9.3099 K |

### Interpretation

- **R²:** Measures how much target variation is explained by the model on the evaluated split.
- **MAE:** The average absolute difference between predicted and recorded target values.
- **RMSE:** A regression error metric that penalizes larger errors more strongly than MAE.

These values come from the original notebook. They are not an independent evaluation of the currently deployed API and do not guarantee similar performance on new materials or inputs outside the training distribution.

For robust evaluation, use held-out data, check for data leakage, document preprocessing and splitting, and evaluate on suitable external datasets where available.

---

## Getting Started

### Prerequisites

Install:

- Python 3.10 or newer, subject to dependency compatibility
- Node.js 20.19+ or 22.12+
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/bagadevaishnavi760-collab/CyroNex.git
cd CyroNex
```

### 2. Set Up the Backend

From the repository root, create a virtual environment.

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, use the activation method appropriate for your shell and local security policy.

### 3. Start the Backend

From the repository root:

```powershell
python backend\app.py
```

The local API is expected to run at:

```text
http://127.0.0.1:5000
```

Keep this terminal open. If model or dataset files are missing, check the paths expected by `backend/app.py`.

### 4. Configure the Frontend

Open a second terminal:

```powershell
cd frontend
```

If `.env.example` exists and `.env.local` does not, create it:

```powershell
Copy-Item .env.example .env.local
```

Set the backend URL in `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

If `.env.local` already exists, edit it rather than overwriting your configuration.

### 5. Install Frontend Dependencies

From `frontend/`:

```powershell
npm install
```

### 6. Run the Frontend

```powershell
npm run dev
```

Open the local URL printed by Vite. It commonly uses:

```text
http://127.0.0.1:5173
```

Both frontend and backend must be running for local API-backed features to work.

---

## API Reference

The following endpoints are documented for this project. Verify the current implementation in `backend/app.py` before relying on exact response fields or error formats.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status and dataset availability |
| GET | `/features` | Retrieve model feature names and schema information |
| POST | `/predict` | Predict critical temperature from numerical descriptors |
| GET | `/sample?index=0` | Retrieve a dataset sample and its recorded target |
| GET | `/dataset/summary` | Retrieve dataset size and summary statistics |
| GET | `/dataset/records?page=1&page_size=20&q=` | Retrieve paginated records with optional filtering |
| GET | `/model/insights` | Retrieve global feature importance, if implemented |
| POST | `/model/explain` | Retrieve a candidate SHAP/TreeSHAP explanation, if implemented |

### Check API Availability

Open:

```text
https://cyronex.onrender.com/features
```

A successful response should return JSON describing the model's input schema.

### Prediction Request

`POST /predict` accepts a JSON object containing the required feature names and finite numerical values. The precise request schema is determined by the backend implementation.

Use `/features` to obtain the canonical feature list where available. Do not guess feature names or ordering.

The request must include the required model features and exclude the recorded target column. A source record can be used to initialize a candidate template, but its recorded target must not be sent as an input.

### Prediction Response

The project documentation identifies the predicted temperature field as:

```text
predicted_critical_temperature_K
```

Verify the exact response keys against the current backend implementation.

### Input Errors

Incomplete, malformed, non-numeric, or non-finite inputs should be handled with clear error messages. The application should not silently substitute arbitrary values or present invalid estimates.

---

## Configuration

The frontend uses `VITE_API_BASE_URL` to determine the backend API origin.

### Local Development

Set in `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

### Production

Set the frontend environment variable to:

```env
VITE_API_BASE_URL=https://cyronex.onrender.com
```

Configure this value in the frontend hosting provider's environment settings. Since Vite variables are incorporated into the frontend build, changing the value requires a new build and deployment.

### CORS

The backend must allow requests from the production frontend origin:

```text
https://cyro-nex-rqob.vercel.app
```

Update the backend CORS configuration if the frontend domain changes. Preview deployments may need separate configuration.

Never expose passwords, private keys, or other secrets through frontend environment variables. Variables prefixed with `VITE_` are exposed to the client.

---

## Production Deployment

CryoNex uses two independently deployed services.

| Component | Hosting Provider | Purpose |
|---|---|---|
| Frontend | Vercel | Hosts the web interface |
| Backend | Render | Hosts the Flask API and model inference |

### Live Links

- **Frontend:** https://cyro-nex-rqob.vercel.app/
- **Backend API:** https://cyronex.onrender.com
- **GitHub:** https://github.com/bagadevaishnavi760-collab/CyroNex

### Backend Deployment

1. Include `backend/app.py` and `backend/requirements.txt`.
2. Ensure required model artifacts and dataset files exist at the paths expected by the application.
3. Install backend dependencies.
4. Configure CORS for the production frontend.
5. Configure a production WSGI server, such as Gunicorn, if supported by the hosting provider.
6. Verify the deployed endpoints and test a valid prediction.

If the Flask application object is named `app` in `backend/app.py`, a typical Gunicorn command from the `backend/` directory is:

```bash
gunicorn app:app
```

Follow the hosting provider's requirements for working directory, port binding, and start command.

### Frontend Deployment

1. Set the project root to `frontend/` if required.
2. Install dependencies.
3. Set `VITE_API_BASE_URL` to the deployed backend URL.
4. Build with `npm run build`.
5. Deploy the generated `dist/` directory.
6. Test the live application against the deployed API.

The static frontend does not replace the Flask API. Both services must be available for API-backed features.

---

## Testing and Validation

Run only commands supported by the test files and scripts in your current checkout.

### Backend Unit Tests

If tests exist under `backend/tests/`, run from the repository root:

```powershell
python -m unittest discover -s backend\tests -v
```

### Model Test Script

If `test_model.py` exists in the repository root:

```powershell
python test_model.py
```

### Frontend Test Scripts

From `frontend/`, inspect available scripts:

```powershell
npm run
```

If an `npm test` script exists:

```powershell
npm test
```

### Production Build

From `frontend/`:

```powershell
npm run build
```

### Validation Checklist

- [ ] Backend starts and loads the saved model.
- [ ] Required dataset files are found.
- [ ] Model feature names and ordering are preserved.
- [ ] Only the expected input features are passed to the model.
- [ ] Recorded target \(T_c\) is excluded from model inputs.
- [ ] Valid requests return finite numerical predictions.
- [ ] Missing or invalid inputs are handled clearly.
- [ ] Dataset endpoints return valid data.
- [ ] Model insights and explanations work when supported.
- [ ] Candidate comparisons use consistent model inputs.
- [ ] Range warnings are not presented as calibrated confidence scores.
- [ ] Frontend and backend communicate correctly in production.
- [ ] The frontend production build succeeds.
- [ ] The deployed website works after refresh.

---

## Limitations

- Predictions depend on the quality and coverage of the training data.
- A prediction is not experimental confirmation of superconductivity.
- The model requires the expected engineered numerical descriptors.
- A chemical formula cannot be used directly without a validated feature-generation pipeline.
- Values within individual feature ranges can still form an unfamiliar combination of inputs.
- Notebook-reported metrics may not represent performance on new materials or the deployed model version.
- Feature importance and SHAP values describe model behavior, not causal physical mechanisms.
- Candidate rankings do not establish chemical stability, synthesizability, or experimental feasibility.
- Browser-local prediction history, if enabled, is browser-specific and may be lost when browser data is cleared.
- Optional features described here must be verified against the current code before being represented as implemented.

---

## Future Enhancements

Potential improvements include:

- Validated feature generation from material composition.
- External-dataset evaluation and improved generalization checks.
- Uncertainty estimation using an appropriate validation method.
- More robust out-of-distribution detection.
- Downloadable candidate comparison reports.
- Improved dataset filtering and visualization.
- Reproducible model evaluation and experiment tracking.
- Automated frontend and API tests.
- Continuous integration and deployment checks.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Run the relevant tests.
5. Submit a pull request describing your changes and test results.

Please preserve existing model artifacts and API contracts unless a deliberate, documented change is required.

---

## License

No license is specified in this README. Unless a license is added to the repository, do not assume permission to redistribute or reuse the code.

If you intend to release the project as open source, add an appropriate `LICENSE` file and confirm that the dataset, model artifacts, dependencies, and other assets permit your intended use.

---

<p align="center">
  <strong>CryoNex — Quantum Materials Intelligence</strong><br/>
  <em>Discover the Critical Frontier.</em>
</p>
