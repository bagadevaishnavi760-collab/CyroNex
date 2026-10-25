
# CryoNex — Quantum Materials Intelligence

### Discover the Critical Frontier.

CryoNex is a machine learning-powered web application for exploring superconductivity data and estimating the critical temperature (\(T_c\)) of superconducting materials using an existing trained XGBoost regression model.

<p align="center">
  <a href="https://cyro-nex-rqob.vercel.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-CryoNex-00A86B?style=for-the-badge" alt="Live Demo" />
  </a>
  <a href="https://github.com/bagadevaishnavi760-collab/CyroNex">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository" />
  </a>
</p>

> **Disclaimer:** Predictions are computational estimates based on engineered material descriptors. They are not experimental measurements or confirmation of superconductivity.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
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

The critical temperature (\(T_c\)) is the temperature below which a material exhibits superconducting behavior under specified conditions. Estimating this property from material descriptors can help researchers explore relationships between material composition and superconducting behavior.

CryoNex combines a React and TypeScript frontend, a Flask REST API, a supplied superconductivity dataset, and an existing trained XGBoost regression model.

The application provides an interactive interface to explore dataset statistics, inspect material records, view model feature importances, and estimate critical temperatures.

### Live Application

**Website:** https://cyro-nex-rqob.vercel.app/

**Backend API:** https://cyronex.onrender.com

---

## Key Features

- **Critical Temperature Prediction:** Estimate superconducting critical temperature in Kelvin using the saved XGBoost model.
- **81 Engineered Features:** Work with the numerical material descriptors expected by the model.
- **Dataset Explorer:** Inspect dataset records, target-temperature distributions, and descriptive statistics.
- **Model Insights:** Explore feature importances exposed by the trained model.
- **Sample Material Records:** Load existing dataset rows as starting points for predictions.
- **Prediction History:** Review previous predictions stored in the browser's local storage.
- **REST API:** Access prediction, dataset, and model-insight functionality through Flask endpoints.
- **Interactive Web Interface:** Use a modern React, TypeScript, and Vite frontend.
- **Separate Frontend and Backend:** Deploy the interface and machine learning API independently.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, Flask |
| Machine Learning | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| Model Artifacts | Existing XGBoost JSON model and `model.pkl` |
| API Communication | REST API, JSON |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
| Version Control | Git, GitHub |

---

## System Architecture

CryoNex uses a client-server architecture in which the frontend communicates with the Flask backend over HTTP.

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
| Prediction UI                    |
| Dataset Explorer                 |
| Model Insights                   |
| Prediction History               |
+----------------+-----------------+
                 |
                 | REST API / JSON
                 v
+----------------------------------+
|           Flask API              |
|                                  |
| Input Validation                 |
| Dataset Operations               |
| Prediction Endpoint              |
| Model Insights Endpoint          |
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

### Request Flow

1. The user enters material descriptor values in the frontend.
2. The frontend sends a request to the Flask API.
3. The backend validates the input against the expected feature schema.
4. The existing trained model generates a critical-temperature estimate.
5. The API returns the prediction in JSON format.
6. The frontend displays the result to the user.

The saved model is used for inference; normal prediction requests do not require model retraining.

---

## Project Structure

The repository contains the frontend, backend, model artifacts, and dataset resources.

```text
CyroNex/
├── backend/
│   ├── app.py
│   └── requirements.txt
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

*Note: The tree is illustrative and may omit additional files or directories. Verify actual filenames and paths in the repository before using the commands below.*

---

## Dataset

CryoNex uses the project's supplied superconductivity dataset, containing engineered numerical descriptors and recorded critical-temperature values.

The feature set describes material properties derived from elemental characteristics, including:

- Number of elements
- Atomic mass
- First ionization energy
- Atomic radius
- Density
- Electron affinity
- Heat of fusion
- Thermal conductivity
- Valence

Statistical descriptors include means, weighted means, geometric means, entropy, ranges, and standard deviations.

### Input Requirements

The saved model expects **all 81 engineered numerical features** using the exact feature names and ordering defined by the backend.

A chemical formula alone is not sufficient to generate a prediction unless an appropriate feature-generation pipeline is implemented.

The application can load sample records from the supplied dataset to make it easier to test the prediction workflow.

---

## Machine Learning Approach

CryoNex uses an existing trained XGBoost regression model to estimate critical temperature.

### Workflow

1. **Data Preparation:** Load the supplied superconductivity dataset and engineered numerical descriptors.
2. **Feature Representation:** Use the 81 material descriptors expected by the model.
3. **Model Loading:** Load the existing saved model artifact.
4. **Inference:** Generate a numerical critical-temperature estimate.
5. **API Response:** Return the prediction in Kelvin.
6. **Visualization:** Display predictions and dataset information through the frontend.

### Model Preservation

The application uses the supplied trained model artifacts. Model retraining is not required for normal predictions.

The existing model files should be preserved unless a separate, intentional model-training and validation workflow is undertaken.

---

## Model Evaluation

The original notebook reports the following holdout evaluation results for an XGBoost regression model using an 80/20 train-test split with `random_state=42`.

| Metric | Notebook-Reported Result |
|---|---:|
| R² Score | 0.9264 |
| Mean Absolute Error (MAE) | 5.5566 K |
| Root Mean Squared Error (RMSE) | 9.3099 K |

### Interpretation

- **R² Score:** Indicates how much variation in the target is explained by the model on the evaluated split.
- **MAE:** Represents the mean absolute difference between predictions and recorded target values.
- **RMSE:** Penalizes larger prediction errors more strongly than MAE.

These values are reported by the original notebook. They are not an independent evaluation of the currently deployed application and do not guarantee equivalent performance on new or unseen materials.

Feature importance indicates patterns learned by the model; it does not establish causation.

---

## Getting Started

Follow these instructions to run the project locally.

### Prerequisites

Install the following:

- Python 3.10 or newer
- Node.js 20.19+ or 22.12+
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/bagadevaishnavi760-collab/CyroNex.git
cd CyroNex
```

### 2. Set Up the Backend

From the repository root, create a Python virtual environment.

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, consult the Python documentation or activate the environment using the appropriate command for your shell.

### 3. Start the Backend

Run the Flask application from the repository root:

```powershell
python backend\app.py
```

The API is expected to run at:

```text
http://127.0.0.1:5000
```

Keep the backend terminal open while using the application.

If the server reports missing model files or dataset files, verify that the required artifacts exist at the paths expected by `backend/app.py`.

### 4. Configure the Frontend

Open a second terminal:

```powershell
cd frontend
```

If `.env.example` exists and `.env.local` does not already exist, create the local environment file:

```powershell
Copy-Item .env.example .env.local
```

Set the backend URL in `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

If `.env.local` already exists, edit it instead of overwriting your current configuration.

### 5. Install Frontend Dependencies

From the `frontend/` directory:

```powershell
npm install
```

### 6. Run the Frontend

```powershell
npm run dev
```

Open the local URL printed in the terminal. Vite commonly uses:

```text
http://127.0.0.1:5173
```

The backend and frontend must both be running for live predictions to work locally.

---

## API Reference

The Flask backend exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status and dataset availability |
| GET | `/features` | Retrieve model feature names, feature groups, and target name |
| POST | `/predict` | Predict critical temperature from the required numerical descriptors |
| GET | `/sample?index=0` | Retrieve a dataset sample and recorded target |
| GET | `/dataset/summary` | Retrieve dataset size, missing-cell count, target statistics, and descriptor statistics |
| GET | `/dataset/records?page=1&page_size=20&q=` | Retrieve paginated dataset records and optionally filter results |
| GET | `/model/insights` | Retrieve feature importances from the saved model |

### Example: Check API Availability

Open the following endpoint in a browser:

```text
https://cyronex.onrender.com/features
```

A successful response should return JSON describing the model's features.

### Prediction Request

The `POST /predict` endpoint accepts a JSON object containing the required feature names and finite numerical values.

The request must match the exact schema expected by the backend.

Use the `/features` endpoint to retrieve the canonical feature list.

### Prediction Response

A successful prediction returns the estimated critical temperature in Kelvin under the field:

```text
predicted_critical_temperature_K
```

The exact request structure, additional response fields, and error formats should be verified against the implementation in `backend/app.py`.

---

## Configuration

The frontend uses the `VITE_API_BASE_URL` environment variable to determine the backend API origin.

### Local Development

Set the following in `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

### Production Deployment

The production frontend uses the deployed Flask API:

```env
VITE_API_BASE_URL=https://cyronex.onrender.com
```

Configure the variable in the frontend hosting provider's environment-variable settings and ensure it applies to the environment being deployed.

Vite environment variables are incorporated into the frontend build. After changing them, create a new deployment.

### CORS Configuration

The Flask backend must allow requests from the frontend's origin.

The current production frontend origin is:

```text
https://cyro-nex-rqob.vercel.app
```

If the production domain changes, update the backend's CORS allowlist accordingly. Preview deployments may require additional configuration if they also need to access the API.

Do not expose passwords, private keys, or other secrets through frontend environment variables.

---

## Production Deployment

CryoNex is deployed as two separate services.

| Component | Hosting Provider | Purpose |
|---|---|---|
| Frontend | Vercel | Hosts the web interface |
| Backend | Render | Hosts the Flask API and model inference |

### Live Links

- **Frontend:** https://cyro-nex-rqob.vercel.app/
- **Backend API:** https://cyronex.onrender.com
- **GitHub Repository:** https://github.com/bagadevaishnavi760-collab/CyroNex

### Backend Deployment

When deploying the Flask backend:

1. Include `backend/app.py` and `backend/requirements.txt`.
2. Ensure the required trained model artifacts and dataset files are available at the paths expected by the application.
3. Install the backend dependencies.
4. Configure CORS for the production frontend origin.
5. Use a production WSGI server such as Gunicorn.
6. Verify the deployed API endpoints.

If the Flask application object is named `app` in `backend/app.py`, a typical Gunicorn command when run from the `backend/` directory is:

```bash
gunicorn app:app
```

Follow the hosting provider's requirements for binding to the assigned port and configuring the working directory.

### Frontend Deployment

When deploying the frontend:

1. Set the project root to `frontend/` if required by the hosting configuration.
2. Install dependencies.
3. Set `VITE_API_BASE_URL` to the public backend URL.
4. Build the application using `npm run build`.
5. Deploy the generated `dist/` directory.
6. Test predictions and dataset endpoints on the live website.

A static frontend host does not replace the Flask API. Both services must be available for live predictions.

---

## Testing and Validation

The project may include a model test script at the repository root.

Run it from the root directory if the file is present:

```powershell
python test_model.py
```

Build the frontend to check for TypeScript and production-build errors:

```powershell
cd frontend
npm run build
```

### Recommended Validation Checklist

Before releasing changes, verify that:

- [ ] The backend starts and loads the saved model.
- [ ] Required dataset files are found.
- [ ] Model feature names and ordering match the input schema.
- [ ] Valid prediction requests return numeric estimates.
- [ ] Invalid or incomplete inputs are handled appropriately.
- [ ] Dataset summary and record endpoints return valid data.
- [ ] Model insights load correctly.
- [ ] The frontend communicates with the deployed backend.
- [ ] The production build completes successfully.
- [ ] The live website works after a page refresh.

---

## Limitations

- Predictions depend on the quality and coverage of the training data.
- A predicted critical temperature is not experimental confirmation of superconductivity.
- The model requires the expected engineered numerical descriptors.
- A chemical formula cannot be used directly unless feature generation is implemented.
- Notebook-reported evaluation metrics may not represent performance on materials outside the original data distribution.
- Feature importance describes model behavior and does not establish causal relationships.
- Prediction history is stored in the current browser's local storage, not a centralized database.
- Local storage history is browser-specific and may be lost when browser data is cleared.

---

## Future Enhancements

Potential improvements include:

- Automated feature generation from validated material-composition inputs.
- Independent evaluation on additional superconductivity datasets.
- Comparison of regression algorithms.
- Enhanced dataset visualization and filtering.
- Downloadable prediction reports.
- Reproducible model evaluation and experiment tracking.
- Automated API and frontend tests.
- Continuous integration and deployment checks.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Run the relevant tests.
5. Submit a pull request describing your changes.

---

## License

No license has been specified yet. Until a license is added, do not assume that the repository grants permission to redistribute or reuse its code.

If you intend to release the project as open source, add a suitable `LICENSE` file and ensure that the licenses of the dataset, dependencies, and other included assets permit your intended use.

---

<p align="center">
  <strong>CryoNex — Quantum Materials Intelligence</strong><br/>
  <em>Discover the Critical Frontier.</em>
</p>
