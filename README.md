
# CryoNex — Quantum Materials Intelligence

**Discover the Critical Frontier.**

CryoNex is a machine learning-powered research interface for exploring superconductivity data and estimating the critical temperature (Tc) of superconducting materials. It combines a saved XGBoost regression model, a Flask REST API, and a React + TypeScript frontend to make material-property exploration and prediction more accessible.

Predictions are computational estimates based on engineered material descriptors, not experimental measurements or confirmation of superconductivity.

## Overview

The critical temperature is the temperature below which a material exhibits superconducting behavior under specified conditions. Estimating this property from material descriptors can help researchers investigate relationships between composition and superconducting behavior.

CryoNex provides an interactive environment to explore the supplied dataset, inspect model insights, and generate predictions using the project's existing trained model.

## Key Features

- **Critical Temperature Prediction:** Estimate Tc in Kelvin using the saved XGBoost regression model.
- **81 Engineered Features:** Work with the numerical material descriptors expected by the model.
- **Dataset Explorer:** Inspect source records, target-temperature distributions, and dataset statistics.
- **Model Insights:** Explore feature importances exposed by the saved model.
- **Sample Material Records:** Load actual dataset rows as starting points for predictions.
- **Prediction History:** Review previous predictions stored in the browser's local storage.
- **REST API:** Access model and dataset functionality through Flask endpoints.
- **Modern Web Interface:** Use a React, TypeScript, and Vite application.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, Flask |
| Machine Learning | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| Model Artifacts | XGBoost JSON, existing model.pkl |
| API Communication | REST / JSON |

## Project Structure

```text
CryoNex/
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
└── README.md
```

*The structure above reflects the supplied project layout. Some files and directories may be omitted for brevity.*

## Dataset

CryoNex uses the project's supplied superconductivity dataset, including engineered numerical descriptors and recorded critical-temperature values.

The feature set describes material composition using statistics derived from elemental properties, including:

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

**Input requirement:** The saved model expects all 81 engineered numerical features in the exact names and order defined by the backend. A chemical formula alone is insufficient unless an appropriate feature-generation pipeline is added.

## Machine Learning Approach

The application uses an existing trained XGBoost regression model to estimate critical temperature.

1. **Data preparation:** Load the supplied dataset and its engineered descriptors.
2. **Feature representation:** Use the numerical material properties expected by the model.
3. **Model inference:** Load the saved XGBoost artifact and generate a prediction.
4. **API response:** Return the predicted critical temperature in Kelvin.
5. **Visualization:** Present predictions and dataset information through the web interface.

The existing model artifacts are preserved; the application does not need to retrain the model during normal prediction requests.

## Model Evaluation

The original notebook reports the following holdout evaluation results for an XGBoost model using an 80/20 split with `random_state=42`.

| Metric | Notebook-reported result |
|---|---:|
| R² score | 0.9264 |
| Mean Absolute Error (MAE) | 5.5566 K |
| Root Mean Squared Error (RMSE) | 9.3099 K |

These are results recorded in the original notebook, not an independent evaluation of the currently deployed application. They should not be interpreted as a guarantee of accuracy for every material.

Feature importance reflects patterns learned by the model and does not establish causation.

## Getting Started

### Prerequisites

Install the following:

- Python 3.10 or newer
- Node.js 20.19+ or 22.12+
- npm

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CryoNex.git
cd CryoNex
```

Replace `YOUR_USERNAME/CryoNex` with your actual GitHub repository path.

### 2. Set Up the Backend

From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Start the Flask API:

```powershell
python backend\app.py
```

The local API is configured to run at:

`http://127.0.0.1:5000`

Keep this terminal open while using the application.

### 3. Set Up the Frontend

Open a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open the Vite URL printed in the terminal. The default is usually:

`http://127.0.0.1:5173`

Ensure that `frontend/.env.local` contains the correct backend URL:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

If `.env.local` already exists, update it instead of overwriting your existing configuration.

## API Reference

The Flask backend exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status and dataset availability |
| GET | `/features` | Exact model feature names, feature groups, and target name |
| POST | `/predict` | Predict critical temperature from all 81 numerical descriptors |
| GET | `/sample?index=0` | Retrieve a source row's features and recorded target |
| GET | `/dataset/summary` | Dataset size, missing-cell count, target statistics, and descriptor statistics |
| GET | `/dataset/records?page=1&page_size=20&q=` | Retrieve paginated records and optionally filter by target temperature |
| GET | `/model/insights` | Retrieve feature importances from the saved XGBoost model |

### Prediction Request

`POST /predict` accepts a JSON object containing all 81 required feature names and their finite numerical values.

The feature names and ordering must match the model's expected schema. Refer to `GET /features` for the canonical feature list.

### Prediction Response

A successful prediction returns the predicted critical temperature in Kelvin under the `predicted_critical_temperature_K` field.

Refer to the actual backend implementation for the complete request schema, response format, and error responses.

## Configuration

The frontend uses `VITE_API_BASE_URL` to determine the backend API origin.

For local development:

```env
VITE_API_BASE_URL=http://127.0.0.1:5000
```

For production, set this variable to the public URL of the deployed Flask API before building the frontend.

The Flask CORS allowlist permits local Vite origins and `https://cyro-nex-rqob.vercel.app`. Update the allowlist if the deployed frontend origin changes.

Do not expose credentials, private keys, or other secrets in frontend environment variables.

## Testing

Run the existing model test script from the repository root:

```powershell
python test_model.py
```

If the script is present at a different path, use its actual location.

Before deployment, verify that:

- The backend loads the saved model successfully.
- The dataset is found at the configured path.
- The feature names match the saved model.
- Valid prediction requests return numeric results.
- Invalid or incomplete requests are handled correctly.
- The frontend can communicate with the backend.
- Dataset records and statistics match the supplied source data.
- The frontend production build completes successfully.

Build the frontend:

```powershell
cd frontend
npm run build
```

## Production Deployment

Deploy the frontend and backend as separate services.

### Backend

- Deploy Flask using a production WSGI server.
- Include `superconductor_model.json` and the canonical CSV at the paths expected by `backend/app.py`.
- Install dependencies from `backend/requirements.txt`.
- Configure CORS for the deployed frontend origin.
- Verify the API endpoints after deployment.

For example, on a compatible Linux hosting environment, if the Flask application object is named `app` in `backend/app.py`, the command may be:

```bash
gunicorn app:app
```

Run this from the `backend/` directory, with the required model and dataset paths available. Follow your hosting provider's instructions for binding the service to its assigned port.

### Frontend

- Build the application with `npm run build` inside `frontend/`.
- Deploy the generated `frontend/dist/` directory to a compatible static hosting service.
- Configure `VITE_API_BASE_URL` to the deployed backend origin at build time.
- Test prediction requests from the public website.

A static frontend host does not replace the Flask API; both services must be available for live predictions.

## Limitations

- Predictions depend on the quality and coverage of the training data.
- A predicted critical temperature is not experimental confirmation of superconductivity.
- The model requires its expected engineered descriptors.
- Notebook-reported metrics may not represent performance on unseen materials outside the original data distribution.
- Feature importance indicates model behavior, not causal relationships.
- Prediction history is stored in the current browser's local storage and can be cleared through the History page. It is not a centralized database.

## Future Enhancements

- Automated feature generation from validated material-composition inputs.
- Additional model evaluation on independent datasets.
- Comparative analysis of regression algorithms.
- Enhanced dataset visualization and filtering.
- Downloadable prediction reports.
- Reproducible model evaluation and experiment tracking.
- Automated testing and continuous integration.

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Run relevant tests.
5. Submit a pull request describing the changes.

## License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the project under an open-source license.

---

**CryoNex — Quantum Materials Intelligence**

*Discover the Critical Frontier.*
@"
.venv/
venv/
node_modules/
.env
.env.local
__pycache__/
*.pyc
"@ | Out-File -FilePath .gitignore -Encoding utf8