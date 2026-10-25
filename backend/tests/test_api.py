import unittest
from unittest.mock import patch

import numpy as np

from backend import app as api


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = api.app.test_client()
        cls.sample = cls.client.get("/sample?index=0").get_json()
        cls.features = api.FEATURES
        cls.payload = {
            feature: cls.sample["features"][feature]
            for feature in cls.features
        }

    def test_features_endpoint_preserves_model_order_and_groups(self):
        response = self.client.get("/features")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["features"], self.features)
        self.assertNotIn(api.TARGET, data["features"])
        grouped = [
            feature
            for group in data["groups"]
            for feature in group["features"]
        ]
        self.assertEqual(grouped, self.features)

    def test_predict_uses_exact_ordered_features_and_excludes_target(self):
        with patch.object(api.model, "predict", return_value=np.array([12.5])) as predict:
            response = self.client.post("/predict", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["predicted_critical_temperature_K"],
            12.5,
        )
        frame = predict.call_args.args[0]
        self.assertEqual(list(frame.columns), self.features)
        self.assertNotIn(api.TARGET, frame.columns)

    def test_predict_rejects_missing_feature(self):
        payload = dict(self.payload)
        payload.pop(self.features[3])
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["missing_features"], [self.features[3]])

    def test_predict_rejects_non_finite_feature(self):
        payload = dict(self.payload)
        payload[self.features[0]] = float("nan")
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("finite", response.get_json()["error"])

    def test_predict_rejects_boolean_feature_values(self):
        payload = dict(self.payload)
        payload[self.features[0]] = True
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("finite", response.get_json()["error"])

    def test_predict_reports_model_failure(self):
        with patch.object(api.model, "predict", side_effect=RuntimeError("model failure")):
            response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Prediction failed.")

    def test_explanation_returns_reconciled_tree_shap_contributions(self):
        response = self.client.post("/model/explain", json=self.payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["method"], "XGBoost TreeSHAP contributions")
        self.assertEqual(
            [item["feature"] for item in data["contributions"]],
            self.features,
        )
        self.assertAlmostEqual(
            sum(item["contribution"] for item in data["contributions"])
            + data["base_value"],
            data["prediction"],
            places=3,
        )

    def test_explanation_rejects_incomplete_candidate(self):
        payload = dict(self.payload)
        payload.pop(self.features[-1])
        response = self.client.post("/model/explain", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(self.features[-1], response.get_json()["missing_features"])

    def test_explanation_reports_unavailable_when_tree_shap_fails(self):
        with patch.object(api.model, "get_booster", side_effect=RuntimeError("unavailable")):
            response = self.client.post("/model/explain", json=self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"],
            "Model explanation is unavailable.",
        )


if __name__ == "__main__":
    unittest.main()
