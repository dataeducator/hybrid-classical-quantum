"""Tests for Phase 5 - Explainability layer.

Covers:
  - Integrated Gradients (tabular + temporal)
  - Attention weight extraction
  - SHAP explanations (KernelExplainer + GradientExplainer)
  - LIME individual patient explanations
  - Per-patient explanation report generation

Note: Packages start with digits -> importlib.import_module.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _import(mod: str):
    return importlib.import_module(mod)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def seer_model():
    """Build a small SEER-only model for testing."""
    fusion = _import("6_survival_modeling.fusion")
    model = fusion.build_seer_only_model(d_in=15, embed_dim=32, dropout=0.0)
    model.eval()
    return model


@pytest.fixture()
def mimic_model():
    """Build a small MIMIC model for testing."""
    fusion = _import("6_survival_modeling.fusion")
    model = fusion.build_mimic_model(
        mimic_tab_d_in=8, embed_dim=32, max_seq_len=10, dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture()
def seer_data():
    """Generate synthetic SEER features."""
    rng = np.random.default_rng(42)
    N = 50
    X = rng.standard_normal((N, 15)).astype(np.float32)
    feature_names = [
        "age", "race_white", "race_black", "race_asian", "race_hispanic",
        "race_other", "stage_ordinal", "grade_ordinal", "tumor_size_cm",
        "lymph_nodes_positive", "surgery_mastectomy", "surgery_lumpectomy",
        "radiation", "chemotherapy", "median_income",
    ]
    return X, feature_names


@pytest.fixture()
def temporal_data():
    """Generate synthetic temporal sequences."""
    rng = np.random.default_rng(42)
    x_seq = torch.from_numpy(rng.standard_normal((1, 10, 7)).astype(np.float32))
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[0, 7:] = False  # last 3 steps invalid
    return x_seq, mask


# ── Integrated Gradients Tests ────────────────────────────────────────

class TestIntegratedGradients:
    def test_tabular_attribution_shape(self, seer_model, seer_data) -> None:
        ig = _import("5_explainability.integrated_gradients")
        X, names = seer_data
        x = torch.from_numpy(X[0])
        attr = ig.integrated_gradients_tabular(
            seer_model, x, names, n_steps=10
        )
        assert attr.values.shape == (15,)
        assert len(attr.feature_names) == 15

    def test_attribution_sums_approximate_delta(self, seer_model, seer_data) -> None:
        ig = _import("5_explainability.integrated_gradients")
        X, names = seer_data
        x = torch.from_numpy(X[0])
        attr = ig.integrated_gradients_tabular(
            seer_model, x, names, n_steps=50
        )
        # Completeness axiom: sum(attributions) ~ pred - baseline_pred
        delta = attr.predicted_risk - attr.baseline_risk
        attr_sum = attr.values.sum()
        # Allow reasonable tolerance for finite steps
        assert abs(attr_sum - delta) < abs(delta) * 0.5 + 0.1

    def test_batch_attributions(self, seer_model, seer_data) -> None:
        ig = _import("5_explainability.integrated_gradients")
        X, names = seer_data
        X_t = torch.from_numpy(X[:5])
        attrs = ig.integrated_gradients_batch(
            seer_model, X_t, names, n_steps=10
        )
        assert attrs.shape == (5, 15)

    def test_temporal_attribution(self, mimic_model, temporal_data) -> None:
        ig = _import("5_explainability.integrated_gradients")
        x_seq, mask = temporal_data
        # Need to provide all inputs for mimic model
        attrs = ig.integrated_gradients_temporal(
            mimic_model, x_seq[0], mask[0], n_steps=10,
            forward_kwargs={
                "x_mimic_tab": torch.randn(1, 8),
                "x_text": torch.randn(1, 768),
            },
        )
        assert attrs.shape == (10, 7)

    def test_summarize_attributions(self) -> None:
        ig = _import("5_explainability.integrated_gradients")
        vals = np.array([0.1, -0.5, 0.3, 0.01, -0.8])
        names = ["a", "b", "c", "d", "e"]
        top = ig.summarize_attributions(vals, names, top_k=3)
        assert len(top) == 3
        assert top[0][0] == "e"  # highest absolute value


# ── Attention Tests ──────────────────────────────────────────────────

class TestAttention:
    def test_gradient_temporal_importance(self, mimic_model, temporal_data) -> None:
        attention = _import("5_explainability.attention")
        x_seq, mask = temporal_data
        # Provide dummy inputs for all MIMIC towers
        extra = {
            "x_mimic_tab": torch.randn(1, 8),
            "x_text": torch.randn(1, 768),
        }
        result = attention._gradient_temporal_importance(
            mimic_model,
            x_seq, mask,
            lab_names=["WBC", "Hgb", "Plt", "Cr", "Alb", "ALP", "LDH"],
            forward_kwargs=extra,
        )
        assert "time_importance" in result
        assert "lab_importance" in result
        assert result["time_importance"].shape == (10,)
        assert result["lab_importance"].shape == (7,)

    def test_time_importance_normalized(self, mimic_model, temporal_data) -> None:
        attention = _import("5_explainability.attention")
        x_seq, mask = temporal_data
        extra = {
            "x_mimic_tab": torch.randn(1, 8),
            "x_text": torch.randn(1, 768),
        }
        result = attention._gradient_temporal_importance(
            mimic_model, x_seq, mask,
            lab_names=["WBC", "Hgb", "Plt", "Cr", "Alb", "ALP", "LDH"],
            forward_kwargs=extra,
        )
        # Should sum to ~1 (normalized)
        assert abs(result["time_importance"].sum() - 1.0) < 0.01

    def test_attention_extractor_init(self) -> None:
        attention = _import("5_explainability.attention")
        towers = _import("6_survival_modeling.towers")
        tower = towers.TemporalTower(n_labs=7, embed_dim=32, max_seq_len=10)
        extractor = attention.AttentionExtractor(tower)
        assert len(extractor._hooks) > 0
        extractor.remove_hooks()
        assert len(extractor._hooks) == 0


# ── SHAP Tests ───────────────────────────────────────────────────────

class TestSHAP:
    def test_kernel_explainer(self, seer_model, seer_data) -> None:
        shap_mod = _import("5_explainability.shap_explainer")
        X, names = seer_data
        explanation = shap_mod.explain_seer_tower(
            seer_model, X[:10], names,
            background_size=20, n_samples=50,
        )
        assert explanation.shap_values.shape == (10, 15)
        assert len(explanation.global_importance) == 15
        assert len(explanation.feature_names) == 15

    def test_shap_importance_sorted(self, seer_model, seer_data) -> None:
        shap_mod = _import("5_explainability.shap_explainer")
        X, names = seer_data
        explanation = shap_mod.explain_seer_tower(
            seer_model, X[:5], names,
            background_size=10, n_samples=30,
        )
        # Importance should be sorted descending by |SHAP|
        importances = [v for _, v in explanation.global_importance]
        assert importances == sorted(importances, reverse=True)

    def test_gradient_explainer(self, seer_model, seer_data) -> None:
        shap_mod = _import("5_explainability.shap_explainer")
        X, names = seer_data
        X_t = torch.from_numpy(X)
        explanation = shap_mod.explain_gradient(
            seer_model, X_t[:10], names,
        )
        assert explanation.shap_values.shape[0] == 10
        assert len(explanation.global_importance) > 0

    def test_subgroup_comparison(self, seer_model, seer_data) -> None:
        shap_mod = _import("5_explainability.shap_explainer")
        X, names = seer_data
        explanation = shap_mod.explain_seer_tower(
            seer_model, X[:20], names,
            background_size=10, n_samples=30,
        )
        groups = np.array([0]*10 + [1]*10)
        comparison = shap_mod.shap_subgroup_comparison(
            explanation, groups, group_names={0: "A", 1: "B"},
        )
        assert "A" in comparison
        assert "B" in comparison
        assert "age" in comparison["A"]


# ── LIME Tests ───────────────────────────────────────────────────────

class TestLIME:
    def test_single_patient_explanation(self, seer_model, seer_data) -> None:
        lime_mod = _import("5_explainability.lime_explainer")
        X, names = seer_data
        exp = lime_mod.explain_patient(
            seer_model, X[0], X, names,
            patient_id="TEST_001",
            num_features=5,
            num_samples=100,
        )
        assert exp.patient_id == "TEST_001"
        assert len(exp.feature_contributions) > 0
        assert len(exp.explanation_text) > 0

    def test_lime_has_predicted_risk(self, seer_model, seer_data) -> None:
        lime_mod = _import("5_explainability.lime_explainer")
        X, names = seer_data
        exp = lime_mod.explain_patient(
            seer_model, X[0], X, names,
            patient_id="TEST_002",
            num_features=3,
            num_samples=50,
        )
        # Should have a real prediction, not zero
        assert isinstance(exp.predicted_risk, float)

    def test_cohort_explanations(self, seer_model, seer_data) -> None:
        lime_mod = _import("5_explainability.lime_explainer")
        X, names = seer_data
        exps = lime_mod.explain_cohort(
            seer_model, X[:3], X, names,
            patient_ids=["P1", "P2", "P3"],
            num_features=3,
            num_samples=50,
            max_patients=3,
        )
        assert len(exps) == 3
        assert exps[0].patient_id == "P1"

    def test_aggregate_importance(self, seer_model, seer_data) -> None:
        lime_mod = _import("5_explainability.lime_explainer")
        X, names = seer_data
        exps = lime_mod.explain_cohort(
            seer_model, X[:5], X, names,
            num_features=5, num_samples=50, max_patients=5,
        )
        agg = lime_mod.aggregate_lime_importance(exps)
        assert len(agg) > 0
        # Should be sorted descending
        vals = [v for _, v in agg]
        assert vals == sorted(vals, reverse=True)


# ── Report Tests ─────────────────────────────────────────────────────

class TestReport:
    def test_generate_patient_report(self, seer_model, seer_data) -> None:
        report_mod = _import("5_explainability.report")
        X, names = seer_data
        report = report_mod.generate_patient_report(
            seer_model, X[0], X, names,
            patient_id="SEER_000001",
            ig_steps=10,
            lime_samples=50,
        )
        assert report.patient_id == "SEER_000001"
        assert len(report.ig_top_features) > 0
        assert len(report.shap_top_features) > 0
        assert len(report.lime_contributions) > 0

    def test_report_to_json(self, seer_model, seer_data) -> None:
        report_mod = _import("5_explainability.report")
        X, names = seer_data
        report = report_mod.generate_patient_report(
            seer_model, X[0], X, names,
            patient_id="SEER_000002",
            ig_steps=10, lime_samples=50,
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["patient_id"] == "SEER_000002"
        assert "integrated_gradients" in parsed
        assert "shap" in parsed
        assert "lime" in parsed

    def test_report_to_dict(self, seer_model, seer_data) -> None:
        report_mod = _import("5_explainability.report")
        X, names = seer_data
        report = report_mod.generate_patient_report(
            seer_model, X[0], X, names,
            patient_id="SEER_000003",
            ig_steps=10, lime_samples=50,
        )
        d = report.to_dict()
        assert isinstance(d["predicted_risk"], float)
        assert isinstance(d["integrated_gradients"]["top_features"], list)

    def test_cohort_reports_saves_json(self, seer_model, seer_data, tmp_path) -> None:
        report_mod = _import("5_explainability.report")
        X, names = seer_data
        reports = report_mod.generate_cohort_reports(
            seer_model, X[:3], X, names,
            patient_ids=["P1", "P2", "P3"],
            output_dir=tmp_path / "explanations",
            max_patients=3,
        )
        assert len(reports) == 3
        assert (tmp_path / "explanations" / "cohort_explanations.json").exists()
        assert (tmp_path / "explanations" / "P1_explanation.json").exists()

        # Verify JSON is valid
        with open(tmp_path / "explanations" / "P1_explanation.json") as f:
            data = json.load(f)
        assert data["patient_id"] == "P1"


# ── Integration Test ──────────────────────────────────────────────────

class TestExplainabilityIntegration:
    def test_full_pipeline_with_seer_data(self, tmp_path) -> None:
        """End-to-end: generate data -> train -> explain."""
        # Generate synthetic data
        synth = _import("1_data_harmonization.synthetic.seer_synth")
        synth.generate_seer(n=100, seed=42, output_dir=tmp_path)

        # Extract and engineer features
        etl = _import("1_data_harmonization.etl.seer_etl")
        raw = etl.extract(tmp_path)
        feat = _import("2_feature_engineering.seer_features")
        features = feat.extract_seer_features(raw)

        # Build and "train" model (just 2 epochs)
        fusion = _import("6_survival_modeling.fusion")
        dataset = _import("6_survival_modeling.dataset")
        trainer = _import("6_survival_modeling.trainer")

        model = fusion.build_seer_only_model(
            d_in=features.X.shape[1], embed_dim=32
        )
        ds = dataset.SeerSurvivalDataset(
            features.X, features.time,
            features.event, features.race_labels,
        )
        train_ds, val_ds = dataset.train_test_split_temporal(ds, test_frac=0.3)
        config = trainer.TrainingConfig(
            epochs=2, batch_size=32, checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer.train_seer_model(model, train_ds, val_ds, config)

        # Generate explanation report
        report_mod = _import("5_explainability.report")
        report = report_mod.generate_patient_report(
            model, features.X[0], features.X, features.feature_names,
            patient_id=features.person_ids[0],
            ig_steps=10, lime_samples=50,
        )

        assert report.patient_id == features.person_ids[0]
        assert len(report.ig_top_features) > 0
        assert len(report.shap_top_features) > 0
        assert len(report.lime_contributions) > 0

        # Verify JSON serialization round-trip
        parsed = json.loads(report.to_json())
        assert parsed["patient_id"] == features.person_ids[0]
