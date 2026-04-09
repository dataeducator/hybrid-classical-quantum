"""Tests for the multi-source deep survival model.

Covers:
  - Feature engineering (SEER tabular, MIMIC temporal/text)
  - Model towers (SEERTower, TemporalTower, TextTower)
  - Fusion model (forward pass, factory functions)
  - Loss functions (Cox PH, fairness regularizer)
  - Evaluation metrics (C-index, Brier score)
  - Fairness audit (demographic parity, equalized odds)
  - Datasets and data splitting

Note: Packages 1_data_harmonization, 2_feature_engineering, etc.
start with digits, so we use importlib.import_module.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Module imports via importlib ──────────────────────────────────────

def _import(mod: str):
    return importlib.import_module(mod)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def seer_raw(tmp_path: Path):
    """Generate synthetic SEER data and extract raw DataFrame."""
    synth = _import("1_data_harmonization.synthetic.seer_synth")
    synth.generate_seer(n=200, seed=42, output_dir=tmp_path)
    etl = _import("1_data_harmonization.etl.seer_etl")
    return etl.extract(tmp_path)


@pytest.fixture()
def mimic_omop(tmp_path: Path):
    """Generate synthetic MIMIC data and run ETL."""
    synth = _import("1_data_harmonization.synthetic.mimic_synth")
    synth.generate_mimic(n=100, seed=42, output_dir=tmp_path)
    etl = _import("1_data_harmonization.etl.mimic_etl")
    return etl.run(tmp_path)


@pytest.fixture()
def seer_features(seer_raw):
    """Extract SEER features from raw data."""
    feat = _import("2_feature_engineering.seer_features")
    return feat.extract_seer_features(seer_raw)


@pytest.fixture()
def mimic_features(mimic_omop):
    """Extract MIMIC features from OMOP tables."""
    feat = _import("2_feature_engineering.mimic_features")
    return feat.extract_mimic_features(mimic_omop, max_seq_len=20)


# ── Feature Engineering Tests ────────────────────────────────────────

class TestSEERFeatures:
    def test_shapes(self, seer_features) -> None:
        f = seer_features
        assert f.X.shape[0] == 200
        assert f.X.shape[1] == 15  # 15 features
        assert f.time.shape == (200,)
        assert f.event.shape == (200,)
        assert f.race_labels.shape == (200,)

    def test_dtypes(self, seer_features) -> None:
        assert seer_features.X.dtype == np.float32
        assert seer_features.time.dtype == np.float32
        assert seer_features.event.dtype == np.int32

    def test_events_binary(self, seer_features) -> None:
        assert set(np.unique(seer_features.event)).issubset({0, 1})

    def test_race_codes_valid(self, seer_features) -> None:
        assert all(r in range(5) for r in seer_features.race_labels)

    def test_feature_names(self, seer_features) -> None:
        assert len(seer_features.feature_names) == seer_features.X.shape[1]


class TestMIMICFeatures:
    def test_shapes(self, mimic_features) -> None:
        f = mimic_features
        assert f.X_tab.shape[0] == 100
        assert f.X_seq.shape == (100, 20, 7)
        assert f.seq_mask.shape == (100, 20)
        assert f.X_text.shape == (100, 768)
        assert f.time.shape == (100,)
        assert f.event.shape == (100,)

    def test_dtypes(self, mimic_features) -> None:
        assert mimic_features.X_tab.dtype == np.float32
        assert mimic_features.X_seq.dtype == np.float32
        assert mimic_features.seq_mask.dtype == bool
        assert mimic_features.X_text.dtype == np.float32

    def test_seq_mask_has_valid_steps(self, mimic_features) -> None:
        # At least some patients should have lab data
        assert mimic_features.seq_mask.any()


# ── Tower Tests ──────────────────────────────────────────────────────

class TestSEERTower:
    def test_forward_shape(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.SEERTower(d_in=15, embed_dim=64)
        x = torch.randn(32, 15)
        out = tower(x)
        assert out.shape == (32, 64)

    def test_gradient_flows(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.SEERTower(d_in=15, embed_dim=64)
        x = torch.randn(16, 15, requires_grad=True)
        out = tower(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTemporalTower:
    def test_forward_shape(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.TemporalTower(n_labs=7, embed_dim=64, max_seq_len=20)
        x = torch.randn(16, 20, 7)
        mask = torch.ones(16, 20, dtype=torch.bool)
        out = tower(x, mask)
        assert out.shape == (16, 64)

    def test_with_partial_mask(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.TemporalTower(n_labs=7, embed_dim=32, max_seq_len=10)
        x = torch.randn(8, 10, 7)
        mask = torch.zeros(8, 10, dtype=torch.bool)
        mask[:, :5] = True  # only first 5 steps valid
        out = tower(x, mask)
        assert out.shape == (8, 32)

    def test_without_mask(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.TemporalTower(n_labs=7, embed_dim=32)
        x = torch.randn(4, 10, 7)
        out = tower(x, None)
        assert out.shape == (4, 32)


class TestTextTower:
    def test_forward_shape(self) -> None:
        towers = _import("6_survival_modeling.towers")
        tower = towers.TextTower(d_bert=768, embed_dim=64)
        x = torch.randn(16, 768)
        out = tower(x)
        assert out.shape == (16, 64)


# ── Fusion Model Tests ──────────────────────────────────────────────

class TestMultiSourceModel:
    def test_seer_only_forward(self) -> None:
        fusion = _import("6_survival_modeling.fusion")
        model = fusion.build_seer_only_model(d_in=15, embed_dim=32)
        x = torch.randn(16, 15)
        out = model(x_seer=x)
        assert out.shape == (16,)

    def test_mimic_forward(self) -> None:
        fusion = _import("6_survival_modeling.fusion")
        model = fusion.build_mimic_model(
            mimic_tab_d_in=8, embed_dim=32, max_seq_len=10
        )
        out = model(
            x_mimic_tab=torch.randn(8, 8),
            x_seq=torch.randn(8, 10, 7),
            seq_mask=torch.ones(8, 10, dtype=torch.bool),
            x_text=torch.randn(8, 768),
        )
        assert out.shape == (8,)

    def test_fusion_forward(self) -> None:
        fusion = _import("6_survival_modeling.fusion")
        model = fusion.build_fusion_model(
            seer_d_in=15, embed_dim=32, max_seq_len=10
        )
        out = model(
            x_seer=torch.randn(4, 15),
            x_seq=torch.randn(4, 10, 7),
            seq_mask=torch.ones(4, 10, dtype=torch.bool),
            x_text=torch.randn(4, 768),
        )
        assert out.shape == (4,)

    def test_no_input_raises(self) -> None:
        fusion = _import("6_survival_modeling.fusion")
        model = fusion.build_seer_only_model()
        with pytest.raises(ValueError, match="No tower inputs"):
            model()

    def test_get_embeddings(self) -> None:
        fusion = _import("6_survival_modeling.fusion")
        model = fusion.build_seer_only_model(d_in=15, embed_dim=32)
        x = torch.randn(8, 15)
        emb = model.get_embeddings(x_seer=x)
        assert emb.shape == (8, 32)


# ── Loss Function Tests ─────────────────────────────────────────────

class TestCoxPHLoss:
    def test_basic_loss(self) -> None:
        losses = _import("6_survival_modeling.losses")
        log_h = torch.tensor([2.0, 1.0, 0.5, -0.5], requires_grad=True)
        time = torch.tensor([10.0, 20.0, 30.0, 40.0])
        event = torch.tensor([1, 1, 0, 1])
        loss = losses.cox_ph_loss(log_h, time, event)
        assert loss.item() > 0
        loss.backward()
        assert log_h.grad is not None

    def test_all_censored_returns_zero(self) -> None:
        losses = _import("6_survival_modeling.losses")
        log_h = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        time = torch.tensor([10.0, 20.0, 30.0])
        event = torch.tensor([0, 0, 0])
        loss = losses.cox_ph_loss(log_h, time, event)
        assert loss.item() == 0.0

    def test_module_wrapper(self) -> None:
        losses = _import("6_survival_modeling.losses")
        criterion = losses.CoxPHLoss()
        log_h = torch.randn(20, requires_grad=True)
        time = torch.rand(20) * 100
        event = torch.randint(0, 2, (20,))
        loss = criterion(log_h, time, event)
        assert loss.shape == ()


class TestFairnessRegularizer:
    def test_demographic_parity(self) -> None:
        losses = _import("6_survival_modeling.losses")
        log_h = torch.tensor([3.0, 3.0, 1.0, 1.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        penalty = losses.fairness_regularizer(log_h, groups, n_groups=2)
        assert penalty.item() > 0  # groups have different means

    def test_equal_groups_low_penalty(self) -> None:
        losses = _import("6_survival_modeling.losses")
        log_h = torch.tensor([2.0, 2.0, 2.0, 2.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        penalty = losses.fairness_regularizer(log_h, groups, n_groups=2)
        assert penalty.item() < 0.01  # should be near zero


# ── Evaluation Tests ─────────────────────────────────────────────────

class TestConcordanceIndex:
    def test_perfect_concordance(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        risk = np.array([3.0, 2.0, 1.0])
        time = np.array([10.0, 20.0, 30.0])
        event = np.array([1, 1, 1])
        ci = evaluate.concordance_index(risk, time, event)
        assert ci == 1.0

    def test_random_around_half(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        rng = np.random.default_rng(42)
        risk = rng.standard_normal(100)
        time = rng.uniform(0, 100, 100)
        event = rng.integers(0, 2, 100)
        ci = evaluate.concordance_index(risk, time, event)
        assert 0.3 < ci < 0.7

    def test_fast_matches_slow(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        rng = np.random.default_rng(123)
        risk = rng.standard_normal(50)
        time = rng.uniform(0, 100, 50)
        event = rng.integers(0, 2, 50)
        ci_slow = evaluate.concordance_index(risk, time, event)
        ci_fast = evaluate.concordance_index_fast(risk, time, event)
        assert abs(ci_slow - ci_fast) < 0.05

    def test_subgroup_c_index(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        rng = np.random.default_rng(99)
        n = 100
        risk = rng.standard_normal(n)
        time = rng.uniform(0, 100, n)
        event = rng.integers(0, 2, n)
        groups = rng.integers(0, 3, n)
        result = evaluate.subgroup_c_index(
            risk, time, event, groups,
            group_names={0: "A", 1: "B", 2: "C"},
        )
        assert set(result.keys()) == {"A", "B", "C"}
        for ci in result.values():
            assert 0.0 <= ci <= 1.0 or np.isnan(ci)


class TestBrierScore:
    def test_perfect_prediction(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        # Patient died at t=5, predicted survival P(T>10) = 0.0 -> good
        surv_prob = np.array([0.0, 1.0])
        time = np.array([5.0, 20.0])
        event = np.array([1, 0])
        bs = evaluate.brier_score(surv_prob, time, event, eval_time=10.0)
        assert bs == 0.0

    def test_bad_prediction(self) -> None:
        evaluate = _import("6_survival_modeling.evaluate")
        # Patient died at t=5, predicted survival P(T>10) = 1.0 -> bad
        surv_prob = np.array([1.0])
        time = np.array([5.0])
        event = np.array([1])
        bs = evaluate.brier_score(surv_prob, time, event, eval_time=10.0)
        assert bs == 1.0


# ── Dataset Tests ────────────────────────────────────────────────────

class TestDatasets:
    def test_seer_dataset_len(self, seer_features) -> None:
        dataset = _import("6_survival_modeling.dataset")
        ds = dataset.SeerSurvivalDataset(
            seer_features.X, seer_features.time,
            seer_features.event, seer_features.race_labels,
        )
        assert len(ds) == 200

    def test_seer_dataset_item(self, seer_features) -> None:
        dataset = _import("6_survival_modeling.dataset")
        ds = dataset.SeerSurvivalDataset(
            seer_features.X, seer_features.time,
            seer_features.event, seer_features.race_labels,
        )
        item = ds[0]
        assert "x_seer" in item
        assert "time" in item
        assert "event" in item
        assert "race" in item
        assert item["x_seer"].shape == (15,)

    def test_mimic_dataset(self, mimic_features) -> None:
        dataset = _import("6_survival_modeling.dataset")
        ds = dataset.MimicSurvivalDataset(
            mimic_features.X_tab, mimic_features.X_seq,
            mimic_features.seq_mask, mimic_features.X_text,
            mimic_features.time, mimic_features.event,
            mimic_features.race_labels,
        )
        assert len(ds) == 100
        item = ds[0]
        assert "x_seq" in item
        assert "x_text" in item

    def test_train_test_split(self, seer_features) -> None:
        dataset = _import("6_survival_modeling.dataset")
        ds = dataset.SeerSurvivalDataset(
            seer_features.X, seer_features.time,
            seer_features.event, seer_features.race_labels,
        )
        train, val = dataset.train_test_split_temporal(ds, test_frac=0.2)
        assert len(train) + len(val) == 200
        assert len(val) > 0


# ── Fairness Tests ───────────────────────────────────────────────────

class TestFairnessMetrics:
    def test_demographic_parity(self) -> None:
        fairness = _import("7_disparity_analysis.fairness")
        rng = np.random.default_rng(42)
        risk = np.concatenate([
            rng.normal(2.0, 0.5, 50),   # group 0: higher risk
            rng.normal(0.0, 0.5, 50),   # group 1: lower risk
        ])
        groups = np.array([0]*50 + [1]*50)
        rates, gap = fairness.demographic_parity(
            risk, groups, group_names={0: "A", 1: "B"}
        )
        assert "A" in rates
        assert "B" in rates
        assert gap > 0  # groups have different risk distributions

    def test_equalized_odds(self) -> None:
        fairness = _import("7_disparity_analysis.fairness")
        rng = np.random.default_rng(42)
        n = 100
        risk = rng.standard_normal(n)
        event = rng.integers(0, 2, n)
        groups = rng.integers(0, 2, n)
        tpr, fpr, tpr_gap, fpr_gap = fairness.equalized_odds(
            risk, event, groups, group_names={0: "A", 1: "B"}
        )
        assert "A" in tpr
        assert "B" in tpr
        assert tpr_gap >= 0
        assert fpr_gap >= 0

    def test_fairness_audit_full(self) -> None:
        fairness = _import("7_disparity_analysis.fairness")
        rng = np.random.default_rng(42)
        n = 200
        risk = rng.standard_normal(n)
        time = rng.uniform(0, 100, n)
        event = rng.integers(0, 2, n)
        groups = rng.integers(0, 3, n)
        report = fairness.fairness_audit(
            risk, time, event, groups,
            group_names={0: "A", 1: "B", 2: "C"},
        )
        assert report.dp_gap >= 0
        assert report.overall_fairness_score >= 0
        summary = report.summary()
        assert "Fairness Audit Report" in summary

    def test_calibration_by_group(self) -> None:
        fairness = _import("7_disparity_analysis.fairness")
        rng = np.random.default_rng(42)
        n = 100
        risk = rng.standard_normal(n)
        time = rng.uniform(0, 100, n)
        event = rng.integers(0, 2, n)
        groups = rng.integers(0, 2, n)
        cal = fairness.calibration_by_group(
            risk, time, event, groups,
            group_names={0: "A", 1: "B"},
        )
        assert "A" in cal
        assert "n_patients" in cal["A"]


# ── Subgroup Analysis Tests ──────────────────────────────────────────

class TestSubgroupAnalysis:
    def test_kaplan_meier(self) -> None:
        subgroup = _import("7_disparity_analysis.subgroup")
        time = np.array([5, 10, 15, 20, 25, 30], dtype=float)
        event = np.array([1, 0, 1, 1, 0, 1])
        km = subgroup.kaplan_meier(time, event)
        assert km.n_events == 4
        assert km.n_censored == 2
        assert km.survival_prob[0] == 1.0  # starts at 1
        assert km.survival_prob[-1] < 1.0  # decreases

    def test_subgroup_analysis_runs(self) -> None:
        subgroup = _import("7_disparity_analysis.subgroup")
        rng = np.random.default_rng(42)
        n = 200
        risk = rng.standard_normal(n)
        time = rng.uniform(1, 100, n)
        event = rng.integers(0, 2, n)
        race = rng.integers(0, 3, n)
        report = subgroup.subgroup_analysis(
            risk, time, event, race,
            race_names={0: "A", 1: "B", 2: "C"},
        )
        assert len(report.km_curves) > 0
        assert len(report.median_risks) > 0


# ── Integration Test ─────────────────────────────────────────────────

class TestEndToEnd:
    def test_seer_training_loop(self, seer_features) -> None:
        """Verify a short SEER training loop runs without error."""
        dataset = _import("6_survival_modeling.dataset")
        fusion = _import("6_survival_modeling.fusion")
        trainer = _import("6_survival_modeling.trainer")

        ds = dataset.SeerSurvivalDataset(
            seer_features.X, seer_features.time,
            seer_features.event, seer_features.race_labels,
        )
        train_ds, val_ds = dataset.train_test_split_temporal(ds, test_frac=0.3)

        model = fusion.build_seer_only_model(
            d_in=seer_features.X.shape[1], embed_dim=32
        )

        config = trainer.TrainingConfig(
            epochs=3,
            batch_size=32,
            lr=1e-3,
            patience=5,
            checkpoint_dir=str(Path(seer_features.person_ids[0]).parent / "ckpt")
            if False else "checkpoints/test",
        )

        history = trainer.train_seer_model(model, train_ds, val_ds, config)
        assert len(history.train_loss) == 3
        assert len(history.val_c_index) == 3
        assert 0 <= history.best_c_index <= 1.0

    def test_transfer_weights(self) -> None:
        """Verify SEER tower weight transfer works."""
        fusion = _import("6_survival_modeling.fusion")
        trainer = _import("6_survival_modeling.trainer")

        source = fusion.build_seer_only_model(d_in=15, embed_dim=32)
        target = fusion.build_fusion_model(seer_d_in=15, embed_dim=32)

        # Before transfer, weights differ
        src_w = list(source.seer_tower.parameters())[0].data.clone()

        trainer.transfer_seer_weights(source, target, freeze_seer=True)

        # After transfer, weights match
        tgt_w = list(target.seer_tower.parameters())[0].data
        assert torch.allclose(src_w, tgt_w)

        # And SEER tower is frozen
        for p in target.seer_tower.parameters():
            assert not p.requires_grad
