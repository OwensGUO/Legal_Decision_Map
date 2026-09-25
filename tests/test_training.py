from __future__ import annotations

import json
import unittest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from legal_landscape.training.train import (
    DeterministicEpochSampler,
    build_experiment,
    format_model_input,
    inspect_model_config,
    load_counterfactual_pairs,
    make_counterfactual_prediction_rows,
    make_static_prediction_rows,
    training_plan,
    validate_resume_metadata,
)


class TrainingPlanTests(unittest.TestCase):
    def test_deterministic_sampler_resume_preserves_remaining_order(self) -> None:
        sampler = DeterministicEpochSampler(6, seed=42)
        full_order = list(sampler)
        self.assertEqual(full_order, [3, 1, 2, 4, 0, 5])

        sampler.advance(2)
        state = sampler.state_dict()
        restored = DeterministicEpochSampler(6, seed=42)
        restored.load_state_dict(state)
        self.assertEqual(list(restored), [2, 4, 0, 5])

        restored.next_epoch()
        self.assertEqual(restored.state_dict(), {"epoch": 1, "start_index": 0, "seed": 42})

    def test_resume_metadata_rejects_seed_or_input_changes(self) -> None:
        progress = {
            "seed": 42,
            "input_identity": {"train": {"sha256": "abc", "bytes": 10}},
        }
        validate_resume_metadata(
            progress,
            expected_seed=42,
            expected_input_identity={"train": {"sha256": "abc", "bytes": 10}},
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_resume_metadata(
                progress,
                expected_seed=2026,
                expected_input_identity=progress["input_identity"],
            )
        with self.assertRaisesRegex(ValueError, "input"):
            validate_resume_metadata(
                progress,
                expected_seed=42,
                expected_input_identity={"train": {"sha256": "changed", "bytes": 10}},
            )

    def test_experiment_matrix_has_all_requested_variants(self) -> None:
        names = ("B0", "B1", "B2", "B3", "B4", "B5", "M", "A1", "A2", "A3", "A4", "A5", "A6")
        experiments = {name: build_experiment(name) for name in names}
        self.assertFalse(experiments["B3"].use_factors)
        self.assertTrue(experiments["B4"].use_factors)
        self.assertTrue(experiments["M"].typed_counterfactuals)
        self.assertFalse(experiments["A1"].use_invariant)
        self.assertTrue(experiments["A5"].hard_charge_sentence)
        self.assertFalse(experiments["A6"].filter_counterfactuals)

    def test_model_loader_choice_is_derived_from_local_config(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen-test", "architectures": ["QwenTestForCausalLM"]}),
                encoding="utf-8",
            )
            inspected = inspect_model_config(root)
        self.assertEqual(inspected.loader_class, "AutoModelForCausalLM")
        self.assertEqual(inspected.architecture_kind, "causal_lm")
        self.assertEqual(inspected.text_backbone_path, ("model",))
        self.assertEqual(inspected.model_type, "qwen-test")

    def test_multimodal_qwen_uses_image_text_loader_and_language_backbone(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "architectures": ["Qwen3_5ForCausalLM"],
                "hidden_size": 4096,
            },
            "vision_config": {"hidden_size": 1024},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            inspected = inspect_model_config(root)
        self.assertEqual(inspected.loader_class, "AutoModelForImageTextToText")
        self.assertEqual(inspected.architecture_kind, "conditional_generation")
        self.assertEqual(inspected.text_backbone_path, ("model", "language_model"))
        self.assertTrue(inspected.has_visual_components)
        self.assertEqual(inspected.hidden_size, 4096)

    def test_training_plan_never_loads_weights(self) -> None:
        plan = training_plan(
            {
                "model": {"path": "/missing/model", "quantization": "nf4"},
                "training": {"seed": 42, "mixed_precision": "bf16"},
            },
            experiment="M",
        )
        self.assertEqual(plan["model_path"], "/missing/model")
        self.assertEqual(plan["mode"], "full")
        self.assertTrue(plan["requires_execute"])

    def test_model_input_has_explicit_dataset_defendant_fact_and_factors(self) -> None:
        text = format_model_input(
            {
                "dataset": "cmdl",
                "target_defendant": "某甲",
                "fact_conservative": "案件事实。",
                "fact_strict": "严格屏蔽后的案件事实。",
                "factors": {"amount": 1000.0, "confession": True},
            },
            use_factors=True,
        )
        self.assertIn("[DATASET]\ncmdl", text)
        self.assertIn("[TARGET_DEFENDANT]\n某甲", text)
        self.assertIn("[FACT]\n严格屏蔽后的案件事实。", text)
        self.assertNotIn("[FACT]\n案件事实。", text)
        self.assertIn("[FACTORS]", text)
        self.assertIn('"amount": 1000.0', text)

    def test_counterfactual_pair_loader_filters_invalid_by_default(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        parent = {
            "case_id": "c1",
            "dataset": "cail",
            "target_defendant": "某甲",
            "fact_conservative": "父事实",
            "factors": {},
            "charges": ["盗窃"],
        }
        valid = {
            "parent_case_id": "c1",
            "intervention_type": "invariant",
            "validation": {"valid": True, "parsed": {"counterfactual_text": "反事实"}},
            "target_charge": None,
            "rank_direction": None,
        }
        invalid = {
            **valid,
            "validation": {"valid": False, "parsed": {"counterfactual_text": "坏样本"}},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cf.jsonl"
            path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in (valid, invalid)),
                encoding="utf-8",
            )
            pairs = load_counterfactual_pairs(path, {"c1": parent}, filter_valid=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["parent_text"], "父事实")
        self.assertEqual(pairs[0]["counterfactual_text"], "反事实")

    def test_static_prediction_rows_preserve_labels_and_group_ids(self) -> None:
        records = [
            {
                "case_id": "c1",
                "group_id": "g1",
                "charges": ["盗窃"],
                "conviction_articles": ["criminal_law:264"],
                "penalty_type": "fixed_term",
                "imprisonment_months": 12,
            },
            {
                "case_id": "c2",
                "group_id": "g2",
                "charges": ["诈骗"],
                "conviction_articles": ["criminal_law:266"],
                "penalty_type": "life",
                "imprisonment_months": None,
            },
        ]
        rows = make_static_prediction_rows(
            records,
            charge_probabilities=[[0.8, 0.2], [0.3, 0.7]],
            article_probabilities=[[0.9, 0.1], [0.2, 0.8]],
            penalty_indices=[0, 1],
            sentence_months=[11.5, 99.0],
            charge_vocabulary=["盗窃", "诈骗"],
            article_vocabulary=["criminal_law:264", "criminal_law:266"],
            penalty_vocabulary=["fixed_term", "life"],
        )
        self.assertEqual(rows[0]["group_id"], "g1")
        self.assertEqual(rows[0]["true_charges"], [1, 0])
        self.assertEqual(rows[0]["predicted_charges"], [1, 0])
        self.assertEqual(rows[0]["true_articles"], [1, 0])
        self.assertEqual(rows[0]["predicted_articles"], [1, 0])
        self.assertEqual(rows[0]["predicted_article_labels"], ["criminal_law:264"])
        self.assertTrue(rows[0]["correct"])
        self.assertIsNone(rows[1]["true_months"])

    def test_static_prediction_rows_support_multiple_predicted_charges(self) -> None:
        record = {
            "case_id": "c1",
            "group_id": "g1",
            "charges": ["盗窃", "诈骗"],
            "conviction_articles": ["criminal_law:264", "criminal_law:266"],
            "penalty_type": "fixed_term",
            "imprisonment_months": 12,
        }
        row = make_static_prediction_rows(
            [record],
            charge_probabilities=[[0.8, 0.7]],
            article_probabilities=[[0.8, 0.7]],
            penalty_indices=[0],
            sentence_months=[12.0],
            charge_vocabulary=["盗窃", "诈骗"],
            article_vocabulary=["criminal_law:264", "criminal_law:266"],
            penalty_vocabulary=["fixed_term"],
        )[0]
        self.assertEqual(row["predicted_charges"], [1, 1])
        self.assertTrue(row["correct"])

    def test_counterfactual_prediction_rows_use_charge_index_targets(self) -> None:
        pairs = [
            {
                "parent_record": {"case_id": "c1", "group_id": "g1"},
                "intervention_type": "charge_flip",
                "target_charge": "诈骗",
                "rank_direction": None,
            }
        ]
        rows = make_counterfactual_prediction_rows(
            pairs,
            parent_probabilities=[[0.9, 0.1]],
            counterfactual_probabilities=[[0.2, 0.8]],
            parent_sentences=[12.0],
            counterfactual_sentences=[12.0],
            charge_vocabulary=["盗窃", "诈骗"],
        )
        self.assertEqual(rows[0]["target_charge"], 1)
        self.assertEqual(rows[0]["group_id"], "g1")


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class TorchTrainingTests(unittest.TestCase):
    def test_masked_mean_pool_reduces_token_states(self) -> None:
        from legal_landscape.training.train import masked_mean_pool

        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean_pool(hidden, mask)
        self.assertEqual(tuple(pooled.shape), (1, 2))
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_dummy_train_step_is_finite(self) -> None:
        from legal_landscape.training.train import run_dummy_train_step

        value = run_dummy_train_step(seed=42)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertGreater(value, 0.0)

    def test_nan_and_nonzero_empty_loss_write_diagnostics(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from legal_landscape.models.losses import LossBreakdown
        from legal_landscape.training.train import validate_loss_breakdown

        zero = torch.tensor(0.0)
        bad = LossBreakdown(
            charge=torch.tensor(float("nan")),
            article=zero,
            sentence=zero,
            invariant=zero,
            boundary=zero,
            response=zero,
            factor=torch.tensor(1.0),
            total=zero,
            active_counts={
                "charge": 1,
                "article": 0,
                "sentence": 0,
                "invariant": 0,
                "boundary": 0,
                "response": 0,
                "factor": 0,
            },
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            with self.assertRaises(FloatingPointError):
                validate_loss_breakdown(bad, path, step=3)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["step"], 3)
        self.assertIn("charge", payload["errors"])
        self.assertIn("factor:nonzero_without_samples", payload["errors"])


if __name__ == "__main__":
    unittest.main()
