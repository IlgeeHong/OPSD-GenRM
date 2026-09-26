# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the self-distillation loss, token mask, and teacher utilities."""

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.self_distillation.loss import (
    _apply_entropy_mask,
    _compute_kl_per_token,
    compute_policy_loss_full_logit_kl,
    compute_sd_loss,
)


def _make_sd_config(**overrides):
    defaults = {
        "distillation_add_tail": True,
        "alpha": 0.5,
        "is_clip": None,
        "token_mask_mode": "none",
        "token_mask_top_pct": 70.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _topk_log_probs(bs, seq, k, vocab=32):
    """Log-probs of the first k entries of a random distribution over `vocab`."""
    return torch.log_softmax(torch.randn(bs, seq, vocab), dim=-1)[..., :k]


class TestComputeKlPerToken:
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("add_tail", [True, False])
    def test_non_negative(self, alpha, add_tail):
        config = _make_sd_config(alpha=alpha, distillation_add_tail=add_tail)
        student, teacher = _topk_log_probs(2, 8, 5), _topk_log_probs(2, 8, 5)
        kl = _compute_kl_per_token(student, teacher, config)
        assert kl.shape == (2, 8)
        assert (kl >= -1e-6).all()

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_zero_for_identical_distributions(self, alpha):
        config = _make_sd_config(alpha=alpha)
        log_probs = _topk_log_probs(2, 8, 5)
        kl = _compute_kl_per_token(log_probs, log_probs.clone(), config)
        torch.testing.assert_close(kl, torch.zeros(2, 8), atol=1e-5, rtol=0)

    def test_forward_and_reverse_kl(self):
        student, teacher = _topk_log_probs(2, 8, 5), _topk_log_probs(2, 8, 5)
        cfg = dict(distillation_add_tail=False)
        s = student - torch.logsumexp(student, dim=-1, keepdim=True)
        t = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
        forward = _compute_kl_per_token(student, teacher, _make_sd_config(alpha=0.0, **cfg))
        reverse = _compute_kl_per_token(student, teacher, _make_sd_config(alpha=1.0, **cfg))
        torch.testing.assert_close(forward, (t.exp() * (t - s)).sum(-1))
        torch.testing.assert_close(reverse, (s.exp() * (s - t)).sum(-1))

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError, match="requires student_topk_log_probs"):
            _compute_kl_per_token(None, _topk_log_probs(2, 8, 5), _make_sd_config())


class TestFullLogitKlLoss:
    def _call(self, sd_config, bs=4, seq=6, **kwargs):
        student = _topk_log_probs(bs, seq, 5).requires_grad_(True)
        args = dict(
            old_log_prob=torch.zeros(bs, seq),
            log_prob=torch.zeros(bs, seq),
            advantages=torch.zeros(bs, seq),
            response_mask=torch.ones(bs, seq),
            loss_agg_mode="token-mean",
            config=SimpleNamespace(self_distillation=sd_config),
            student_topk_log_probs=student,
            teacher_topk_log_probs=_topk_log_probs(bs, seq, 5),
        )
        args.update(kwargs)
        return student, compute_policy_loss_full_logit_kl(**args)

    def test_scalar_loss_with_gradient(self):
        student, (loss, _) = self._call(_make_sd_config())
        assert loss.dim() == 0
        loss.backward()
        assert student.grad is not None

    def test_self_distillation_mask_drops_samples(self):
        bs, seq = 4, 6
        sd_config = _make_sd_config()
        student, teacher = _topk_log_probs(bs, seq, 5), _topk_log_probs(bs, seq, 5)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        _, (loss, metrics) = self._call(
            sd_config, student_topk_log_probs=student, teacher_topk_log_probs=teacher, self_distillation_mask=mask
        )
        per_token = _compute_kl_per_token(student, teacher, sd_config)
        torch.testing.assert_close(loss, per_token[mask.bool()].mean())
        assert metrics["self_distillation/empty_target_batch"] is False

    def test_importance_ratio_is_clipped(self):
        bs, seq = 4, 6
        inputs = dict(
            student_topk_log_probs=_topk_log_probs(bs, seq, 5),
            teacher_topk_log_probs=_topk_log_probs(bs, seq, 5),
            log_prob=torch.full((bs, seq), 5.0),
        )
        _, (clipped, _) = self._call(_make_sd_config(is_clip=2.0), **inputs)
        _, (unclipped, _) = self._call(_make_sd_config(is_clip=None), **inputs)
        # exp(5) exceeds the clip, so every token is weighted by exactly 2.
        torch.testing.assert_close(clipped, 2.0 * unclipped)

    def test_requires_self_distillation_config(self):
        with pytest.raises(ValueError, match="self_distillation config"):
            compute_policy_loss_full_logit_kl(
                old_log_prob=None, log_prob=None, advantages=None,
                response_mask=torch.ones(1, 1), config=SimpleNamespace(),
            )

    def test_sd_alias(self):
        from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

        assert POLICY_LOSS_REGISTRY["full_logit_kl"] is compute_policy_loss_full_logit_kl
        assert POLICY_LOSS_REGISTRY["sd"] is compute_sd_loss


class TestEntropyMask:
    def test_none_mode_is_identity(self):
        loss_mask = torch.ones(2, 10)
        mask, metrics = _apply_entropy_mask(None, loss_mask, _make_sd_config(), None, None)
        assert mask is loss_mask
        assert metrics == {}

    def test_entropy_diff_mask_keeps_lowest_student_minus_teacher(self):
        seq = 10
        loss_mask = torch.ones(1, seq)
        teacher_entropy = torch.arange(seq, dtype=torch.float32).unsqueeze(0)
        student_entropy = torch.zeros(1, seq)
        config = _make_sd_config(token_mask_mode="entropy_diff_mask", token_mask_top_pct=70.0)
        mask, metrics = _apply_entropy_mask(None, loss_mask, config, student_entropy, teacher_entropy)
        # H_student - H_teacher = -[0..9]; the 3 lowest values are kept.
        assert mask[0].tolist() == [0.0] * 7 + [1.0] * 3
        assert metrics["actor/entropy_mask_frac"] == pytest.approx(0.3)

    def test_reverse_entropy_diff_mask_keeps_highest(self):
        seq = 10
        loss_mask = torch.ones(1, seq)
        teacher_entropy = torch.arange(seq, dtype=torch.float32).unsqueeze(0)
        config = _make_sd_config(token_mask_mode="reverse_entropy_diff_mask", token_mask_top_pct=70.0)
        mask, _ = _apply_entropy_mask(None, loss_mask, config, torch.zeros(1, seq), teacher_entropy)
        assert mask[0].tolist() == [1.0] * 3 + [0.0] * 7

    def test_unknown_mode_raises(self):
        config = _make_sd_config(token_mask_mode="ht_pos_mask")
        with pytest.raises(ValueError, match="Unsupported self_distillation.token_mask_mode"):
            _apply_entropy_mask(None, torch.ones(1, 4), config, torch.zeros(1, 4), torch.zeros(1, 4))

    def test_padding_is_never_selected(self):
        loss_mask = torch.tensor([[1.0] * 5 + [0.0] * 5])
        teacher_entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0] + [100.0] * 5])
        config = _make_sd_config(token_mask_mode="entropy_diff_mask", token_mask_top_pct=60.0)
        mask, _ = _apply_entropy_mask(None, loss_mask, config, torch.zeros(1, 10), teacher_entropy)
        assert mask[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0] + [0.0] * 5


class TestSdAdvantage:
    def test_passes_through_masked_log_ratio(self):
        import verl.trainer.ppo.self_distillation.reward  # noqa: F401
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn

        rewards = torch.randn(3, 5)
        mask = torch.tensor([[1.0] * 5, [1.0] * 3 + [0.0] * 2, [0.0] * 5])
        advantages, returns = get_adv_estimator_fn("sd")(token_level_rewards=rewards, response_mask=mask)
        torch.testing.assert_close(advantages, rewards * mask)
        torch.testing.assert_close(returns, advantages)


class TestSelfDistillationConfig:
    def test_defaults(self):
        from verl.workers.config.actor import SelfDistillationConfig

        cfg = SelfDistillationConfig()
        assert cfg.enable is False
        assert cfg.distillation_topk == 100
        assert cfg.alpha == 0.5
        assert cfg.teacher_regularization == "ema"
        assert cfg.teacher_update_rate == 0.05
        assert cfg.is_clip == 2.0
        assert cfg.token_mask_mode == "none"

    def test_ema_teacher_rejects_kl_loss(self):
        from verl.workers.config.actor import FSDPActorConfig, PolicyLossConfig, SelfDistillationConfig

        with pytest.raises(ValueError, match="Self-distillation with an EMA teacher is incompatible"):
            FSDPActorConfig(
                strategy="fsdp",
                ppo_micro_batch_size_per_gpu=4,
                rollout_n=4,
                policy_loss=PolicyLossConfig(loss_mode="sd"),
                self_distillation=SelfDistillationConfig(teacher_regularization="ema", teacher_update_rate=0.05),
                use_kl_loss=True,
            )


class TestTrustRegionTeacher:
    """Tests for TrustRegionTeacher."""

    def test_forward_interpolation(self):
        """Test that teacher output is linear interpolation of ref and student."""
        from verl.trainer.ppo.self_distillation.teacher_model import TrustRegionTeacher

        ref = torch.nn.Linear(4, 8)
        student = torch.nn.Linear(4, 8)
        mix_coef = 0.3

        # Wrap in a simple module that returns SimpleNamespace with logits
        class FakeModel(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
            def forward(self, x):
                return SimpleNamespace(logits=self.linear(x))

        ref_model = FakeModel(ref)
        student_model = FakeModel(student)
        teacher = TrustRegionTeacher(ref_model, student_model, mix_coef)

        x = torch.randn(2, 4)
        result = teacher(x)
        expected = torch.lerp(ref(x), student(x), mix_coef)
        torch.testing.assert_close(result.logits, expected)

    def test_ema_update(self):
        """Test EMA update modifies teacher params toward student."""
        from verl.trainer.ppo.self_distillation.teacher_model import update_ema_teacher

        teacher = torch.nn.Linear(4, 4, bias=False)
        student = torch.nn.Linear(4, 4, bias=False)

        # Set known weights
        teacher.weight.data.fill_(1.0)
        student.weight.data.fill_(0.0)

        update_ema_teacher(teacher, student, update_rate=0.1)
        # Expected: (1 - 0.1) * 1.0 + 0.1 * 0.0 = 0.9
        expected = torch.full_like(teacher.weight, 0.9)
        torch.testing.assert_close(teacher.weight.data, expected)
