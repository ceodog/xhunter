import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from planetx.constants import OBJECT_FEATURE_KEYS, THETA_KEYS  # noqa: E402
from planetx.model.encoder import SetTransformerEncoder  # noqa: E402
from planetx.model.posterior_net import PosteriorNet  # noqa: E402


def _random_batch(batch_size=4, max_n=6, feat_dim=len(OBJECT_FEATURE_KEYS)):
    torch.manual_seed(0)
    feats = torch.randn(batch_size, max_n, feat_dim)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    ns = torch.randint(1, max_n + 1, (batch_size,))
    for i, n in enumerate(ns):
        mask[i, n:] = True
        feats[i, n:] = 0.0
    return feats, mask


def test_set_transformer_output_shape():
    feats, mask = _random_batch()
    enc = SetTransformerEncoder(in_dim=feats.shape[-1], d_model=16, n_layers=2, n_heads=2, n_seeds=3)
    z = enc(feats, mask)
    assert z.shape == (feats.shape[0], 16 * 3)


def test_set_transformer_handles_all_masked_row():
    feat_dim = len(OBJECT_FEATURE_KEYS)
    feats = torch.zeros(1, 4, feat_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)  # every real object masked out
    enc = SetTransformerEncoder(in_dim=feat_dim, d_model=8, n_layers=1, n_heads=2, n_seeds=1)
    z = enc(feats, mask)  # must not raise / produce nan, thanks to the null token
    assert torch.isfinite(z).all()


def test_set_transformer_is_permutation_invariant():
    feat_dim = len(OBJECT_FEATURE_KEYS)
    torch.manual_seed(1)
    feats = torch.randn(1, 5, feat_dim)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    enc = SetTransformerEncoder(in_dim=feat_dim, d_model=16, n_layers=2, n_heads=2, n_seeds=2)
    enc.eval()

    perm = torch.randperm(5)
    with torch.no_grad():
        z1 = enc(feats, mask)
        z2 = enc(feats[:, perm], mask[:, perm])
    assert torch.allclose(z1, z2, atol=1e-5)


def test_posterior_net_loss_and_backward():
    feats, mask = _random_batch()
    survey_meta = torch.randn(feats.shape[0], 3)
    theta = torch.randn(feats.shape[0], len(THETA_KEYS))

    net = PosteriorNet(object_feature_dim=feats.shape[-1], theta_dim=len(THETA_KEYS),
                        d_model=16, n_layers=1, n_heads=2, n_seeds=2)
    loss = net.loss(feats, mask, survey_meta, theta)
    assert loss.dim() == 0
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_posterior_samples_shape():
    feats, mask = _random_batch(batch_size=1)
    survey_meta = torch.randn(1, 3)
    net = PosteriorNet(object_feature_dim=feats.shape[-1], theta_dim=len(THETA_KEYS),
                        d_model=16, n_layers=1, n_heads=2, n_seeds=2)
    samples = net.posterior_samples(feats, mask, survey_meta, n=32)
    assert samples.shape == (32, 1, len(THETA_KEYS))
