"""CPU-only mathematical checks; no model or tokenizer downloads."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from jlens import HFLens, LensDictionary, estimate_jacobians, mean_transport, readout, project_direction, neuron_directions, main


@pytest.fixture
def tiny():
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=19, hidden_size=8, intermediate_size=16, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=32, attention_dropout=0.0,
    )).double().eval()
    model.config._attn_implementation = "eager"
    return HFLens(model), [torch.tensor([[1, 4, 2, 8, 3]]), torch.tensor([[2, 7, 5, 1]])]


def explicit(adapter, ids, layer, target, skip=0):
    # Independent full source-position x target-position Jacobian.
    baseline = adapter.capture(ids, [layer])[layer]
    fn = lambda h: adapter.forward_from_residual(ids, layer, h, target)
    jac = torch.autograd.functional.jacobian(fn, baseline)
    valid = range(skip, ids.shape[1] - 1)
    return sum(jac[0, u, :, 0, t, :] for t in valid for u in valid) / len(valid)


def test_jvp_and_fitted_matrix_match_explicit_jacobian(tiny):
    adapter, prompts = tiny
    direction = torch.randn(2, 8, dtype=torch.float64)
    expected = torch.stack([explicit(adapter, ids, 0, 2, 1) for ids in prompts]).mean(0)
    result = mean_transport(adapter, prompts, direction, 0, target_layer=2, skip_first=1)
    # HF Qwen3RMSNorm explicitly casts to fp32, even on a double model.
    torch.testing.assert_close(result, direction @ expected.T, rtol=2e-6, atol=1e-7)
    fitted = estimate_jacobians(adapter, prompts, [0, 1], target_layer=2, skip_first=1, dim_batch=2)
    torch.testing.assert_close(fitted[0].double(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fitted[1].double(), torch.stack([
        explicit(adapter, ids, 1, 2, 1) for ids in prompts]).mean(0), rtol=1e-5, atol=1e-6)
    assert fitted[0].shape == (8, 8)


def test_readout_normalizes_after_corpus_average_and_is_deterministic(tiny):
    adapter, prompts = tiny
    direction = torch.randn(2, 8, dtype=torch.float64)
    a = readout(adapter, prompts, direction, 0, target_layer=1, skip_first=0)
    b = readout(adapter, prompts, direction, 0, target_layer=1, skip_first=0)
    assert a["logits"].shape == (2, 19)
    assert a["logit_lens_logits"].shape == (2, 19)
    torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    expected_j = torch.stack([explicit(adapter, ids, 0, 1) for ids in prompts]).mean(0)
    expected = adapter.unembed(direction @ expected_j.T)
    torch.testing.assert_close(a["logits"].double(), expected, rtol=1e-6, atol=1e-7)
    assert not any(p.grad is not None for p in adapter.model.parameters())
    assert not any(block._forward_hooks for block in adapter.layers)


def test_sparse_nonnegative_projection_and_signed_asymmetry():
    dictionary = torch.eye(4, dtype=torch.float64)
    x = torch.tensor([2., 1., 0., 0.], dtype=torch.float64)
    result = project_direction(x, dictionary, k=2)
    torch.testing.assert_close(result["component"], x)
    assert result["explained_energy"] == pytest.approx(1)
    assert (result["coefficients"] >= 0).all()
    negative = project_direction(-x, dictionary, k=2)
    assert negative["explained_energy"] == pytest.approx(0)
    assert negative["token_ids"].dtype == torch.long
    assert dictionary[negative["token_ids"]].shape == (0, 4)
    assert project_direction(torch.zeros(4), dictionary, k=2)["explained_energy"] == 0


def test_neuron_write_vectors_and_invalid_inputs(tiny):
    adapter, prompts = tiny
    result = neuron_directions(adapter, 0, [1, 3], [2., -1.])
    expected = adapter.layers[0].mlp.down_proj.weight[:, [1, 3]].T
    torch.testing.assert_close(result, expected * torch.tensor([2., -1.])[:, None])
    with pytest.raises(ValueError):
        mean_transport(adapter, prompts, torch.ones(9), 0, target_layer=2)
    with pytest.raises(ValueError):
        estimate_jacobians(adapter, [torch.tensor([[1]])], [0], skip_first=0)
    with pytest.raises(ValueError):
        estimate_jacobians(adapter, prompts, [2], target_layer=1)


def test_other_decoder_layout():
    torch.manual_seed(7)
    adapter = HFLens(GPT2LMHeadModel(GPT2Config(
        vocab_size=13, n_embd=8, n_layer=2, n_head=2, n_positions=16,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )).double().eval())
    adapter.model.config._attn_implementation = "eager"
    ids = torch.tensor([[1, 2, 3, 4]])
    x = torch.randn(8, dtype=torch.float64)
    actual = mean_transport(adapter, [ids], x, 0, target_layer=1, skip_first=0)
    torch.testing.assert_close(actual, x @ explicit(adapter, ids, 0, 1).T)


def test_lazy_dictionary_matches_explicit_rows(tiny):
    adapter, _ = tiny
    j = torch.randn(8, 8)
    lazy = LensDictionary(adapter, j, chunk_size=3)
    dense = torch.nn.functional.normalize(adapter.head.weight.float() @ j, dim=1)
    x = torch.randn(8)
    torch.testing.assert_close(lazy.correlations(x), dense @ x)
    torch.testing.assert_close(lazy.rows([0, 7, 18]), dense[[0, 7, 18]])
    actual = project_direction(x, lazy, k=3)
    expected = project_direction(x, dense, k=3)
    # Greedy refinements amplify tiny GEMV/GEMM rounding differences.
    torch.testing.assert_close(actual["token_ids"], expected["token_ids"])
    torch.testing.assert_close(actual["component"], expected["component"], rtol=5e-4, atol=2e-4)
    assert actual["explained_energy"] == pytest.approx(expected["explained_energy"], abs=1e-5)


def test_resume_matches_uninterrupted_fit(tiny):
    adapter, prompts = tiny
    snapshots = []
    options = dict(target_layer=2, skip_first=0, dim_batch=2)
    full = estimate_jacobians(adapter, prompts, [0, 1], **options)
    estimate_jacobians(adapter, prompts[:1], [0, 1], **options,
                       checkpoint=lambda s: snapshots.append({"n_done": s["n_done"],
                           "jacobian_sum": {l: j.clone() for l, j in s["jacobian_sum"].items()}}))
    resumed = estimate_jacobians(adapter, prompts, [0, 1], **options, initial_state=snapshots[0])
    for layer in full:
        torch.testing.assert_close(full[layer], resumed[layer], rtol=0, atol=0)


def test_cpu_cli_local_model_readout_and_jspace(tiny, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    adapter, _ = tiny
    saved_model = tmp_path / "model"
    adapter.model.float().save_pretrained(saved_model)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(saved_model)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one two three four\nfour three one two\n")
    dirs = tmp_path / "dirs.pt"
    torch.save({0: torch.randn(8)}, dirs)
    common = ["--model", str(saved_model), "--prompts", str(prompts), "--device", "cpu", "--skip-first", "0"]
    space = tmp_path / "space.pt"
    main(["jspace", *common, "--layers", "0,1", "--sample-positions", "1", "--projection-k", "2", "--out", str(space)])
    direct = tmp_path / "direct.pt"
    main(["readout", *common, "--layer", "0", "--acts-dir-file", str(dirs), "--out", str(direct)])
    cached = tmp_path / "cached.pt"
    main(["readout", *common, "--layer", "0", "--acts-dir-file", str(dirs), "--lens-file", str(space), "--out", str(cached)])
    a = torch.load(direct, weights_only=True)
    b = torch.load(cached, weights_only=True)
    torch.testing.assert_close(a["logits"], b["logits"], rtol=2e-5, atol=1e-6)
    assert torch.load(space, weights_only=True)["activation_reconstruction"][0]["n_activations"] == 2
    with pytest.raises(FileExistsError):
        main(["jspace", *common, "--layers", "0", "--out", str(space)])
    # Same path and configuration, different local weight bytes: must reject reuse.
    with torch.no_grad():
        adapter.model.model.layers[0].mlp.down_proj.weight.add_(0.01)
    adapter.model.save_pretrained(saved_model)
    with pytest.raises(ValueError, match="model_fingerprint"):
        main(["readout", *common, "--layer", "0", "--acts-dir-file", str(dirs), "--lens-file", str(space), "--out", str(tmp_path / "changed.pt")])
    prompts.write_text("three two one four\n")
    with pytest.raises(ValueError, match="provenance mismatch"):
        main(["readout", *common, "--layer", "0", "--acts-dir-file", str(dirs), "--lens-file", str(space), "--out", str(tmp_path / "bad.pt")])
