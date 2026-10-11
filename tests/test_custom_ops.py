"""Registration, mathematical gradient and graph-capture checks for exercise 11."""

import pytest
import torch

from gpu_prep.custom_ops import bias_relu, bias_relu_triton


@pytest.fixture(params=[bias_relu, bias_relu_triton], ids=["cuda_registration", "triton_registration"])
def operation(request):
    return request.param


@pytest.mark.parametrize("shape", [(3, 5), (0, 5), (3, 0)])
def test_custom_op_forward_and_backward_match_reference(shape, operation):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(shape[1], dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    br = bias.detach().clone().requires_grad_()
    actual = operation(x, bias)
    expected = torch.relu(xr + br)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(bias.grad, br.grad)
    assert actual.is_contiguous()


def test_relu_at_zero_uses_zero_derivative(operation):
    x = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    operation(x, bias).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(bias.grad, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_inactive_relu_discards_nonfinite_upstream_gradient(operation):
    x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    br = bias.detach().clone().requires_grad_()
    upstream = torch.tensor([[float("inf"), float("nan"), 1.0]], dtype=torch.float64)
    actual = torch.autograd.grad(operation(x, bias), (x, bias), grad_outputs=upstream)
    expected = torch.autograd.grad(torch.relu(xr + br), (xr, br), grad_outputs=upstream)
    for gradient, reference in zip(actual, expected):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual[0], torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(actual[1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_custom_op_gradcheck_away_from_relu_kink(operation):
    x = torch.tensor([[-2.0, 2.0, -3.0], [1.0, -4.0, 3.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(operation, (x, bias))


@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (2, 0)])
def test_custom_op_registration_contract(shape, operation):
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(shape[1], dtype=torch.float64, requires_grad=True)
    results = torch.library.opcheck(operation, (x, bias))
    assert all(result == "SUCCESS" for result in results.values())


def test_fake_metadata_without_running_cpu_or_cuda_data_kernel(operation):
    # FakeTensorMode is a diagnostic utility under torch._subclasses; production
    # fake registration uses the public register_fake API in custom_ops.py.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 3, dtype=torch.float64)
        bias = torch.empty(3, dtype=torch.float64)
        output = operation(x, bias)
        assert output.shape == (2, 3)
        assert output.stride() == (3, 1)
        assert output.dtype == x.dtype
        assert output.device == x.device


def test_fullgraph_capture_with_eager_backend_and_backward(operation):
    # backend=eager verifies Dynamo capture; it does not prove Inductor codegen,
    # AOT training compilation or GPU speedup. Those are separate experiments.
    fn = lambda x, b: operation(x, b).square().sum()
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    x = torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.3, 0.2, -0.1], dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    br = b.detach().clone().requires_grad_()
    actual = compiled(x, b)
    expected = fn(xr, br)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(b.grad, br.grad)


def test_invalid_contract_is_rejected(operation):
    with pytest.raises(ValueError, match="length"):
        operation(torch.randn(2, 3), torch.randn(4))
    with pytest.raises(ValueError, match="contiguous"):
        operation(torch.randn(3, 2).t(), torch.randn(3))
    with pytest.raises(TypeError, match="same dtype"):
        operation(torch.randn(2, 3), torch.randn(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="expects"):
        operation(torch.randn(3), torch.randn(3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA runtime, GPU and nvcc extension build")
def test_cuda_custom_op_forward_and_backward(operation):
    torch.manual_seed(11)
    x = torch.randn(17, 33, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(33, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    br = bias.detach().clone().requires_grad_()
    actual = operation(x, bias)
    expected = torch.relu(xr + br)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(bias.grad, br.grad)
    results = torch.library.opcheck(operation, (x, bias))
    assert all(result == "SUCCESS" for result in results.values())
