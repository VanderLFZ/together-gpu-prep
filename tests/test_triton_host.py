"""CPU-only checks for Triton wrapper metadata; these do not execute GPU kernels."""

import math
import struct

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from gpu_prep.triton_ops import _check_eps


@pytest.mark.parametrize("eps", [1e-5, 1.0, 1e-30, 1e-45, 3e38])
def test_eps_is_positive_finite_after_float32_conversion(eps):
    result = _check_eps(eps)
    assert result == struct.unpack(">f", struct.pack(">f", eps))[0]
    assert math.isfinite(result) and result > 0


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), 1e-300, 1e300, 1e-46])
def test_eps_rejects_invalid_python_or_float32_values(eps):
    with pytest.raises(ValueError, match="eps"):
        _check_eps(eps)


@pytest.mark.parametrize("eps", [True, False, "1e-5", None])
def test_eps_rejects_nonreal_metadata(eps):
    with pytest.raises(TypeError, match="eps"):
        _check_eps(eps)
