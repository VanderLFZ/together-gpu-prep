"""Exercises 10--11: graph breaks, tensor control flow and custom-op capture.

CPU capture smoke check:
  python -m examples.compile_demo --device cpu --backend eager
Actual Inductor experiment (requires an appropriate installed toolchain):
  python -m examples.compile_demo --device cuda --backend inductor

backend='eager' tests Dynamo graph capture and result agreement, not Inductor
code generation or GPU acceleration. Results are measured on YOUR machine.
"""

import argparse
import statistics
import time

import torch

from gpu_prep.custom_ops import bias_relu, bias_relu_triton


def bad_python_branch(x: torch.Tensor) -> torch.Tensor:
    # .item() transfers a data value to Python; the if depends on that value.
    if x.sum().item() > 0:
        return x.sin()
    return x.cos()


def tensor_where(x: torch.Tensor) -> torch.Tensor:
    # Semantically simple and easy to capture. Both sin/cos expressions are
    # evaluated before where selects their results; it is not lazy branching.
    return torch.where(x.sum() > 0, x.sin(), x.cos())


def tensor_cond(x: torch.Tensor) -> torch.Tensor:
    # Structured control flow: branch outputs need matching metadata and the
    # branches must meet torch.cond's capture restrictions.
    return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_wall_ms(fn, fn_args, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn(*fn_args)
    _sync(device)
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn(*fn_args)
        _sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("eager", "inductor"), default="eager")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--include-cond", action="store_true",
                        help="also test structured torch.cond (availability/limits vary by PyTorch release)")
    parser.add_argument("--operator-backend", choices=("cuda", "triton", "both"), default="both",
                        help="custom-op GPU implementation; both use CPU references when --device cpu")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        raise SystemExit("repeats must be positive and warmup nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; use --device cpu for capture-only checks")
    print("PyTorch:", torch.__version__, "device:", device, "backend:", args.backend)
    if args.backend == "eager":
        print("Scope: Dynamo capture/result checks. This does NOT benchmark Inductor or prove GPU speedup.")
    torch.manual_seed(2027)
    positive = torch.ones(8192, device=device)
    negative = -positive
    with torch.no_grad():
        try:
            broken = torch.compile(bad_python_branch, backend=args.backend, fullgraph=True)
            broken(positive)
        except Exception as error:
            print("Expected fullgraph failure for data-dependent Python branch:", type(error).__name__)
            print(str(error).splitlines()[0][:250])
        else:
            print("This PyTorch/configuration captured the example; inspect its graph and guards rather than assuming a break.")

        functions = [("where", tensor_where)]
        if args.include_cond:
            functions.append(("cond", tensor_cond))
        for name, fn in functions:
            compiled = torch.compile(fn, backend=args.backend, fullgraph=True)
            _sync(device)
            start = time.perf_counter()
            first = compiled(positive)
            _sync(device)
            first_call_ms = (time.perf_counter() - start) * 1000
            torch.testing.assert_close(first, bad_python_branch(positive))
            torch.testing.assert_close(compiled(negative), bad_python_branch(negative))
            eager_ms = median_wall_ms(fn, (positive,), device, args.warmup, args.repeats)
            compiled_ms = median_wall_ms(compiled, (positive,), device, args.warmup, args.repeats)
            print(f"{name}: first compiled call including capture/codegen/initialization = {first_call_ms:.3f} ms")
            print(f"{name}: steady median wall time eager={eager_ms:.4f} ms, compiled={compiled_ms:.4f} ms")

        x = torch.randn(17, 33, device=device)
        bias = torch.randn(33, device=device)
        operators = []
        if args.operator_backend in ("cuda", "both"):
            operators.append(("CUDA wrapper", bias_relu))
        if args.operator_backend in ("triton", "both"):
            operators.append(("Triton wrapper", bias_relu_triton))
        for name, operation in operators:
            # First GPU use lazily builds/JITs the teaching implementation. Do it
            # before capture; compilation is not steady-state kernel latency.
            torch.testing.assert_close(operation(x, bias), torch.relu(x + bias))
            op_graph = torch.compile(lambda a, b: operation(a, b).square().sum(),
                                     backend=args.backend, fullgraph=True)
            torch.testing.assert_close(op_graph(x, bias), torch.relu(x + bias).square().sum())
            print(name + ": fullgraph capture and numerical agreement passed for this device/backend.")

    # Keep the registration check independent of the timing region.
    cpu_x = torch.tensor([[-2.0, 2.0, -3.0], [1.0, -4.0, 3.0]], dtype=torch.float64, requires_grad=True)
    cpu_bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    for name, operation in operators:
        print(name, "opcheck (registration):", torch.library.opcheck(operation, (cpu_x, cpu_bias)))
        print(name, "gradcheck (mathematical derivative):", torch.autograd.gradcheck(operation, (cpu_x, cpu_bias)))


if __name__ == "__main__":
    main()
