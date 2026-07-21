#!/usr/bin/env python3
"""Convert the legacy InsightFace MXNet checkpoint to ONNX.

The conversion must run in an environment that can import MXNet 1.9.1. MXNet
is only needed for this one-time conversion; production inference can use
onnxruntime-gpu without importing MXNet.

Default source checkpoint:
    models/celeb_detection/models/model-r100-ii/model-0000.params
    models/celeb_detection/models/model-r100-ii/model-symbol.json

Example:
    python convert_mxnet_to_onnx.py --verify

Known-compatible conversion environment:
    pip install numpy==1.19.5 mxnet==1.9.1 onnx==1.10.2 protobuf==3.20.3
    pip install onnxruntime==1.10.0  # only needed for --verify
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PREFIX = (
    REPO_ROOT
    / "models"
    / "celeb_detection"
    / "models"
    / "model-r100-ii"
    / "model"
)
DEFAULT_OUTPUT = DEFAULT_PREFIX.parent / "model-r100-ii.onnx"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the existing MXNet R100 face-embedding checkpoint to "
            "an ONNX model with a dynamic batch dimension."
        )
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help=(
            "MXNet checkpoint prefix, without '-symbol.json' or "
            "'-NNNN.params' (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="MXNet checkpoint epoch (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination ONNX file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-layer",
        default="fc1",
        help="MXNet internal layer to export (default: %(default)s)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=112,
        help="Input image height (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=112,
        help="Input image width (default: %(default)s)",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export a fixed batch size instead of a dynamic batch dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Concrete batch size used for shape inference; also the fixed "
            "batch size with --static-batch (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Compare MXNet and ONNX Runtime CPU outputs using deterministic "
            "synthetic inputs (requires onnxruntime)"
        ),
    )
    parser.add_argument(
        "--verify-batch-size",
        type=int,
        default=2,
        help="Batch size for numerical verification (default: %(default)s)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for numerical verification (default: %(default)s)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for numerical verification (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose MXNet exporter output",
    )
    return parser.parse_args(argv)


def checkpoint_paths(prefix: Path, epoch: int) -> Tuple[Path, Path]:
    symbol_path = Path(f"{prefix}-symbol.json")
    params_path = Path(f"{prefix}-{epoch:04d}.params")
    return symbol_path, params_path


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_mxnet():
    try:
        import mxnet as mx
    except ImportError as exc:
        raise RuntimeError(
            "MXNet is required for conversion. Run this script in the legacy "
            "Python 3.8 environment containing mxnet==1.9.1 or "
            "mxnet-cu101==1.9.1."
        ) from exc
    return mx


def select_output_symbol(mx, symbol, output_layer: str):
    output_name = (
        output_layer if output_layer.endswith("_output") else f"{output_layer}_output"
    )
    internals = symbol.get_internals()
    available = set(internals.list_outputs())
    if output_name not in available:
        similar = sorted(name for name in available if output_layer in name)
        hint = f" Similar outputs: {similar[:20]}" if similar else ""
        raise ValueError(f"MXNet output '{output_name}' was not found.{hint}")
    return internals[output_name]


def model_input_names(symbol, arg_params, aux_params) -> list[str]:
    parameter_names = set(arg_params) | set(aux_params)
    return [name for name in symbol.list_arguments() if name not in parameter_names]


def infer_and_report_shape(symbol, input_name: str, input_shape: tuple[int, ...]) -> None:
    _, output_shapes, _ = symbol.infer_shape(**{input_name: input_shape})
    output_names = symbol.list_outputs()
    shapes = dict(zip(output_names, output_shapes))
    print(f"Input:  {input_name} {input_shape} float32")
    for name, shape in shapes.items():
        print(f"Output: {name} {shape} float32")


def prepare_onnx_for_runtime(onnx, model):
    """Fix legacy mx2onnx output for correct and optimized ONNX inference.

    MXNet's PReLU gamma is stored as a one-dimensional channel vector. ONNX
    uses multidirectional (trailing-dimension) broadcasting, so a [C] slope
    is incorrectly matched with the image width in an NCHW tensor. Reshaping
    the same values to [1, C, 1, 1] preserves MXNet's channel-wise behavior.

    The MXNet 1.9 exporter also lists every initializer as a graph input. That
    legacy ONNX convention makes ONNX Runtime treat weights as overridable and
    prevents constant folding. Initializers remain in graph.initializer after
    being removed from graph.input.
    """
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    fixed_prelu_slopes = 0
    for node in model.graph.node:
        if node.op_type != "PRelu" or len(node.input) < 2:
            continue
        slope = initializers.get(node.input[1])
        if slope is None:
            raise ValueError(
                f"PRelu node '{node.name}' has a non-constant slope '{node.input[1]}'"
            )
        if len(slope.dims) == 1:
            channels = slope.dims[0]
            del slope.dims[:]
            slope.dims.extend((1, channels, 1, 1))
            fixed_prelu_slopes += 1

    producer_by_output = {
        output_name: node for node in model.graph.node for output_name in node.output
    }
    consumers_by_input = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers_by_input.setdefault(input_name, []).append(node)

    fixed_fully_connected = 0
    for gemm in model.graph.node:
        if gemm.op_type != "Gemm" or not gemm.input:
            continue
        flatten = producer_by_output.get(gemm.input[0])
        if flatten is None or flatten.op_type != "Flatten":
            continue
        axis_attributes = [attr for attr in flatten.attribute if attr.name == "axis"]
        if len(axis_attributes) != 1 or axis_attributes[0].i != -1:
            continue
        output_consumers = consumers_by_input.get(gemm.output[0], [])
        if len(output_consumers) != 1 or output_consumers[0].op_type != "Reshape":
            continue

        # MXNet FullyConnected defaults to flatten=True: preserve the batch
        # dimension and flatten every remaining dimension before GEMM.
        axis_attributes[0].i = 1
        reshape = output_consumers[0]
        reshape.op_type = "Identity"
        del reshape.input[1:]
        del reshape.attribute[:]
        fixed_fully_connected += 1

    initializer_names = set(initializers)
    runtime_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]
    removed_initializer_inputs = len(model.graph.input) - len(runtime_inputs)
    del model.graph.input[:]
    model.graph.input.extend(runtime_inputs)

    # mx2onnx already populated value_info using the pre-correction graph.
    # Regenerate those derived intermediate shapes after the rewrites above.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    print(
        "ONNX runtime preparation: "
        f"reshaped {fixed_prelu_slopes} PReLU slopes; "
        f"fixed {fixed_fully_connected} FullyConnected nodes; "
        f"removed {removed_initializer_inputs} initializer inputs"
    )
    return model


def export_model(args: argparse.Namespace) -> Path:
    if args.epoch < 0:
        raise ValueError("--epoch must be non-negative")
    if args.batch_size < 1 or args.verify_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.height < 1 or args.width < 1:
        raise ValueError("input dimensions must be positive")

    prefix = args.prefix.expanduser().resolve()
    output = args.output.expanduser().resolve()
    symbol_path, params_path = checkpoint_paths(prefix, args.epoch)
    require_file(symbol_path, "MXNet symbol")
    require_file(params_path, "MXNet parameters")

    if output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --force to replace it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    mx = load_mxnet()
    try:
        import numpy as np
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The conversion environment must contain numpy and onnx."
        ) from exc

    source_symbol, arg_params, aux_params = mx.model.load_checkpoint(
        str(prefix), args.epoch
    )
    output_symbol = select_output_symbol(mx, source_symbol, args.output_layer)
    input_names = model_input_names(output_symbol, arg_params, aux_params)
    if input_names != ["data"]:
        raise ValueError(
            "Expected the checkpoint to have exactly one input named 'data', "
            f"but found {input_names}"
        )

    input_name = input_names[0]
    input_shape = (args.batch_size, 3, args.height, args.width)
    infer_and_report_shape(output_symbol, input_name, input_shape)

    dynamic = not args.static_batch
    dynamic_shapes = [("batch", 3, args.height, args.width)] if dynamic else None

    temp_handle, temp_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".onnx", dir=str(output.parent)
    )
    os.close(temp_handle)
    temp_output = Path(temp_name)
    temp_output.unlink()

    try:
        print(f"Exporting {symbol_path}")
        print(f"Using parameters {params_path}")
        mx.onnx.export_model(
            output_symbol,
            [arg_params, aux_params],
            [input_shape],
            [np.float32],
            str(temp_output),
            verbose=args.verbose,
            dynamic=dynamic,
            dynamic_input_shapes=dynamic_shapes,
            run_shape_inference=True,
        )

        onnx_model = onnx.load(str(temp_output))
        onnx_model = prepare_onnx_for_runtime(onnx, onnx_model)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(temp_output))
        print("ONNX structural validation passed")

        if args.verify:
            verify_numerical_parity(
                mx=mx,
                np=np,
                symbol=output_symbol,
                arg_params=arg_params,
                aux_params=aux_params,
                onnx_path=temp_output,
                input_name=input_name,
                batch_size=args.verify_batch_size,
                height=args.height,
                width=args.width,
                atol=args.atol,
                rtol=args.rtol,
            )

        os.replace(temp_output, output)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    return output


def verify_numerical_parity(
    *,
    mx,
    np,
    symbol,
    arg_params,
    aux_params,
    onnx_path: Path,
    input_name: str,
    batch_size: int,
    height: int,
    width: int,
    atol: float,
    rtol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "--verify requires onnxruntime (the CPU package is sufficient)."
        ) from exc

    rng = np.random.RandomState(20260721)
    input_data = rng.uniform(
        low=0.0, high=255.0, size=(batch_size, 3, height, width)
    ).astype(np.float32)

    module = mx.mod.Module(symbol=symbol, context=mx.cpu(), label_names=None)
    module.bind(
        for_training=False,
        data_shapes=[(input_name, input_data.shape)],
    )
    module.set_params(
        arg_params,
        aux_params,
        allow_missing=False,
        allow_extra=True,
    )
    module.forward(
        mx.io.DataBatch(data=(mx.nd.array(input_data),)),
        is_train=False,
    )
    expected = module.get_outputs()[0].asnumpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {session.get_inputs()[0].name: input_data})[0]

    if expected.shape != actual.shape:
        raise AssertionError(
            f"Output shape mismatch: MXNet {expected.shape}, ONNX {actual.shape}"
        )

    difference = np.abs(expected - actual)
    max_abs_error = float(difference.max(initial=0.0))
    denominator = np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
    max_rel_error = float((difference / denominator).max(initial=0.0))

    expected_norm = expected / np.maximum(
        np.linalg.norm(expected, axis=1, keepdims=True),
        np.finfo(np.float32).tiny,
    )
    actual_norm = actual / np.maximum(
        np.linalg.norm(actual, axis=1, keepdims=True),
        np.finfo(np.float32).tiny,
    )
    minimum_cosine = float(np.sum(expected_norm * actual_norm, axis=1).min())

    print(
        "Numerical comparison: "
        f"max_abs_error={max_abs_error:.8g}, "
        f"max_rel_error={max_rel_error:.8g}, "
        f"minimum_cosine={minimum_cosine:.10f}"
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    print("MXNet/ONNX numerical parity passed")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = export_model(args)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
