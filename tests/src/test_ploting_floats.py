import torch
import numpy as np

from src.ploting_floats import to_fp8

from src.ploting_floats import to_fp16

from src.ploting_floats import stochastic_round


def test_to_fp8_E4M3():
    tensor = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    fp8_tensor = to_fp8(tensor, encoding='E4M3')
    assert isinstance(fp8_tensor, torch.Tensor)
    assert fp8_tensor.dtype == torch.uint8


def test_to_fp8_E5M2():
    tensor = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    fp8_tensor = to_fp8(tensor, encoding='E5M2')
    assert isinstance(fp8_tensor, torch.Tensor)
    assert fp8_tensor.dtype == torch.uint8


def test_to_fp16_E4M3():
    fp8_tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
    fp16_tensor = to_fp16(fp8_tensor, encoding='E4M3')
    assert isinstance(fp16_tensor, torch.Tensor)
    assert fp16_tensor.dtype == torch.float16


def test_to_fp16_E5M2():
    fp8_tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
    fp16_tensor = to_fp16(fp8_tensor, encoding='E5M2')
    assert isinstance(fp16_tensor, torch.Tensor)
    assert fp16_tensor.dtype == torch.float16


def test_round_trip_conversion():
    original_tensor = torch.tensor([1.0, 2.5, -3.3], dtype=torch.float16)

    # Convert to fp8
    fp8_tensor = to_fp8(original_tensor, encoding='E4M3')

    # Convert back to fp16
    recovered_tensor = to_fp16(fp8_tensor, encoding='E4M3')

    # Check that the recovered tensor is close to the original tensor
    assert torch.allclose(original_tensor, recovered_tensor, atol=0.1)  # Increased tolerance


def test_stochastic_round_to_self():
    assert stochastic_round(5) == 5
    assert stochastic_round(10) == 10
    assert stochastic_round(0) == 0


def test_stochastic_round_up():
    value = 3.6
    prob = value - int(value)

    results = [stochastic_round(value) for _ in range(10000)]

    count_round_up = sum(1 for result in results if result == int(value) + 1)
    # print(f"Counted round up: {count_round_up} out of {len(results)}")
    tolerance = 0.05
    assert np.isclose(count_round_up / len(results), prob, atol=tolerance)


def test_stochastic_round_down():
    value = 3.3
    prob = 1 - (value - int(value))

    results = [stochastic_round(value) for _ in range(10000)]

    count_round_down = sum(1 for result in results if result == int(value))
    # print(f"Counted round down: {count_round_down} out of {len(results)}")
    tolerance = 0.05
    assert np.isclose(count_round_down / len(results), prob, atol=tolerance)
