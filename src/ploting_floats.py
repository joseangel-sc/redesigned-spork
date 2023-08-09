import torch
import numpy as np

ENCODING_CONFIG = {
    'E4M3': {
        'exp_bits': 4,
        'mantissa_bits': 3,
    },
    'E5M2': {
        'exp_bits': 5,
        'mantissa_bits': 2,
    },

}


def stochastic_round(value):
    floor_value = int(value)
    prob = value - floor_value
    return floor_value + np.random.choice([0, 1], p=[1 - prob, prob])


def to_fp8(tensor, encoding='E4M3', mantissa_bits=None):
    tensor = tensor.to(torch.float32)
    exp_bits = ENCODING_CONFIG[encoding]['exp_bits']
    if mantissa_bits is None:
        mantissa_bits = ENCODING_CONFIG[encoding]['mantissa_bits']

    exp_bias = (1 << (exp_bits - 1)) - 1
    exp_mask = (1 << exp_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    fp8_tensor = torch.zeros_like(tensor, dtype=torch.uint8)

    for index in np.ndindex(tensor.shape):
        value = tensor[index].item()
        raw_bits = np.float32(value).view(np.int32)
        sign = (raw_bits >> 31) & 1
        exponent = (raw_bits >> 23) & 0xFF
        mantissa = raw_bits & 0x7FFFFF
        print(f"value: {value}, sign: {sign}, exponent: {exponent}, mantissa: {mantissa}")

        if encoding == 'E4M3' and exponent == 0xFF:
            exponent = 0xF
            mantissa = 0 if np.isnan(value) else mantissa_mask
        exponent = (exponent - 127 + exp_bias) & exp_mask
        mantissa >>= (23 - mantissa_bits)
        rounded_mantissa = stochastic_round(mantissa / float(mantissa_mask) * mantissa_mask)
        fp8_value = (sign << (exp_bits + mantissa_bits)) | (exponent << mantissa_bits) | rounded_mantissa
        fp8_tensor[index] = fp8_value

    return fp8_tensor


def to_fp16(fp8_tensor, encoding='E4M3', mantissa_bits=None):
    exp_bits = ENCODING_CONFIG[encoding]['exp_bits']
    if mantissa_bits is None:
        mantissa_bits = ENCODING_CONFIG[encoding]['mantissa_bits']

    exp_bias_fp8 = (1 << (exp_bits - 1)) - 1
    exp_bias_fp16 = (1 << 4) - 1

    fp16_tensor = torch.zeros_like(fp8_tensor, dtype=torch.float16)

    for index in np.ndindex(fp8_tensor.shape):
        value = fp8_tensor[index].item()

        sign = (value >> (exp_bits + mantissa_bits)) & 1
        exponent = (value >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = value & ((1 << mantissa_bits) - 1)

        exponent = (exponent - exp_bias_fp8 + exp_bias_fp16) & 0x1F
        mantissa <<= (10 - mantissa_bits)

        fp16_value = (sign << 15) | (exponent << 10) | mantissa
        fp16_tensor[index] = torch.from_numpy(np.frombuffer(np.uint16(fp16_value).tobytes(), dtype=np.float16))

    return fp16_tensor
