import pathlib
import torch

def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.int32: "int32_t",
    torch.uint32: "uint32_t",
    torch.int64: "int64_t",
    torch.uint64: "uint64_t",
}

filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int32: "i32",
    torch.uint32: "u32",
    torch.int64: "i64",
    torch.uint64: "u64",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

mask_mode_literal = {
    0: "MaskMode::kNone",
    1: "MaskMode::kCausal",
    2: "MaskMode::kCustom",
    3: "MaskMode::kMultiItemScoring",
}