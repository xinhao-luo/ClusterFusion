from enum import Enum

class TensorLayout(Enum):
    NHD = 0
    HND = 1

def _check_kv_layout(kv_layout: str) -> None:
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalid kv_layout {}".format(kv_layout))