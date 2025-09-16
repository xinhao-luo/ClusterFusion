from typing import List


def generate_additional_params(
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    is_sm90_template: bool = False,
):
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_tensor_dtypes,
                additional_tensor_names,
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(additional_scalar_dtypes, additional_scalar_names)
        ]
    )
    additional_func_params = "".join(
        [
            (
                f", std::optional<at::Tensor> {var}"
                if var.startswith("maybe")
                else f", at::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(additional_scalar_dtypes, additional_scalar_names)
        ]
    )
    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
            ]
            + [
                f"params.additional_params.{var} = {var};"
                for var in additional_scalar_names
            ]
        )
    else:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)
