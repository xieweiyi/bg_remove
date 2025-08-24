with open('deform_conv2d_onnx_exporter.py') as fp:
    file_lines = fp.read()

file_lines = file_lines.replace(
    "return sym_help._get_tensor_dim_size(tensor, dim)",
    '''
    tensor_dim_size = sym_help._get_tensor_dim_size(tensor, dim)
    if tensor_dim_size == None and (dim == 2 or dim == 3):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()

        tensor_dim_size = x_strides[2] if dim == 3 else x_strides[1] // x_strides[2]
    elif tensor_dim_size == None and (dim == 0):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()
        tensor_dim_size = x_strides[3]

    return tensor_dim_size
    ''',
)

with open('deform_conv2d_onnx_exporter.py', mode="w") as fp:
    fp.write(file_lines)