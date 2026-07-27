# Models

Place the person-segmentation model used by the AI background blur preset here:

```text
assets/models/person-segmentation.onnx
```

ONNX Runtime is the inference engine. The `.onnx` file is the model. The current
adapter expects a person-segmentation model with RGB float input in `1x3xHxW`
layout and a mask-like float output.

When the model is loaded successfully, streaming prints:

```text
background_blur_mask_active=onnx
```

If the model is missing or incompatible, the stream keeps running with the
centered fallback mask and prints `background_blur_mask_active=center` or
`background_blur_mask_active=onnx_unavailable_center`.
