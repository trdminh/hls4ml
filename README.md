# HLS4ML

A Python library for converting Deep Learning models into synthesizable C/C++ High-Level Synthesis (HLS) code for FPGA deployment.
---

## 1. Project Structure

```text
hls/
│
├── pyproject.toml
├── README.md
├── examples/
│   ├── mlp.py
│   ├── cnn.py
│   └── export.py
│
└── model2hls/
    │
    ├── __init__.py
    │
    ├── exporter.py
    ├── quantizer.py
    ├── graph.py
    │
    ├── hls/
    │   ├── __init__.py
    │   ├── writer.py
    │   ├── template.py
    │   ├── pragma.py
    │   ├── interface.py
    │   └── project.py
    │
    ├── layers/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── dense.py
    │   ├── conv2d.py
    │   ├── relu.py
    │   ├── maxpool.py
    │   ├── softmax.py
    │   ├── flatten.py
    │   ├── batchnorm.py
    │   └── activation.py
    │
    ├── parsers/
    │   ├── keras.py
    │   ├── onnx.py
    │   └── tflite.py
    │
    ├── model/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── sequential.py
    │   └── graph.py
    │
    ├── templates/
    │   ├── dense.cpp.j2
    │   ├── dense.h.j2
    │   ├── conv.cpp.j2
    │   ├── model.cpp.j2
    │   ├── model.h.j2
    │   └── testbench.cpp.j2
    │
    └── utils/
        ├── io.py
        ├── logger.py
        └── math.py
```


