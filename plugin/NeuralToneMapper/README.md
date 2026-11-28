# Neural Tone Mapper 1.1.3 (UE 5.8, Win64)

## Install

Copy this complete folder to <Project>/Plugins/NeuralToneMapper, enable the plugin, and restart the editor. Models are selected by highest numeric suffix; v1.1.3 ships NTMO_M_v13.onnx. GPU inference uses Unreal's NNERuntimeRDG. If it is unavailable, the plugin falls back to its analytic curve.

## Model contract

The model predicts the global tone curve from display context only. Scene statistics feed the local Laplacian operator, not a day/night semantic classifier. Linear display values use 1.0 = 100 nits. The LUT stores linear-light mapped values; output transfer encoding is applied afterward: sRGB/Rec.709 for SDR, ST.2084 PQ or linear scRGB for HDR. PQ and scRGB therefore share the same linear-light teacher curve for equivalent display limits.

The v1.1.3 training update widens the curve head's representable black-point range to the declared log2 interval [-14, 8], caps its upper endpoint, and calibrates its display-context head with paired PQ/scRGB contexts. The local head and normalization statistics remain from v12 because available local-operation labels are not validated against UE golden renders.

## Validation status

The Win64 plugin has been built against UE 5.8. The ONNX export passes graph/operator, numerical parity, and monotonicity checks. UE runtime shader compilation and captured SDR/HDR image comparison remain required before treating this beta candidate as visually approved.
