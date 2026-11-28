# Neural Tone Mapper 1.1.4 (UE 5.8, Win64)

## Install

Copy this complete folder to <Project>/Plugins/NeuralToneMapper, enable the plugin, and restart the editor. Models are selected by highest numeric suffix; this build retains NTMO_M_v13.onnx. GPU inference uses Unreal's NNERuntimeRDG. If it is unavailable, the plugin falls back to its analytic curve.

## Model contract

The model predicts the global tone curve from display context only. Scene statistics feed the local Laplacian operator, not a day/night semantic classifier. Linear display values use 1.0 = 100 nits. The LUT stores linear-light mapped values; output transfer encoding is applied afterward: sRGB/Rec.709 for SDR, ST.2084 PQ or linear scRGB for HDR. PQ and scRGB therefore share the same linear-light teacher curve for equivalent display limits.

The v1.1.4 output pass fixes a critical HDR bug: post-LUT linear values are no longer clamped to 1.0 (100 nits) before PQ/scRGB encoding. HDR LUT output is also anchored at the active `r.HDR.Display.MidLuminance` value for an 18% neutral scene input; SDR does not use this HDR-only anchor. HDR output remains bounded by the active display peak. `NTMO_M_v14.onnx` keeps the v1.1.3 global curve head exactly and replaces only the local parameter head, trained from paired HDR+ 480p images.

## Validation status

The current AI is not yet the complete LLF-LUT method described in the algorithm reference: it predicts one frame-global curve and coarse local parameters; it does not predict pixel-level weights over multiple basis 3D LUTs or reconstruct a full learned LLF pyramid. The v14 supervised fine-tune uses 675 training pairs, 13 validation pairs, and 234 held-out test pairs. On the test split, local-parameter MAE fell from 0.3596 to 0.2320 (35.5%); the global curve is numerically unchanged. This is a parameter-label metric, not an image-quality score. PNG input is treated as relative linear RGB and JPEG reference as sRGB; the 16-bit PNGs have no embedded transfer/color profile, so absolute radiometric accuracy is not claimed. Unreal Engine 5.8 Win64 Editor Development, Game Development, and Shipping C++ targets build successfully. Runtime shader compilation and captured SDR/HDR image comparison remain required before visual approval.
