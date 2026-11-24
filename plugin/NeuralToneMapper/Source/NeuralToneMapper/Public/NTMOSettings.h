// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "NTMOTypes.h"
#include "NTMOSettings.generated.h"

/** How the neural curve predictor is selected. */
UENUM()
enum class ENTMOCurveSource : uint8
{
	/** Built-in GPU analytic policy. Always available, no assets required. */
	Analytic UMETA(DisplayName = "Analytic (built-in)"),

	/** NNE neural model. Falls back to Analytic when no model / runtime is available. */
	Neural UMETA(DisplayName = "Neural (NNE)"),
};

/**
 * Project settings for the Neural Tone Mapper plugin.
 *
 * "Enable AI Tone Mapping" is the same flag that the editor's Build menu toggles, so
 * designers can drive it from either place.
 */
UCLASS(config = NeuralToneMapper, defaultconfig, meta = (DisplayName = "Neural Tone Mapper"))
class NEURALTONEMAPPER_API UNTMOSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UNTMOSettings();

	/**
	 * Master switch. When enabled the plugin replaces Unreal's built-in tone mapper
	 * (the engine's tonemapping pass and combined LUT are skipped entirely).
	 */
	UPROPERTY(config, EditAnywhere, Category = "General", meta = (DisplayName = "AI 动态色调映射 (Enable AI Tone Mapping)"))
	bool bEnabled = false;

	/** How the per-frame curve is produced. */
	UPROPERTY(config, EditAnywhere, Category = "General", meta = (DisplayName = "Curve Source"))
	ENTMOCurveSource CurveSource = ENTMOCurveSource::Neural;

	/**
	 * Trained UNNEModelData asset for the neural curve predictor.
	 * Only used when Curve Source is Neural. When unset the plugin logs once and
	 * falls back to the analytic policy, so the feature stays functional.
	 */
	// ---- Display overrides -------------------------------------------------
	// EDID is authoritative when it is available, but plenty of setups cannot report
	// it: projectors, virtual displays, capture cards, some laptops, and any machine
	// driving an HDR panel over a converter. These overrides exist for those cases,
	// and they are also what lets you verify the PQ / HDR adaptation path on an SDR
	// desktop monitor.
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override Display Metadata"))
	bool bOverrideDisplayMetadata = false;

	/** Panel peak luminance in nits (e.g. 1000 for a typical HDR LCD, 400 for SDR-ish). */
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override Peak Nits", ClampMin = "10.0", ClampMax = "10000.0", EditCondition = "bOverrideDisplayMetadata"))
	float OverridePeakNits = 1000.0f;

	/** Full-frame (ABL limited) peak. Lower than the panel peak on OLED / aggressive ABL panels. */
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override Full Frame Nits", ClampMin = "10.0", ClampMax = "10000.0", EditCondition = "bOverrideDisplayMetadata"))
	float OverrideFullFrameNits = 600.0f;

	/** Paper white: what SDR code value 1.0 should map to under HDR output. */
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override Paper White Nits", ClampMin = "10.0", ClampMax = "2000.0", EditCondition = "bOverrideDisplayMetadata"))
	float OverridePaperWhiteNits = 203.0f;

	/** 0=sRGB, 1=Rec709, 2=Explicit gamma, 3=ST2084 (PQ), 4=scRGB. */
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override Output Device (0=sRGB 1=709 2=Gamma 3=PQ 4=scRGB)", ClampMin = "0", ClampMax = "4", EditCondition = "bOverrideDisplayMetadata"))
	int32 OverrideOutputDevice = 3;

	/** 1 when the display path really is HDR, so the ABL term becomes active. */
	UPROPERTY(config, EditAnywhere, Category = "Display", meta = (DisplayName = "Override HDR Enabled", ClampMin = "0", ClampMax = "1", EditCondition = "bOverrideDisplayMetadata"))
	int32 OverrideHDREnabled = 1;
	/**
	 * Temporal smoothing of the predicted curve, in seconds (half-life).
	 *
	 * The network predicts each frame independently from image statistics, so a hard
	 * content change -- a camera cut, a light switching on, a bright object entering
	 * frame -- makes the curve jump, which reads on screen as the whole image flashing
	 * for a frame or two. An exponential moving average over the 8 knots removes it.
	 * 0 disables smoothing (useful to see the raw per-frame prediction).
	 *
	 * Typical values: 0.10 (responsive, slight residual flash) to 0.30 (very stable,
	 * noticeable lag on fast cuts). 0.15 is a good default for gameplay.
	 */
	/**
	 * Run the network once every N frames and reuse the last curve in between.
	 *
	 * Measured cost of the neural path is ~1.0 ms per frame at 1.55 Mpix, and it is
	 * dominated by the ONNX runtime's per-operator dispatches (15 ops, unfused, fp32
	 * intermediates), not by the arithmetic. The tone curve is a slow-moving, global
	 * statistic: adapting it at 30 Hz instead of 120 Hz is not observable, especially
	 * with the temporal smoothing above already blending successive curves.
	 *
	 * Interval 4 cuts the cost to ~0.25 ms; 1 means "every frame" (use this when
	 * profiling or when the content changes violently frame to frame).
	 */
	UPROPERTY(config, EditAnywhere, Category = "Neural", meta = (DisplayName = "Inference Interval (frames)", ClampMin = "1", ClampMax = "8"))
	int32 InferenceInterval = 4;
	/**
	 * How much of the learned DETAIL bands to apply, on top of the learned local
	 * exposure.
	 *
	 * 0   = pure local exposure: dark regions up, bright regions down. This is what
	 *       reads as local tone mapping, and 1.0.3 defaults to a quarter.
	 * 1   = also apply the Durand-style detail-restoration bands at full strength.
	 *       Measured side effect: those bands carry a positive DC (a concave tone curve
	 *       maps the blurred base systematically higher than the detailed signal), so
	 *       full strength lifts the whole frame by ~0.16 display luma and washes it out.
	 */
	/**
	 * Multiplier on the learned local-exposure term.
	 *
	 * The teacher asks for 0.35 stops of lift/compression per stop a region sits away
	 * from the frame mean, which over a normal scene accumulates to roughly a stop. That
	 * reads as too much on a standard SDR frame, so the shipped default halves it. 0
	 * disables local exposure entirely (detail bands only); 1 is the teacher's strength.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Local", meta = (DisplayName = "Local Exposure Scale", ClampMin = "0.0", ClampMax = "2.0"))
	float LocalExposureScale = 0.5f;
	UPROPERTY(config, EditAnywhere, Category = "Local", meta = (DisplayName = "Local Detail Mix", ClampMin = "0.0", ClampMax = "1.0"))
	float LocalDetailMix = 0.25f;
	UPROPERTY(config, EditAnywhere, Category = "Neural", meta = (DisplayName = "Curve Smoothing Half-Life (s)", ClampMin = "0.0", ClampMax = "2.0"))
	float CurveSmoothingHalfLife = 0.15f;
	UPROPERTY(config, EditAnywhere, Category = "Neural", meta = (DisplayName = "Curve Model Asset (UNNEModelData)", AllowedClasses = "/Script/NNE.NNEModelData"))
	FSoftObjectPath CurveModelAsset;

	/**
	 * Absolute path to the trained .onnx on disk. PREFERRED over the asset above.
	 *
	 * The model is built at runtime with UNNEModelData::Init("onnx", bytes), which is
	 * public API in 5.8, so there is no import step and no .uasset to keep in sync with
	 * the file export_onnx.py produces. Empty falls back to Curve Model Asset.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Neural", meta = (DisplayName = "Curve Model File (.onnx)"))
	FString CurveModelFile;

	/**
	 * NNE runtime used for inference. NNERuntimeRDGHlsl is the zero-sync path and
	 * should be preferred; NNERuntimeORTDml needs one frame of pipelining.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Neural", meta = (DisplayName = "NNE Runtime"))
	FString NNERuntimeName = TEXT("NNERuntimeRDGHlsl");

	/**
	 * Master intensity. 0 produces a near-neutral (ACES-like) look which is useful for
	 * A/B regression; 1 is the full effect.
	 */
	UPROPERTY(config, EditAnywhere, Category = "General", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Intensity"))
	float Intensity = 1.0f;

	/** Extra brightness preference, in stops. Applied on top of post process exposure. */
	UPROPERTY(config, EditAnywhere, Category = "General", meta = (ClampMin = "-3.0", ClampMax = "3.0", DisplayName = "Brightness Bias (stops)"))
	float UserBrightnessBias = 0.0f;

	/**
	 * How strongly the mapper respects the panel's full-frame (ABL) luminance limit.
	 * 0 disables ABL awareness, 1 compresses the shoulder aggressively for ABL-limited
	 * panels.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Display Adaptation", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "ABL Awareness"))
	float ABLAwareness = 1.0f;

	/**
	 * When the panel reports MaxFullFrameLuminance == MaxLuminance (typical for displays
	 * that do not populate EDID properly) assume this fraction instead.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Display Adaptation", meta = (ClampMin = "0.1", ClampMax = "1.0", DisplayName = "Assume FullFrame Fraction"))
	float AssumeFullFrameFraction = 0.6f;

	/** Highlight desaturation strength in [0,1]. Pushes very bright values toward white. */
	UPROPERTY(config, EditAnywhere, Category = "Look", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Highlight Desaturation"))
	float HighlightDesaturation = 0.15f;

	/** Saturation multiplier applied after the tone curve. */
	UPROPERTY(config, EditAnywhere, Category = "Look", meta = (ClampMin = "0.5", ClampMax = "2.0", DisplayName = "Saturation"))
	float Saturation = 1.0f;

	/** Black lift in linear units, for LCD panels with a poor black level. */
	UPROPERTY(config, EditAnywhere, Category = "Look", meta = (ClampMin = "0.0", ClampMax = "0.02", DisplayName = "Black Lift"))
	float BlackLift = 0.0f;

	/**
	 * Reproduce Unreal's own colour grading inside the LUT bake.
	 *
	 * Replacing the tonemapper makes Unreal skip AddCombineLUTPass, and that pass is
	 * where ExpandGamut (default 1.0), the shadow/midtone/highlight grading triples and
	 * the "expand bright saturated colours into a wider gamut" step all live. Without
	 * them the plugin's output is measurably flatter than Unreal's: on a night-sky
	 * scene it measured -21.5% mean HSV saturation and -48% chroma on the most
	 * saturated 0.6% of pixels, which reads on screen as a grey, washed-out image.
	 *
	 * Turning this ON restores exactly what the project asked for. Every parameter is
	 * read from the view's post process settings, and every one of them defaults to its
	 * identity value, so this changes nothing at all for a project that does not grade
	 * -- it can only put back grading that exists.
	 *
	 * Turn it OFF only to deliberately bypass the project's grading.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Look", meta = (DisplayName = "Match Unreal Colour Grading"))
	bool bMatchUnrealColorGrading = false;

	/**
	 * Match Unreal's HUE behaviour as well as its grading.
	 *
	 * Unreal's FilmToneMap compresses R, G and B independently, which is what makes
	 * ACES-derived curves shift hue on bright saturated colours (the "notorious six"),
	 * and what makes an Unreal sky read blue-violet. The plugin's default operator maps
	 * max(RGB) and scales the rest by one ratio, so it preserves hue by construction.
	 *
	 * Those are two different pictures, not two ways to compute one. With this ON the
	 * learned curve is applied per channel in AP1, wrapped in Unreal's blue correction
	 * exactly as CombineLUTs does, so the hue drift matches Unreal's. With it OFF the
	 * image keeps its hue exactly and looks cleaner but not Unreal-like.
	 *
	 * Reported as "the colours are still off" against a reference image: this is the
	 * switch that changes it.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Look", meta = (DisplayName = "Match Unreal Hue Behaviour"))
	bool bMatchUnrealHueBehavior = false;

	// -----------------------------------------------------------------------
	// Curve reshape (1.0.7)
	//
	// The network's curve is internally consistent but sits low with a flat shoulder.
	// Measured on the night-sky project (8 knots, panel peak 100 nit):
	//
	//     input        predicted    Unreal ACES
	//     18% grey     0.152        0.267        (-0.81 stops)
	//     100 nit      0.584        0.804
	//     peak         90 nit       100 nit      (10% of the panel unused)
	//     shoulder     0.03 slope    -
	//
	// Two monotone remaps in output space, applied at bake time in the LUT:
	//   MidGreyAnchorStops  lifts a bell centred on the curve's own 18% grey
	//   PeakFill            stretches the shoulder toward the real panel peak
	//
	// Defaults were chosen, not guessed: sweeping mid-lift x width x peak-fill and
	// picking the combination that puts 18% grey and 100 nit onto ACES while still
	// using 97.5% of the panel peak. Measured result of the defaults below:
	//
	//     input        reshaped    ACES
	//     18% grey     0.240       0.267
	//     100 nit      0.802       0.804
	//     400 nit      0.961       0.973
	//     peak         97.5 nit    -
	// -----------------------------------------------------------------------

	/** Mid-grey lift in stops. 0 disables the lift (pre-1.0.7 behaviour). */
	UPROPERTY(config, EditAnywhere, Category = "Look|Curve", meta = (ClampMin = "0.0", ClampMax = "2.0", DisplayName = "Mid-Grey Anchor (stops)"))
	float MidGreyAnchorStops = 0.45f;

	/** Width of the mid-grey lift in stops. Narrower keeps the effect off the shadows. */
	UPROPERTY(config, EditAnywhere, Category = "Look|Curve", meta = (ClampMin = "0.25", ClampMax = "6.0", DisplayName = "Mid-Grey Anchor Width"))
	float MidGreyAnchorWidth = 1.6f;

	/** How much of the remaining gap to the panel peak the shoulder closes (0..1). */
	UPROPERTY(config, EditAnywhere, Category = "Look|Curve", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Peak Fill"))
	float PeakFill = 0.70f;

	/**
	 * Chroma compensation: give colour back exactly where luminance was compressed.
	 *
	 * Losing lightness costs colourfulness (Hunt effect); recovering some chroma where
	 * the display still has room is what keeps a compressed image looking deep instead
	 * of washed out. Unlike a saturation slider this does nothing at the peak, so it
	 * cannot inflate highlights that are already clipped.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Look|Curve", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Chroma Compensation"))
	float ChromaCompensation = 0.0f;

	/**
	 * Multiplier on the learned local shadow lift.
	 *
	 * The lift was already signed correctly (regions below the local mean are raised)
	 * but measured only ~0.1 stops at the default s_low, which is invisible. 3x makes a
	 * region 2 stops below the mean open by ~0.6 stops without touching global exposure.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (ClampMin = "1.0", ClampMax = "8.0", DisplayName = "Shadow Boost"))
	float ShadowBoost = 3.0f;

	/** Resolution of the baked tone mapping LUT. 33 matches the paper/industry default. */
	UPROPERTY(config, EditAnywhere, Category = "Quality", meta = (ClampMin = "16", ClampMax = "64", DisplayName = "LUT Size"))
	int32 LUTSize = 33;

	// -----------------------------------------------------------------------
	// Local operator (guide gain + half-resolution Laplacian pyramid)
	//
	// This replaces what Unreal normally does inside PostProcessTonemap.usf with its
	// bilateral Local Exposure. It is deliberately NOT stacked on top of Unreal's
	// local exposure: two cascaded local operators overshoot and turn shadows grey.
	// Note that Unreal's GLOBAL automatic exposure is untouched and still applies
	// before this pass; the plugin only owns the tone curve and the local operator.
	// -----------------------------------------------------------------------

	/** Enable the local operator. Off means a pure global LUT (previous behaviour). */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (DisplayName = "Enable Local Operator"))
	bool bLocalOperator = true;

	/**
	 * Low-frequency local exposure strength. 0 disables the local-contrast term;
	 * 0.3-0.6 is a sensible range, above 1 the image starts to look flat and HDR-ish
	 * in the wrong way (every region normalised to the same brightness).
	 */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (ClampMin = "0.0", ClampMax = "2.0", DisplayName = "Local Exposure Strength"))
	float LocalExposureStrength = 0.60f;

	/**
	 * Mid-frequency detail (the L0 Laplacian band). This is the term that competes with
	 * a sharpening filter; keep it modest or it looks crunchy.
	 */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Detail Strength (mid)"))
	float DetailStrengthMid = 0.25f;

	/** Low-frequency detail (the L1 Laplacian band). */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (ClampMin = "0.0", ClampMax = "1.0", DisplayName = "Detail Strength (low)"))
	float DetailStrengthLow = 0.10f;

	/** Symmetric clamp on the gain, in log2 stops. 1.5 keeps the operator well conditioned. */
	UPROPERTY(config, EditAnywhere, Category = "Local Operator", meta = (ClampMin = "0.1", ClampMax = "4.0", DisplayName = "Gain Clamp (stops)"))
	float LocalGainClampStops = 1.5f;

	/** Enable the on-screen debug view (curve plot in the corner). */
	UPROPERTY(config, EditAnywhere, Category = "Debug", meta = (DisplayName = "Show Debug Overlay"))
	bool bShowDebugOverlay = false;

	//~ Begin UDeveloperSettings
	virtual FName GetCategoryName() const override { return TEXT("Plugins"); }
	//~ End UDeveloperSettings

	/** Helper used by the editor menu: flip the flag and persist it. */
	static void SetEnabled(bool bInEnabled);

	/** Helper: current flag, safe to call from any thread. */
	static bool IsEnabled();
};
