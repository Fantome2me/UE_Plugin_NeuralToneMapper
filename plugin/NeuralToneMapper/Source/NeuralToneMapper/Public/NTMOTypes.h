// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * Per-frame hardware + engine context consumed by the tone mapping policy.
 *
 * This mirrors the "display capability" idea used by Dolby Vision (target display
 * descriptors) but is populated from DXGI / EDID data that Unreal already exposes
 * through FHDRMetaData (see HDRHelper.h).
 *
 * Values are gathered on the game thread (cheap, only when the display config
 * changes) and consumed on the render thread.
 */
struct FNTMOContext
{
	// ---------------------------------------------------------------- Display

	/** Black level of the panel, in nits. */
	float MinLuminanceNits = 0.001f;

	/** Small-area peak luminance (specular highlights), in nits. */
	float MaxLuminanceNits = 1000.0f;

	/** Full-frame sustained luminance (ABL limited). <= MaxLuminanceNits. */
	float MaxFullFrameLuminanceNits = 600.0f;

	/** Diffuse / graphics white reference, in nits. Typically 80..300. */
	float PaperWhiteNits = 203.0f;

	/** 0 = SDR_sRGB, 1 = SDR_Rec709, 2 = ExplicitGamma, 3 = HDR PQ, 4 = HDR scRGB(linear). */
	int32 OutputDevice = 0;

	/** 0 = sRGB_D65, 1 = DCIP3_D65, 2 = Rec2020_D65, 3 = ACES_D60, 4 = ACEScg_D60. */
	int32 OutputGamut = 0;

	/** True when the engine is currently presenting HDR. */
	int32 bHDREnabled = 0;

	/** Ratio MaxFullFrame/Max, precomputed. Small => strong ABL. */
	float ABLRatio = 0.6f;

	// ------------------------------------------------------------------ Engine

	/** Lens attenuation (q) from r.EyeAdaptation.LensAttenuation. Default 0.78. */
	float LensAttenuation = 0.78f;

	/** User/asset exposure bias in stops (already includes post process AutoExposureBias). */
	float ExposureBias = 0.0f;

	// ------------------------------------------------------------------- User

	/** Extra brightness preference in stops, exposed via plugin settings. */
	float UserBrightnessBias = 0.0f;

	/** Master intensity of the plugin, 0 = effectively ACES-like, 1 = full. */
	float Intensity = 1.0f;

	/** True when the plugin is allowed to take over tone mapping this frame. */
	int32 bEnabled = 0;

	void UpdateDerived()
	{
		MaxFullFrameLuminanceNits = FMath::Clamp(MaxFullFrameLuminanceNits, 1.0f, MaxLuminanceNits);
		ABLRatio = FMath::Clamp(MaxFullFrameLuminanceNits / FMath::Max(MaxLuminanceNits, 1.0f), 0.0f, 1.0f);
		PaperWhiteNits = FMath::Clamp(PaperWhiteNits, 20.0f, 1000.0f);
		MinLuminanceNits = FMath::Clamp(MinLuminanceNits, 0.0f, 10.0f);
	}
};

/**
 * Parameters of the tone curve, produced every frame by the policy stage.
 *
 * 8 knots in log2 space drive a monotone cubic. The curve maps
 *   in_log2  = log2(LinearColor)     (linear domain, 1.0 == 100 nits, UE convention)
 *   out_log2 = EvaluateCurve(in_log2)
 * and out is also in the same linear domain (so out * 100 == nits).
 *
 * This is deliberately the same "piecewise / paramatric curve" representation that
 * Dolby Vision uses for its reshaping curve, so an NNE model can regress it.
 */
struct FNTMOCurveParams
{
	/** Monotone knot values, log2 domain. Index i corresponds to x = KnotX0 + i*KnotStep. */
	float Knots[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

	/** Saturation multiplier applied after the hue-preserving MaxRGB scale. */
	float Saturation = 1.0f;

	/** Gamut compression strength in [0,1]. */
	float GamutCompress = 0.0f;

	/** Highlight desaturation amount in [0,1]. */
	float HighlightDesat = 0.0f;

	/** Black lift in linear domain, in [0, 0.02]. */
	float BlackLift = 0.0f;

	static constexpr float KnotX0 = -10.0f;
	static constexpr float KnotX1 = 4.0f;
};

/** Small GPU-side stats block produced by the analysis + policy passes. */
struct FNTMOStats
{
	/** Average scene luminance (pre-exposure removed), in linear units. */
	float AverageLuminance = 0.18f;

	/** 99.99th percentile luminance - the "MaxCLL" style highlight estimate. */
	float HighlightLuminance = 1.0f;

	/** Resolved exposure scale applied to scene colour before tone curve. */
	float Exposure = 1.0f;
};
