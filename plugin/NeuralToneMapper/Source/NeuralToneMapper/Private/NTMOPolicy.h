// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NTMOTypes.h"

/** Where the per-frame tone curve comes from. */
enum class ENTMOCurveProviderKind : uint8
{
	/** Built-in GPU analytic policy (always available). */
	Analytic,

	/** Neural curve predictor running through NNE/RDG. */
	Neural,
};

/**
 * Interface for the "brain" of the tone mapper.
 *
 * The analytic implementation is a closed form, hardware-aware policy. The neural
 * implementation runs a small network through NNE and regresses the *same*
 * FNTMOCurveParams struct. Keeping both behind one interface means the renderer
 * does not care which one produced the curve - swapping them is a single line.
 */
class INTMOCurveProvider
{
public:
	virtual ~INTMOCurveProvider() = default;

	virtual ENTMOCurveProviderKind GetKind() const = 0;

	/**
	 * Produce this frame's curve. Returns false when the provider cannot run, and
	 * the caller falls back to the analytic policy.
	 */
	virtual bool ComputeCurve(const FNTMOContext& Context, FNTMOCurveParams& OutParams) = 0;

	/** Human readable status, used by the debug overlay and the log. */
	virtual FString GetStatusString() const = 0;
};

/**
 * Closed form policy. This is what runs today.
 *
 * It is genuinely adaptive rather than a fixed curve: it consumes the panel's
 * black level, small-area peak, full-frame (ABL) peak, HDR/SDR mode and the
 * user's brightness preference, and moves the shoulder accordingly.
 */
class FNTMOCurvePolicyAnalytic : public INTMOCurveProvider
{
public:
	//~ Begin INTMOCurveProvider
	virtual ENTMOCurveProviderKind GetKind() const override { return ENTMOCurveProviderKind::Analytic; }
	virtual bool ComputeCurve(const FNTMOContext& Context, FNTMOCurveParams& OutParams) override;
	virtual FString GetStatusString() const override;
	//~ End INTMOCurveProvider

	/** 0..1, how strongly the ABL ratio moves the shoulder. */
	float ABLAwareness = 1.0f;

	/** Look parameters, mirrored from the project settings. */
	float Saturation = 1.0f;
	float GamutCompress = 0.0f;
	float HighlightDesat = 0.35f;
	float BlackLift = 0.0f;
};

/**
 * Gathers display capability + engine exposure context.
 *
 * Display data comes from FHDRMetaData, which Unreal fills from
 * IDXGIOutput6::GetDesc1 (i.e. EDID / SMPTE ST.2086). When no OS window is
 * available - commandlets, -nullrhi, capture - it degrades to the
 * r.HDR.Display.* cvars that the engine's own tonemapper consumes, so the plugin
 * still behaves sensibly.
 */
class FNTMOContextProvider
{
public:
	/** Game thread: re-query the display. Throttled unless bForceRefresh. */
	void RefreshFromGameThread(bool bForceRefresh = false);

	/** Game thread: push the user-facing settings into the shared snapshot. */
	void UpdateUserOverrides(float InIntensity, float InBrightnessBias);

	/** Thread safe snapshot for the render thread. */
	FNTMOContext GetContext() const;

	/** True if the last query produced real display metadata (debug). */
	bool HadDisplayMetadata() const { return bHadDisplayMetadata; }

private:
	mutable FCriticalSection Sync;
	FNTMOContext Cached;
	bool bHadDisplayMetadata = false;
	double LastRefreshTime = 0.0;

	// Kept separately so the throttled display refresh cannot clobber them.
	float PendingIntensity = 1.0f;
	float PendingBrightnessBias = 0.0f;
};
