// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/ThreadSafeBool.h"
#include "RenderGraphResources.h"
#include "NTMOPolicy.h"
#include "NTMOTypes.h"

class FRDGBuilder;
class FSceneView;
struct FPostProcessMaterialInputs;
struct FScreenPassRenderTarget;
struct FScreenPassTexture;

/**
 * Owns the render-thread side of the plugin: the per-frame tone mapping LUT bake
 * and the final full screen pass that replaces Unreal's tonemapper.
 *
 * Everything here is transient RDG memory. There is no readback, no fence and no
 * CPU/GPU sync of any kind. The per-frame GPU work is:
 *
 *   1. LutBake  - 33^3 texels, evaluates the full hue-preserving tone map so the
 *                 per-pixel cost collapses to one 3D texture fetch.
 *   2. Tonemap  - full screen, one LUT fetch + bloom add + output encoding.
 *
 * Both are resolution independent except the final full screen pass, which the
 * engine was already paying for.
 */
class FNTMORenderer
{
public:
	FNTMORenderer();
	~FNTMORenderer();

	/** Game thread. Pulls the project settings into the providers. Throttled. */
	void SyncSettingsFromDefaults(bool bForce = false);

	/** Render thread safe. */
	bool IsActiveThisFrame() const { return bCachedEnabled; }

	/** Human readable status, for logs and the debug overlay. */
	FString GetStatusString() const;

	/**
	 * Replaces the tonemapper. Render thread.
	 * Returns the new scene colour, or Output unchanged when the plugin decides to
	 * fall back to Unreal's tonemapper.
	 */
	FScreenPassTexture AddPasses(
		FRDGBuilder& GraphBuilder,
		const FSceneView& View,
		const FPostProcessMaterialInputs& Inputs,
		FScreenPassRenderTarget& Output);

private:
	FNTMOContextProvider ContextProvider;

	/** Always available. This is what runs today. */
	TUniquePtr<FNTMOCurvePolicyAnalytic> AnalyticPolicy;

	/** Used when Curve Source is Neural and a model asset resolves. */
	TUniquePtr<class FNTMOCurvePolicyNeural> NeuralPolicy;

	/**
	 * Picks the provider for this frame: neural when it can run, analytic otherwise.
	 * Returns null only if every provider declined.
	 */
	INTMOCurveProvider* SelectProvider(const FNTMOContext& Context) const;

	/** Written on the game thread, read on the render thread. */
	FThreadSafeBool bCachedEnabled;

	/**
	 * Previous frame's (already smoothed) knots, kept alive across frames so the curve
	 * can be temporally filtered. Registered into each frame's render graph as an
	 * external buffer; the graph writes it back through a copy pass.
	 */
	TRefCountPtr<FRDGPooledBuffer> KnotsHistory;

	/** When the history was last written. A long gap (view not rendered) resets it. */
	double KnotsHistoryTime = -1.0;

	/**
	 * One-shot GPU readback of the network's 8 knots, used by r.NTMO.DumpKnots.
	 * This is the only way to see what the AI actually produced: the CPU FNTMOCurveParams
	 * still holds the analytic fallback, so logging that would report the wrong curve.
	 */
	TUniquePtr<class FRHIGPUBufferReadback> KnotsReadback;

	mutable double LastSettingsSyncTime = 0.0;
	bool bLoggedFallback = false;
};
