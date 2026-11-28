// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/ThreadSafeBool.h"
#include "SceneViewExtension.h"

class FNTMORenderer;
struct FPostProcessMaterialInputs;

/**
 * Hooks into the post processing chain at ReplacingTonemapper.
 *
 * This is the only engine-level seam the plugin needs. When the extension
 * subscribes, PostProcessing.cpp skips both AddCombineLUTPass and AddTonemapPass,
 * so Unreal's tone mapping is completely off for that view - no engine source
 * modification required.
 */
class FNTMOSceneViewExtension : public FSceneViewExtensionBase
{
public:
	FNTMOSceneViewExtension(const FAutoRegister& AutoRegister);
	virtual ~FNTMOSceneViewExtension() override;

	//~ Begin ISceneViewExtension
	virtual int32 GetPriority() const override;
	virtual bool IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const override;
	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override;
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override;
	virtual void SubscribeToPostProcessingPass(
		EPostProcessingPass Pass,
		const FSceneView& InView,
		FPostProcessingPassDelegateArray& InOutPassCallbacks,
		bool bIsPassEnabled) override;
	//~ End ISceneViewExtension

private:
	/** Executes inside the render graph, replacing the tonemapper. */
	FScreenPassTexture PostProcessPassReplacingTonemapper_RenderThread(
		FRDGBuilder& GraphBuilder,
		const FSceneView& View,
		const FPostProcessMaterialInputs& Inputs);

	TSharedPtr<FNTMORenderer, ESPMode::ThreadSafe> Renderer;

	/** Cached so the render thread does not have to re-read the context. */
	FThreadSafeBool bIsActive = false;
};
