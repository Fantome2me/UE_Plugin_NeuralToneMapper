// Copyright (c) 2026. All Rights Reserved.

#include "NTMOSceneViewExtension.h"

#include "NeuralToneMapperModule.h"
#include "NTMORenderer.h"
#include "NTMOSettings.h"

#include "PostProcess/PostProcessMaterialInputs.h"
#include "RenderGraphBuilder.h"
#include "ScreenPass.h"

FNTMOSceneViewExtension::FNTMOSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
	Renderer = MakeShared<FNTMORenderer, ESPMode::ThreadSafe>();
}

FNTMOSceneViewExtension::~FNTMOSceneViewExtension() = default;

int32 FNTMOSceneViewExtension::GetPriority() const
{
	// Run after most other extensions so we own the final image.
	return 100;
}

bool FNTMOSceneViewExtension::IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const
{
	// Scene captures, thumbnails and commandlets have no viewport. Leaving them
	// alone keeps editor asset previews and reflection captures untouched.
	if (Context.Viewport == nullptr)
	{
		return false;
	}

	// Read the switch DIRECTLY rather than returning a cached flag.
	//
	// This used to return bIsActive, which is only written in SetupViewFamily(). The
	// engine filters extensions by IsActiveThisFrame() *before* dispatching the setup
	// hooks, so the flag was never written and the extension was永远 inactive: the
	// post process pass was never subscribed and AddPasses never ran. That is why the
	// image looked identical to Unreal's tonemapper -- nothing was ever replaced.
	// FNeuralToneMapperModule::IsEnabled() is null-safe and consults r.NTMO.Enable
	// before the project settings, so the render thread sees changes immediately.
	return FNeuralToneMapperModule::IsEnabled();
}

void FNTMOSceneViewExtension::SetupViewFamily(FSceneViewFamily& InViewFamily)
{
	if (Renderer.IsValid())
	{
		Renderer->SyncSettingsFromDefaults();
		bIsActive = Renderer->IsActiveThisFrame();
	}
}

void FNTMOSceneViewExtension::SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView)
{
	if (Renderer.IsValid())
	{
		Renderer->SyncSettingsFromDefaults();
		bIsActive = Renderer->IsActiveThisFrame();
	}
}

void FNTMOSceneViewExtension::SubscribeToPostProcessingPass(
	EPostProcessingPass Pass,
	const FSceneView& InView,
	FPostProcessingPassDelegateArray& InOutPassCallbacks,
	bool bIsPassEnabled)
{
	// Only take over when the tonemapper would otherwise run. Registering anything
	// here makes PostProcessing.cpp skip AddCombineLUTPass *and* AddTonemapPass.
	if (Pass != EPostProcessingPass::ReplacingTonemapper)
	{
		return;
	}

	// Gate on the live switch too, not the cached flag, for the same reason as
	// IsActiveThisFrame_Internal above.
	if (!bIsPassEnabled || !Renderer.IsValid() || !FNeuralToneMapperModule::IsEnabled())
	{
		// Not active: contributing nothing leaves Unreal's tonemapper in charge.
		return;
	}

	InOutPassCallbacks.Add(FAfterPassCallbackDelegate::CreateRaw(
		this, &FNTMOSceneViewExtension::PostProcessPassReplacingTonemapper_RenderThread));
}

FScreenPassTexture FNTMOSceneViewExtension::PostProcessPassReplacingTonemapper_RenderThread(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	const FPostProcessMaterialInputs& Inputs)
{
	check(IsInRenderingThread());

	const FScreenPassTexture SceneColor =
		FScreenPassTexture::CopyFromSlice(GraphBuilder, Inputs.GetInput(EPostProcessMaterialInput::SceneColor));

	// If the override output is provided, this is the last pass in post processing.
	FScreenPassRenderTarget Output = Inputs.OverrideOutput;

	// CRITICAL: never write into the texture we are sampling.
	//
	// In the ReplacingTonemapper path the engine is free to hand us an OverrideOutput
	// that IS the input scene colour (there is nothing downstream, so why allocate a
	// copy). Sampling a resource as an SRV in the same pass that binds it as the render
	// target is invalid on D3D12; it surfaced as
	//   "Failed to create graphics pipeline ... Vertex: <unknown> Pixel: <unknown>"
	// followed by UE's fatal "Shader compilation failures are Fatal", which is a
	// misleading message -- the shader had compiled fine.
	if (!Output.IsValid() || Output.Texture == SceneColor.Texture)
	{
		Output = FScreenPassRenderTarget::CreateFromInput(
			GraphBuilder, SceneColor, View.GetOverwriteLoadAction(), TEXT("NTMO.Output"));
	}

	if (!Renderer.IsValid())
	{
		return Output;
	}

	return Renderer->AddPasses(GraphBuilder, View, Inputs, Output);
}
