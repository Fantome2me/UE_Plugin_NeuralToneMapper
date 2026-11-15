// Copyright (c) 2026. All Rights Reserved.

#include "NTMORenderer.h"

// Declared in Runtime/Renderer/Private/SceneTextureParameters.h, which a plugin cannot
// include. The symbol is exported (RENDERER_API) so an exact re-declaration links. This
// is the same buffer Unreal's tonemapper reads for GlobalExposure. A signature change in
// a future engine version fails at link time rather than silently mis-exposing.
extern RENDERER_API FRDGBufferRef GetEyeAdaptationBuffer(FRDGBuilder& GraphBuilder, const FSceneView& View);

#include "NTMOSettings.h"
#include "NTMOCurvePolicyNeural.h"

#include "GlobalShader.h"
#include "HAL/IConsoleManager.h"
#include "PixelShaderUtils.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "SceneView.h"
#include "ScreenPass.h"
#include "ShaderParameterStruct.h"



// ---------------------------------------------------------------------------
// CVars
// ---------------------------------------------------------------------------
// Dumps the 8 knots the network produced (one GPU readback, then stops). This is the
// only way to verify what the AI actually emits: the CPU curve struct still holds the
// analytic fallback, so logging it would report the wrong numbers.
// Gamut compression strength in percent (0 = off). Kept as a cvar because it is a
// grading decision that varies per project, not something to hardcode.
// Runtime overrides for cost attribution and tuning. -1 = follow project settings.
static TAutoConsoleVariable<int32> CVarNTMOInterval(
	TEXT("r.NTMO.Interval"),
	-1,
	TEXT("Run the curve network every N frames (-1 = follow settings). Cost scales 1/N."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMOLocalOverride(
	TEXT("r.NTMO.Local"),
	-1,
	TEXT("Local operator: -1 = follow settings, 0 = off, 1 = on."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMONeuralOverride(
	TEXT("r.NTMO.Neural"),
	-1,
	TEXT("Neural curve: -1 = follow settings, 0 = analytic fallback, 1 = neural."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMOGamutCompressPercent(
	TEXT("r.NTMO.GamutCompress"),
	0,
	TEXT("Gamut compression strength in percent (0..100). 0 = off."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMOLogLiveness(
	TEXT("r.NTMO.LogLiveness"),
	0,
	TEXT("If 1, log AddPasses call counts so you can prove the tonemapper runs every frame."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMODumpKnots(
	TEXT("r.NTMO.DumpKnots"),
	0,
	TEXT("If 1, read back and log the neural curve knots once, then revert to 0."),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMOLUTSize(
	TEXT("r.NTMO.LUTSize"),
	0,
	TEXT("Override the baked LUT size. 0 follows project settings (default 33).\n"),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarNTMOBloomScale(
	TEXT("r.NTMO.BloomScale"),
	1.0f,
	TEXT("Scale applied to the combined bloom before adding it to scene colour.\n")
	TEXT("Unreal multiplies bloom by 1/(SceneColorTint * LocalExposure) which is\n")
	TEXT("approximately 1 for most projects.\n"),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarNTMOGamma(
	TEXT("r.NTMO.Gamma"),
	2.2f,
	TEXT("Inverse gamma used for SDR output encoding.\n"),
	ECVF_RenderThreadSafe);

extern int32 NTMO_GetRequestedEnable();

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------
class FNTMOLutBakeCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOLutBakeCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOLutBakeCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FVector4f, CurveKnots01)
		SHADER_PARAMETER(FVector4f, CurveKnots23)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, NTMOKnotsBuffer)
		SHADER_PARAMETER(float, NTMOExposure)
		SHADER_PARAMETER(int32, NTMOCurveFromBuffer)
		SHADER_PARAMETER(float, NTMOSaturation)
		SHADER_PARAMETER(float, NTMOGamutCompress)
		SHADER_PARAMETER(float, NTMOHighlightDesat)
		SHADER_PARAMETER(float, NTMOBlackLift)
		SHADER_PARAMETER(float, NTMOMinNits)
		SHADER_PARAMETER(float, NTMOMaxNits)
		SHADER_PARAMETER(float, NTMOMaxFullFrameNits)
		SHADER_PARAMETER(float, NTMOPaperWhiteNits)
		SHADER_PARAMETER(int32, NTMOLutSize)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture3D<float4>, RWOutputLUT)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOLutBakeCS, "/NTMO/Private/NTMOLutBake.usf", "MainCS", SF_Compute);

class FNTMOTonemapPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOTonemapPS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOTonemapPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>, NTMOEyeAdaptationBuffer)
		SHADER_PARAMETER(FVector3f, NTMOSceneColorTint)
		SHADER_PARAMETER(int32, NTMOApplyExposure)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, NTMOSceneColor)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOSceneColorSampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, NTMOBloom)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOBloomSampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture3D, NTMOLut)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLutSampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float2>, NTMOLocalCorrection)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLocalCorrectionSampler)
		SHADER_PARAMETER(int32, NTMOLocalEnabled)
		SHADER_PARAMETER(float, NTMOLutSize)
		SHADER_PARAMETER(float, NTMOLutScale)
		SHADER_PARAMETER(float, NTMOLutOffset)
		SHADER_PARAMETER(float, NTMOBloomScale)
		SHADER_PARAMETER(int32, NTMOOutputDevice)
		SHADER_PARAMETER(float, NTMOGamma)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		// Compile for every platform. The previous IsFeatureLevelSupported(SM5) guard
		// compiled the shader yet left the RHI without its bytecode at PSO creation
		// ("Pixel: <unknown>"), which UE reports as a fatal shader compilation failure.
		return true;
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOTonemapPS, "/NTMO/Private/NTMOTonemap.usf", "MainPS", SF_Pixel);

// ---------------------------------------------------------------------------
// Local operator shaders (guide gain + half-resolution Laplacian pyramid)
//
// Bandwidth is the whole argument here. At 4K a single full-resolution RGBA16F
// read+write pass costs ~0.38 ms, so the pyramid's finest level runs at half
// resolution and the result is a small gain map consumed by the tonemap pass that
// already existed -- the local stage therefore adds NO full-resolution pass.
// ---------------------------------------------------------------------------
class FNTMOLumFromSceneCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOLumFromSceneCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOLumFromSceneCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, NTMOSceneColorLocal)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOSceneColorLocalSampler)
		SHADER_PARAMETER(FVector2f, NTMOSceneColorTexelSize)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWNTMOG0)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOLumFromSceneCS, "/NTMO/Private/NTMOLocal.usf", "MainCS_LumFromScene", SF_Compute);

class FNTMODownsampleCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMODownsampleCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMODownsampleCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOLevelIn)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLevelInSampler)
		SHADER_PARAMETER(FVector2f, NTMOLevelInTexelSize)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWNTMOLevelOut)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMODownsampleCS, "/NTMO/Private/NTMOLocal.usf", "MainCS_Downsample", SF_Compute);

// Learned local Laplacian correction (NTM 1.0.2).
//
// Outputs a half-res log2 display correction that the tonemap shader multiplies by
// exp2(). Its per-tile band strengths are the 48 values the network predicts, so the
// local operator is now content-adaptive instead of using three fixed constants.
class FNTMOCorrectionCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOCorrectionCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOCorrectionCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOLum0)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLum0Sampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOLum1)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLum1Sampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOLum2)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLum2Sampler)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOLum3)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOLum3Sampler)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, NTMONeuralOut)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint32>, NTMOStatsBuf)
		SHADER_PARAMETER(float, NTMOStrengthScale)
		SHADER_PARAMETER(float, NTMODetailMix)
		SHADER_PARAMETER(float, NTMOExposureScale)
		SHADER_PARAMETER(float, NTMOClampStops)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float2>, RWNTMOCorrection)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOCorrectionCS, "/NTMO/Private/NTMOLocal.usf", "MainCS_Correction", SF_Compute);

// Curve temporal smoothing. Blends this frame's predicted knots toward the previous
// frame's, so a hard content change does not flash the whole image.
class FNTMOBlendKnotsCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOBlendKnotsCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOBlendKnotsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, NTMOPrevKnots)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, NTMONewKnots)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, RWNTMOSmoothedKnots)
		SHADER_PARAMETER(float, NTMOBlendAlpha)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOBlendKnotsCS, "/NTMO/Private/NTMOLocal.usf", "MainCS_BlendKnots", SF_Compute);

// ---------------------------------------------------------------------------
// Neural input assembly (see NTMOFeatures.usf)
//
// The trained network consumes a 1x48x72x128 tensor: PQ scene colour plus 45
// per-frame context values (display scalars, 4x4 tile means, 16-bin histogram).
// Two dispatches build it on the GPU so nothing ever round-trips to the CPU.
// ---------------------------------------------------------------------------
class FNTMOSceneStatsCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOSceneStatsCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOSceneStatsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOStatsLum)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOStatsLumSampler)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint32>, RWNTMOStats)
		SHADER_PARAMETER(uint32, NTMOStatsWidth)
		SHADER_PARAMETER(uint32, NTMOStatsHeight)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOSceneStatsCS, "/NTMO/Private/NTMOFeatures.usf", "MainCS_SceneStats", SF_Compute);

class FNTMOAssembleCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNTMOAssembleCS);
	SHADER_USE_PARAMETER_STRUCT(FNTMOAssembleCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, NTMOSceneColorFeat)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOSceneColorFeatSampler)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint32>, NTMOStats)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, NTMOCtxScalars)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, RWNTMOInputTensor)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, NTMOGuideTex)
		SHADER_PARAMETER_SAMPLER(SamplerState, NTMOGuideTexSampler)
		SHADER_PARAMETER(FVector2f, NTMOGuideTexelSize)
		SHADER_PARAMETER(uint32, NTMOFeatW)
		SHADER_PARAMETER(uint32, NTMOFeatH)
		SHADER_PARAMETER(uint32, NTMOFeatPlane)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

IMPLEMENT_GLOBAL_SHADER(FNTMOAssembleCS, "/NTMO/Private/NTMOFeatures.usf", "MainCS_Assemble", SF_Compute);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace
{
	/** Map EDisplayOutputFormat onto the ids used by NTMOCommon.ush. */
	int32 OutputDeviceToShaderValue(int32 RawOutputDevice)
	{
		switch (RawOutputDevice)
		{
		// Must mirror EDisplayOutputFormat exactly, so the encode in the LUT bake is the
		// same transfer function Unreal's tonemapper would have used.
		case 0:	 return 0;	// SDR_sRGB               -> NTMO_OUTPUT_SRGB
		case 1:	 return 1;	// SDR_Rec709             -> NTMO_OUTPUT_REC709
		case 2:	 return 2;	// SDR_ExplicitGamma      -> NTMO_OUTPUT_EXPLICIT_GAMMA
		case 3:	// HDR_ACES1000nitST2084
		case 4:	// HDR_ACES2000nitST2084
			return 3;	// NTMO_OUTPUT_PQ
		case 5:	// HDR_ACES1000nitScRGB
		case 6:	// HDR_ACES2000nitScRGB
			return 4;	// NTMO_OUTPUT_SCRGB
		default:
			return 0;
		}
	}

	FIntVector LutSize3D(int32 LutSize)
	{
		return FIntVector(LutSize, LutSize, LutSize);
	}
}

// ---------------------------------------------------------------------------
// FNTMORenderer
// ---------------------------------------------------------------------------
FNTMORenderer::FNTMORenderer()
{
	// NEVER touch UObjects here. This constructor runs from the scene view extension's
	// constructor, which runs from the module's StartupModule -- i.e. before the
	// UDeveloperSettings CDO is guaranteed to exist. Calling GetDefault<UNTMOSettings>()
	// from that phase crashed the editor with an access violation inside CoreUObject
	// (observed on 2026-11-13, callstack FNTMORenderer::FNTMORenderer -> StartupModule).
	//
	// SyncSettingsFromDefaults() is called from SetupViewFamily/SetupView on the game
	// thread before the first render, which is the safe phase. Until that happens
	// bCachedEnabled is false, so Unreal's own tonemapper stays in charge -- a safe
	// default rather than a half-initialised one.
	AnalyticPolicy = MakeUnique<FNTMOCurvePolicyAnalytic>();
	NeuralPolicy = MakeUnique<FNTMOCurvePolicyNeural>();
	bCachedEnabled = false;
	bLoggedFallback = false;
	LastSettingsSyncTime = 0.0;
}

FNTMORenderer::~FNTMORenderer() = default;

INTMOCurveProvider* FNTMORenderer::SelectProvider(const FNTMOContext& Context) const
{
	// Neural first when it can actually run; otherwise the analytic policy.
	if (NeuralPolicy.IsValid() && NeuralPolicy->IsReady())
	{
		return NeuralPolicy.Get();
	}

	return AnalyticPolicy.Get();
}

void FNTMORenderer::SyncSettingsFromDefaults(bool bForce)
{
	const double Now = FPlatformTime::Seconds();
	if (!bForce && (Now - LastSettingsSyncTime) < 0.25)
	{
		return;
	}
	LastSettingsSyncTime = Now;

	const UNTMOSettings* Settings = GetDefault<UNTMOSettings>();
	if (Settings == nullptr)
	{
		// Class default object not created yet (early loading phase). Stay inert so
		// Unreal's own tonemapper keeps rendering until settings are readable.
		bCachedEnabled = false;
		return;
	}

	if (AnalyticPolicy.IsValid())
	{
		AnalyticPolicy->ABLAwareness = Settings->ABLAwareness;
		AnalyticPolicy->Saturation = Settings->Saturation;
		AnalyticPolicy->GamutCompress = 0.0f;
		AnalyticPolicy->HighlightDesat = Settings->HighlightDesaturation;
		AnalyticPolicy->BlackLift = Settings->BlackLift;
	}

	if (NeuralPolicy.IsValid())
	{
		NeuralPolicy->SetRuntimeName(Settings->NNERuntimeName);
		NeuralPolicy->SetModelPath(Settings->CurveModelAsset);
		NeuralPolicy->SetModelFilePath(Settings->CurveModelFile);

		// We are on the game thread here, so this is the right place to load the .onnx
		// and build the NNE model. Doing it lazily from the render path would be too
		// late: SelectProvider() asks IsReady(), which stays false until this runs.
		if (Settings->CurveSource == ENTMOCurveSource::Neural)
		{
			NeuralPolicy->EnsureProbed();
		}
	}

	ContextProvider.UpdateUserOverrides(Settings->Intensity, Settings->UserBrightnessBias);
	ContextProvider.RefreshFromGameThread(bForce);

	const int32 Requested = NTMO_GetRequestedEnable();
	bCachedEnabled = (Requested == -1) ? Settings->bEnabled : (Requested != 0);
}

FString FNTMORenderer::GetStatusString() const
{
	const FNTMOContext Ctx = ContextProvider.GetContext();
	return FString::Printf(
		TEXT("NTMO[%s] %s | peak %.0f nit | fullframe %.0f nit | black %.4f nit | paper white %.0f nit | HDR %d | outdev %d | intensity %.2f"),
		bCachedEnabled ? TEXT("ON") : TEXT("OFF"),
		AnalyticPolicy.IsValid() ? *AnalyticPolicy->GetStatusString() : TEXT("none"),
		Ctx.MaxLuminanceNits,
		Ctx.MaxFullFrameLuminanceNits,
		Ctx.MinLuminanceNits,
		Ctx.PaperWhiteNits,
		Ctx.bHDREnabled,
		Ctx.OutputDevice,
		Ctx.Intensity);
}

FScreenPassTexture FNTMORenderer::AddPasses(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	const FPostProcessMaterialInputs& Inputs,
	FScreenPassRenderTarget& Output)
{
	check(IsInRenderingThread());

	// Liveness counter, silent unless r.NTMO.LogLiveness is on.
	//
	// "Curve in use" is one-shot, so without a counter there is no way to tell "the
	// pass runs every frame" from "the pass ran once at startup and never again" --
	// exactly the ambiguity that once made an A/B read as identical images.
	if (CVarNTMOLogLiveness.GetValueOnRenderThread() != 0)
	{
		static int32 NTMOCallCounter = 0;
		++NTMOCallCounter;
		if (NTMOCallCounter == 1 || (NTMOCallCounter % 300) == 0)
		{
			UE_LOG(LogTemp, Log, TEXT("[NTMO] AddPasses call #%d"), NTMOCallCounter);
		}
	}

	const FScreenPassTexture SceneColor =
		FScreenPassTexture::CopyFromSlice(GraphBuilder, Inputs.GetInput(EPostProcessMaterialInput::SceneColor));
	if (!SceneColor.IsValid())
	{
		return Output;
	}

	// Bloom is optional; anything can be missing if a view disables bloom.
	FScreenPassTexture Bloom = FScreenPassTexture::CopyFromSlice(
		GraphBuilder, Inputs.GetInput(EPostProcessMaterialInput::CombinedBloom));

	// ---- Context + curve -------------------------------------------------
	const FNTMOContext Ctx = ContextProvider.GetContext();

	INTMOCurveProvider* Provider = SelectProvider(Ctx);

	FNTMOCurveParams Curve;
	if (Provider == nullptr || !Provider->ComputeCurve(Ctx, Curve))
	{
		// Neural declined: fall back to analytic rather than surrendering the effect.
		if (Provider != nullptr && Provider->GetKind() == ENTMOCurveProviderKind::Neural
			&& AnalyticPolicy.IsValid() && AnalyticPolicy->ComputeCurve(Ctx, Curve))
		{
			// fall through with the analytic curve
		}
		else
		{
			if (!bLoggedFallback)
			{
				bLoggedFallback = true;
				UE_LOG(LogTemp, Warning,
					TEXT("[NTMO] No curve provider available, falling back to Unreal's tonemapper."));
			}
			return Output;
		}
	}

	// ---- Look parameters -------------------------------------------------
	// These are the artist-facing controls (saturation, highlight desaturation, black
	// lift, gamut compression). They must be applied HERE, not left to whichever
	// provider filled FNTMOCurveParams.
	//
	// The neural provider returns false from ComputeCurve() by design (its knots live in
	// a GPU buffer), so `Curve` used to keep the ANALYTIC fallback's look values. The
	// consequence was that Saturation and Highlight Desaturation in the project settings
	// did nothing at all when the AI was active -- measured: changing them moved the sky
	// saturation by 0.0000. Applying them after provider selection makes the controls
	// work on both paths.
	if (const UNTMOSettings* LookSettings = GetDefault<UNTMOSettings>())
	{
		const float LookIntensity = FMath::Clamp(Ctx.Intensity, 0.0f, 1.0f);
		const int32 OutputDevForLook = OutputDeviceToShaderValue(Ctx.OutputDevice);
		const bool bHdrOutput = (OutputDevForLook == 3 || OutputDevForLook == 4);

		Curve.Saturation = FMath::Lerp(1.0f, FMath::Max(LookSettings->Saturation, 0.0f), LookIntensity);

		// Highlight desaturation is a display-referred look: on an HDR panel the extra
		// range is real, so the same amount would wash out legitimate colour.
		const float DesatScale = bHdrOutput ? 0.6f : 1.0f;
		Curve.HighlightDesat = FMath::Clamp(LookSettings->HighlightDesaturation, 0.0f, 1.0f)
			* LookIntensity * DesatScale;

		Curve.BlackLift = FMath::Max(LookSettings->BlackLift, 0.0f) * LookIntensity;
		Curve.GamutCompress = FMath::Clamp(
			CVarNTMOGamutCompressPercent.GetValueOnRenderThread() > 0
				? (float)CVarNTMOGamutCompressPercent.GetValueOnRenderThread() / 100.0f
				: 0.0f, 0.0f, 1.0f);
	}


	// ---- LUT --------------------------------------------------------------
	// One-shot dump of the curve actually in use. Without this there is no way to tell
	// "the plugin is doing nothing" from "the plugin's curve happens to agree with
	// Unreal's" -- they look identical on screen but are very different claims.
	{
		static bool bDumpedCurve = false;
		if (!bDumpedCurve)
		{
			bDumpedCurve = true;
			const bool bNeuralCurve = (Provider != nullptr)
				&& (Provider->GetKind() == ENTMOCurveProviderKind::Neural);
			UE_LOG(LogTemp, Log,
				TEXT("[NTMO] Curve in use (%s) | panel peak %.0f nit, fullframe %.0f nit, ABL ratio %.2f, HDR %d, outdev %d | knots(log2 out vs in):"),
				bNeuralCurve ? TEXT("neural") : TEXT("analytic"),
				Ctx.MaxLuminanceNits, Ctx.MaxFullFrameLuminanceNits,
				Ctx.ABLRatio, Ctx.bHDREnabled ? 1 : 0, Ctx.OutputDevice);
			UE_LOG(LogTemp, Log, TEXT("[NTMO]   sat=%.3f gamut=%.3f desat=%.3f lift=%.4f"),
				Curve.Saturation, Curve.GamutCompress, Curve.HighlightDesat, Curve.BlackLift);
			for (int32 i = 0; i < 8; ++i)
			{
				const float Xin = FNTMOCurveParams::KnotX0
					+ (FNTMOCurveParams::KnotX1 - FNTMOCurveParams::KnotX0) * (float)i / 7.0f;
				UE_LOG(LogTemp, Log, TEXT("[NTMO]   in %7.3f -> out %7.3f  (delta %+7.3f stops)"),
					Xin, Curve.Knots[i], Curve.Knots[i] - Xin);
			}
		}
	}

	// ---- Unreal's exposure -------------------------------------------------
	// PostProcessTonemap.usf applies the exposure chain inside the pass we replace, so it
	// has to happen somewhere or the image is off by the whole exposure factor. It is
	// folded into the LUT bake rather than the pixel shader, because binding the view
	// uniform buffer / eye adaptation buffer into the tonemap pixel shader made its
	// graphics PSO fail to create, which UE treats as fatal.
	//
	// GetLastEyeAdaptationExposure() is the CPU-side eye adaptation result UE exposes on
	// the view. It lags one frame, which is invisible for tone mapping (exposure is
	// temporally smoothed upstream), and it keeps automatic exposure fully in charge:
	// every exposure setting, bias and compensation still funnels through it.
	// Exposure is applied in the tonemap pixel shader, exactly as Unreal does it:
	//   Exposure = View.OneOverPreExposure * EyeAdaptationBuffer[0].x
	// The eye adaptation buffer is Unreal's own, so automatic exposure, manual
	// exposure and every exposure bias keep working untouched.
	FRDGBufferRef EyeAdaptationBuffer = GetEyeAdaptationBuffer(GraphBuilder, View);
	if (EyeAdaptationBuffer == nullptr)
	{
		const FVector4f DefaultEyeAdaptation(1.0f, 0.0f, 0.0f, 1.0f);
		EyeAdaptationBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("NTMO.DefaultEyeAdaptation"),
			sizeof(DefaultEyeAdaptation), 1, &DefaultEyeAdaptation,
			sizeof(DefaultEyeAdaptation), ERDGInitialDataFlags::None);
	}

	// NTMOExposure stays at 1: exposure belongs to the pixel shader now.
	const float ExposureValue = 1.0f;

	int32 LutSize = CVarNTMOLUTSize.GetValueOnRenderThread();
	if (LutSize <= 0)
	{
		LutSize = FMath::Clamp(GetDefault<UNTMOSettings>()->LUTSize, 16, 64);
	}

	const FRDGTextureDesc LutDesc = FRDGTextureDesc::Create3D(
		LutSize3D(LutSize),
		PF_FloatRGBA,
		FClearValueBinding::Black,
		TexCreate_ShaderResource | TexCreate_UAV);

	FRDGTextureRef LutTexture = GraphBuilder.CreateTexture(LutDesc, TEXT("NTMO.LUT"));

	// The bake is DEFERRED. The 8 knots may be produced by the neural network, whose
	// inference needs the luminance pyramid first, so the required order is:
	//   pyramid -> stats -> assemble tensor -> NNE EnqueueRDG -> LUT bake -> tonemap
	// Baking before the network ran would bake the analytic fallback curve instead.
	//
	// The curve is ALWAYS read from a buffer, never from loose uniforms, so there is
	// exactly one code path in the shader: on the analytic path the CPU knots are
	// uploaded into that buffer instead.
	FRDGBufferRef KnotsBuffer = nullptr;

	auto BakeLut = [&](FRDGBufferRef InKnots)
	{
		FNTMOLutBakeCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNTMOLutBakeCS::FParameters>();
		PassParameters->CurveKnots01 = FVector4f(
			Curve.Knots[0], Curve.Knots[1], Curve.Knots[2], Curve.Knots[3]);
		PassParameters->CurveKnots23 = FVector4f(
			Curve.Knots[4], Curve.Knots[5], Curve.Knots[6], Curve.Knots[7]);
		PassParameters->NTMOKnotsBuffer = GraphBuilder.CreateSRV(InKnots);
		PassParameters->NTMOCurveFromBuffer = 1;
		PassParameters->NTMOExposure = ExposureValue;
		PassParameters->NTMOSaturation = Curve.Saturation;
		PassParameters->NTMOGamutCompress = Curve.GamutCompress;
		PassParameters->NTMOHighlightDesat = Curve.HighlightDesat;
		PassParameters->NTMOBlackLift = Curve.BlackLift;
		PassParameters->NTMOMinNits = Ctx.MinLuminanceNits;
		PassParameters->NTMOMaxNits = Ctx.MaxLuminanceNits;
		PassParameters->NTMOMaxFullFrameNits = Ctx.MaxFullFrameLuminanceNits;
		PassParameters->NTMOPaperWhiteNits = Ctx.PaperWhiteNits;
		PassParameters->NTMOLutSize = LutSize;
		PassParameters->RWOutputLUT = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(LutTexture));

		const int32 GroupCount = FMath::DivideAndRoundUp(LutSize, 4);
		const FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(View.GetFeatureLevel());
		TShaderMapRef<FNTMOLutBakeCS> ComputeShaderRef(ShaderMap);
		const TShaderRef<FNTMOLutBakeCS> ComputeShader = ComputeShaderRef;

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("NTMO::LutBake(%d)", LutSize),
			ComputeShader,
			PassParameters,
			FIntVector(GroupCount, GroupCount, GroupCount));
	};

	// ---- Local operator: guide gain + half-resolution Laplacian pyramid ---
	// Four single-channel R16F levels at 1/2, 1/4, 1/8, 1/16 plus one half-res gain
	// map. Total traffic is ~2.7 Mpix * 2B * 2 (read+write) ~= 11 MB at 4K, i.e.
	// roughly 0.05-0.06 ms, and no full-resolution pass is added: the gain is consumed
	// by the tonemap pass that already runs.
	const UNTMOSettings* LocalSettings = GetDefault<UNTMOSettings>();
	const int32 LocalOverride = CVarNTMOLocalOverride.GetValueOnRenderThread();
	const bool bLocalOperator = (LocalOverride >= 0)
		? (LocalOverride != 0)
		: LocalSettings->bLocalOperator;

	// Neural inference needs the same luminance pyramid the local operator builds, so
	// either consumer is enough to pay for it. The model must have been accepted by
	// NNE (EnsureProbed on the game thread) before we can enqueue it.
	const int32 NeuralOverride = CVarNTMONeuralOverride.GetValueOnRenderThread();
	const bool bUseNeural = (NeuralOverride != 0)
		&& NeuralPolicy.IsValid()
		&& NeuralPolicy->IsReady()
		&& NeuralPolicy->GetModelInstance().IsValid();

	FRDGTextureRef LocalGainTexture = SceneColor.Texture;
	FRDGTextureRef LocalGuideTexture = SceneColor.Texture;
	FIntPoint LocalGainExtent(1, 1);

	if (bLocalOperator || bUseNeural)
	{
		const FIntRect SrcRect = SceneColor.ViewRect;
		const FIntPoint SrcSize(FMath::Max(1, SrcRect.Width()), FMath::Max(1, SrcRect.Height()));

		FIntPoint Sizes[4];
		Sizes[0] = FIntPoint(FMath::Max(1, SrcSize.X / 2), FMath::Max(1, SrcSize.Y / 2));
		for (int32 i = 1; i < 4; ++i)
		{
			Sizes[i] = FIntPoint(FMath::Max(1, Sizes[i - 1].X / 2), FMath::Max(1, Sizes[i - 1].Y / 2));
		}

		const ETextureCreateFlags LumFlags = TexCreate_ShaderResource | TexCreate_UAV;
		FRDGTextureRef Lum[4] = { nullptr, nullptr, nullptr, nullptr };
		for (int32 i = 0; i < 4; ++i)
		{
			Lum[i] = GraphBuilder.CreateTexture(
				FRDGTextureDesc::Create2D(Sizes[i], PF_R16F, FClearValueBinding::None, LumFlags),
				TEXT("NTMO.Local.Lum"));
		}

		// The correction map is produced AFTER the network ran (it needs the 48 learned
		// strengths), so the texture is only allocated here and filled further down.
		// RG16F: R = pre-curve scene gain, G = post-curve display offset (both in stops).
		FRDGTextureRef CorrectionTex = GraphBuilder.CreateTexture(
			FRDGTextureDesc::Create2D(Sizes[0], PF_G16R16F, FClearValueBinding::None, LumFlags),
			TEXT("NTMO.Local.Correction"));

		const FGlobalShaderMap* LocalShaderMap = GetGlobalShaderMap(View.GetFeatureLevel());
		const auto ClampSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();

		// Frame statistics live here rather than inside the neural branch because the
		// local correction pass needs the mean log2 luminance for its exposure band,
		// and that pass must also work when the network is disabled.
		const int32 NumStats = 34;	// see NTMO_STAT_* in NTMOFeatures.usf
		TArray<uint32> ZeroStats;
		ZeroStats.SetNumZeroed(NumStats);
		FRDGBufferRef StatsBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("NTMO.Feat.Stats"), sizeof(uint32), NumStats,
			ZeroStats.GetData(), ZeroStats.Num() * sizeof(uint32), ERDGInitialDataFlags::None);

		{
			FNTMOSceneStatsCS::FParameters* P = GraphBuilder.AllocParameters<FNTMOSceneStatsCS::FParameters>();
			P->NTMOStatsLum = Lum[2];
			P->NTMOStatsLumSampler = ClampSampler;
			P->RWNTMOStats = GraphBuilder.CreateUAV(StatsBuffer);
			P->NTMOStatsWidth = (uint32)Sizes[2].X;
			P->NTMOStatsHeight = (uint32)Sizes[2].Y;

			TShaderMapRef<FNTMOSceneStatsCS> StatsRef(LocalShaderMap);
			const TShaderRef<FNTMOSceneStatsCS> StatsShader = StatsRef;
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NTMO::SceneStats"),
				StatsShader, P, FComputeShaderUtils::GetGroupCount(Sizes[2], 8));
		}

		// G0: full-res scene colour -> half-res log2 luminance
		{
			FNTMOLumFromSceneCS::FParameters* P = GraphBuilder.AllocParameters<FNTMOLumFromSceneCS::FParameters>();
			P->NTMOSceneColorLocal = SceneColor.Texture;
			P->NTMOSceneColorLocalSampler = ClampSampler;
			P->NTMOSceneColorTexelSize = FVector2f(1.0f / (float)SrcSize.X, 1.0f / (float)SrcSize.Y);
			P->RWNTMOG0 = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(Lum[0]));

			TShaderMapRef<FNTMOLumFromSceneCS> Ref(LocalShaderMap);
			const TShaderRef<FNTMOLumFromSceneCS> Sh = Ref;
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NTMO::LocalLum"),
				Sh, P, FComputeShaderUtils::GetGroupCount(Sizes[0], 8));
		}

		// G1..G3
		for (int32 i = 1; i < 4; ++i)
		{
			FNTMODownsampleCS::FParameters* P = GraphBuilder.AllocParameters<FNTMODownsampleCS::FParameters>();
			P->NTMOLevelIn = Lum[i - 1];
			P->NTMOLevelInSampler = ClampSampler;
			P->NTMOLevelInTexelSize = FVector2f(1.0f / (float)Sizes[i - 1].X, 1.0f / (float)Sizes[i - 1].Y);
			P->RWNTMOLevelOut = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(Lum[i]));

			TShaderMapRef<FNTMODownsampleCS> Ref(LocalShaderMap);
			const TShaderRef<FNTMODownsampleCS> Sh = Ref;
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NTMO::LocalDown%d", i),
				Sh, P, FComputeShaderUtils::GetGroupCount(Sizes[i], 8));
		}

		// The correction pass is emitted after the network ran; see below.

		// ---- Neural inference -------------------------------------------
		// Builds the 1x48x72x128 input tensor on the GPU (channels 3..47 are constants
		// across the image, exactly as assemble_tensor_torch broadcasts them during
		// training), then lets NNE emit the 8 knots straight into a buffer. Nothing is
		// read back to the CPU: the LUT bake below consumes the buffer directly.
		// Decide whether to run the network this frame. The curve is a global, slowly
		// varying statistic, so running it every N frames and reusing the last result is
		// invisible -- and it is where nearly all of the plugin's cost lives.
		bool bRunInferenceThisFrame = false;
		if (bUseNeural)
		{
			const double NowForInterval = FPlatformTime::Seconds();
			const bool bHistogramOk = KnotsHistory.IsValid() && (NowForInterval - KnotsHistoryTime) < 1.0;
			const int32 IntervalOverride = CVarNTMOInterval.GetValueOnRenderThread();
			const int32 Interval = FMath::Clamp(
				IntervalOverride > 0
					? IntervalOverride
					: (LocalSettings ? LocalSettings->InferenceInterval : 1),
				1, 8);
			static int32 InferenceTick = 0;
			bRunInferenceThisFrame = !bHistogramOk || ((InferenceTick++ % Interval) == 0);
		}

		if (bUseNeural && !bRunInferenceThisFrame)
		{
			// Reuse the previous smoothed curve. No passes at all this frame.
			KnotsBuffer = GraphBuilder.RegisterExternalBuffer(KnotsHistory);
		}
		else if (bUseNeural)
		{
			const int32 Plane = NTMOInput::Height * NTMOInput::Width;
			const int32 InputElements = NTMOInput::Channels * Plane;
			const int32 NumCtxScalars = 13;

			// StatsBuffer and its pass are created above, in the pyramid scope, because
			// the local correction pass also needs them.

			// Display/scene scalars. These come from the same context the analytic
			// policy uses, normalised exactly like ntmo_pack.pack_sample().
			float CtxScalars[NumCtxScalars] = {};
			CtxScalars[0] = 0.0f;	// log2_mean  -- filled by the network's own stats? no: see note
			CtxScalars[1] = 0.0f;
			CtxScalars[2] = FMath::Min((float)Ctx.MaxLuminanceNits / 4000.0f, 1.0f);
			CtxScalars[3] = FMath::Min((float)Ctx.MaxFullFrameLuminanceNits / 4000.0f, 1.0f);
			CtxScalars[4] = FMath::Clamp(FMath::LogX(10.0f, FMath::Max((float)Ctx.MinLuminanceNits, 1e-4f)) / 4.0f, -1.0f, 0.0f);
			CtxScalars[5] = FMath::Min((float)Ctx.PaperWhiteNits / 300.0f, 1.5f);
			CtxScalars[6] = Ctx.bHDREnabled ? 1.0f : 0.0f;
			{
				const int32 OutDev = OutputDeviceToShaderValue(Ctx.OutputDevice);
				CtxScalars[7] = (Ctx.OutputDevice <= 2) ? 1.0f : 0.0f;
				CtxScalars[8] = (OutDev == 3) ? 1.0f : 0.0f;
				CtxScalars[9] = (OutDev == 4) ? 1.0f : 0.0f;
			}
			CtxScalars[10] = (float)Ctx.OutputGamut / 2.0f;
			CtxScalars[11] = Ctx.UserBrightnessBias / 3.0f;
			CtxScalars[12] = Ctx.Intensity;

			FRDGBufferRef CtxBuffer = CreateStructuredBuffer(
				GraphBuilder, TEXT("NTMO.Feat.Ctx"), sizeof(float), NumCtxScalars,
				CtxScalars, sizeof(CtxScalars), ERDGInitialDataFlags::None);

			FRDGBufferRef InputTensor = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(float), InputElements),
				TEXT("NTMO.Feat.Input"));

			{
				FNTMOAssembleCS::FParameters* P = GraphBuilder.AllocParameters<FNTMOAssembleCS::FParameters>();
				P->NTMOSceneColorFeat = SceneColor.Texture;
				P->NTMOSceneColorFeatSampler = ClampSampler;
				P->NTMOStats = GraphBuilder.CreateSRV(StatsBuffer);
				P->NTMOCtxScalars = GraphBuilder.CreateSRV(CtxBuffer);
				P->RWNTMOInputTensor = GraphBuilder.CreateUAV(InputTensor);
				// 1/2-res log2 luminance drives the edge channel.
				P->NTMOGuideTex = Lum[0];
				P->NTMOGuideTexSampler = ClampSampler;
				P->NTMOGuideTexelSize = FVector2f(
					1.0f / (float)FMath::Max(1, Sizes[0].X),
					1.0f / (float)FMath::Max(1, Sizes[0].Y));
				P->NTMOFeatW = (uint32)NTMOInput::Width;
				P->NTMOFeatH = (uint32)NTMOInput::Height;
				P->NTMOFeatPlane = (uint32)Plane;

				TShaderMapRef<FNTMOAssembleCS> Ref(LocalShaderMap);
				const TShaderRef<FNTMOAssembleCS> Sh = Ref;
				FComputeShaderUtils::AddPass(
					GraphBuilder,
					RDG_EVENT_NAME("NTMO::AssembleInput(%dx%dx%d)",
						NTMOInput::Channels, NTMOInput::Height, NTMOInput::Width),
					Sh, P,
					FIntVector(FMath::DivideAndRoundUp(NTMOInput::Width, 8),
						FMath::DivideAndRoundUp(NTMOInput::Height, 8), 1));
			}

			// Network output: 8 curve knots + 48 local Laplacian strengths.
			FRDGBufferRef NeuralKnots = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateStructuredDesc(sizeof(float), NTMOInput::Outputs),
				TEXT("NTMO.Neural.Out"));

			{
				TArray<UE::NNE::FTensorBindingRDG> InBindings;
				InBindings.Emplace(UE::NNE::FTensorBindingRDG{ InputTensor });

				TArray<UE::NNE::FTensorBindingRDG> OutBindings;
				OutBindings.Emplace(UE::NNE::FTensorBindingRDG{ NeuralKnots });

				const UE::NNE::IModelInstanceRDG::EEnqueueRDGStatus EnqueueStatus =
					NeuralPolicy->GetModelInstance()->EnqueueRDG(GraphBuilder, InBindings, OutBindings);

				if (EnqueueStatus == UE::NNE::EResultStatus::Ok)
				{
					KnotsBuffer = NeuralKnots;

					// ---- Temporal smoothing ---------------------------------------
					// The network predicts each frame independently, so a hard content
					// change (cut, light switching on, bright object entering frame) makes
					// the curve jump and the whole image flashes for a frame or two. An
					// EMA over the 8 knots removes that. Blending two increasing knot
					// sequences stays increasing, so monotonicity is preserved exactly.
					if (LocalSettings != nullptr && LocalSettings->CurveSmoothingHalfLife > 0.0f)
					{
						const double NowSeconds = FPlatformTime::Seconds();
						const bool bHistoryValid = KnotsHistory.IsValid()
							&& (NowSeconds - KnotsHistoryTime) < 1.0;	// stale after a pause

						if (!KnotsHistory.IsValid())
						{
							KnotsHistory = AllocatePooledBuffer(
								FRDGBufferDesc::CreateStructuredDesc(sizeof(float), NTMOInput::Outputs),
								TEXT("NTMO.KnotsHistory"));
						}

						FRDGBufferRef PrevKnots = GraphBuilder.RegisterExternalBuffer(KnotsHistory);
						FRDGBufferRef SmoothKnots = GraphBuilder.CreateBuffer(
							FRDGBufferDesc::CreateStructuredDesc(sizeof(float), NTMOInput::Outputs),
							TEXT("NTMO.Neural.SmoothedKnots"));

						// alpha = 1 - exp(-dt / tau), tau = halfLife / ln2. Frame-rate
						// independent, and clamped so a hitch cannot snap the curve.
						float BlendAlpha = 1.0f;
						if (bHistoryValid)
						{
							const float DtSec = FMath::Clamp(
								(float)(NowSeconds - KnotsHistoryTime), 1.0f / 240.0f, 0.25f);
							const float TauSec = FMath::Max(LocalSettings->CurveSmoothingHalfLife, 1.0e-3f) / 0.6931472f;
							BlendAlpha = FMath::Clamp(1.0f - FMath::Exp(-DtSec / TauSec), 0.02f, 1.0f);
						}

						{
							FNTMOBlendKnotsCS::FParameters* P = GraphBuilder.AllocParameters<FNTMOBlendKnotsCS::FParameters>();
							P->NTMOPrevKnots = GraphBuilder.CreateSRV(PrevKnots);
							P->NTMONewKnots = GraphBuilder.CreateSRV(NeuralKnots);
							P->RWNTMOSmoothedKnots = GraphBuilder.CreateUAV(SmoothKnots);
							P->NTMOBlendAlpha = BlendAlpha;

							TShaderMapRef<FNTMOBlendKnotsCS> BlendRef(LocalShaderMap);
							const TShaderRef<FNTMOBlendKnotsCS> BlendShader = BlendRef;
							FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NTMO::BlendKnots"),
								BlendShader, P, FIntVector(1, 1, 1));
						}

						// Feed the smoothed curve forward as next frame's history.
						AddCopyBufferPass(GraphBuilder, PrevKnots, SmoothKnots);
						GraphBuilder.QueueBufferExtraction(PrevKnots, &KnotsHistory);
						KnotsHistoryTime = NowSeconds;

						KnotsBuffer = SmoothKnots;
					}

					// One-shot verification readback (r.NTMO.DumpKnots 1).
					static bool bKnotsDumped = false;
					if (!bKnotsDumped && CVarNTMODumpKnots.GetValueOnRenderThread() != 0)
					{
						if (KnotsReadback.IsValid() && KnotsReadback->IsReady())
						{
							const float* Data = static_cast<const float*>(KnotsReadback->Lock(NTMOInput::Outputs * sizeof(float)));
							UE_LOG(LogTemp, Warning,
								TEXT("[NTMO] Neural knots (log2 out) from GPU: ")
								TEXT("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"),
								Data[0], Data[1], Data[2], Data[3], Data[4], Data[5], Data[6], Data[7]);
							// Local band strengths for the four corner tiles, so a
							// mis-signed exposure term is visible in the log instead of
							// having to be inferred from screenshots.
							if (NTMOInput::Outputs > NTMOInput::Knots + 4)
							{
								UE_LOG(LogTemp, Warning,
									TEXT("[NTMO] Local bands (s_base s_mid s_high s_low), tile0/5/10/15: ")
									TEXT("[%.3f %.3f %.3f %.3f] [%.3f %.3f %.3f %.3f] ")
									TEXT("[%.3f %.3f %.3f %.3f] [%.3f %.3f %.3f %.3f]"),
									Data[8+0], Data[8+1], Data[8+2], Data[8+3],
									Data[8+20], Data[8+21], Data[8+22], Data[8+23],
									Data[8+40], Data[8+41], Data[8+42], Data[8+43],
									Data[8+60], Data[8+61], Data[8+62], Data[8+63]);
							}
							UE_CLOG(!(Data[7] > Data[6] && Data[6] > Data[5] && Data[5] > Data[4]
								&& Data[4] > Data[3] && Data[3] > Data[2] && Data[2] > Data[1]
								&& Data[1] > Data[0]), LogTemp, Error,
								TEXT("[NTMO] Neural knots are NOT monotonically increasing - the LUT "
									"would invert contrast. This indicates garbage network output."));
							KnotsReadback->Unlock();
							KnotsReadback.Reset();
							CVarNTMODumpKnots->Set(0, ECVF_SetByCode);
						}
						else if (!KnotsReadback.IsValid())
						{
							KnotsReadback = MakeUnique<FRHIGPUBufferReadback>(TEXT("NTMO.KnotsReadback"));
							AddEnqueueCopyPass(GraphBuilder, KnotsReadback.Get(), NeuralKnots, NTMOInput::Outputs * sizeof(float));
						}
					}

					static bool bLoggedNeuralOnce = false;
					if (!bLoggedNeuralOnce)
					{
						bLoggedNeuralOnce = true;
						UE_LOG(LogTemp, Log,
							TEXT("[NTMO] Neural inference enqueued (input %dx%dx%d float, output %d values: 8 knots + 64 local). ")
							TEXT("The LUT is now baked from the network output."),
							NTMOInput::Channels, NTMOInput::Height, NTMOInput::Width, NTMOInput::Outputs);
					}
				}
				else
				{
					static bool bLoggedEnqueueFail = false;
					if (!bLoggedEnqueueFail)
					{
						bLoggedEnqueueFail = true;
						UE_LOG(LogTemp, Error,
							TEXT("[NTMO] EnqueueRDG failed with status %d; keeping the analytic curve."),
							(int32)EnqueueStatus);
					}
				}
			}
		}

		// ---- Fallback strength set ----------------------------------------
		// With the neural path off (CurveSource=Analytic, or r.NTMO.Neural 0) there is
		// no network output, but the local operator still needs a 48-float parameter
		// set. These are the v1.0.1 hand-tuned constants expressed in the new
		// de-normalised units, so the analytic fallback keeps the same look it had.
		if (KnotsBuffer == nullptr)
		{
			float Defaults[NTMOInput::Outputs] = {};
			for (int32 i = 0; i < NTMOInput::Knots; ++i)
			{
				Defaults[i] = Curve.Knots[i];
			}
			// Analytic fallback strengths (label space, [-1,1] == +-1.5 stops):
			// base compression, coarse detail, fine detail, and local exposure. The
			// exposure value reproduces v1.0.1's shadow-opening character.
			const float BandDefaults[NTMOInput::LocalBands] = {
				0.90f / 1.5f, 0.25f / 1.5f, 0.10f / 1.5f, -0.35f / 1.5f };
			for (int32 Tile = 0; Tile < NTMOInput::LocalTiles * NTMOInput::LocalTiles; ++Tile)
			{
				for (int32 Band = 0; Band < NTMOInput::LocalBands; ++Band)
				{
					Defaults[NTMOInput::Knots + Tile * NTMOInput::LocalBands + Band] = BandDefaults[Band];
				}
			}
			KnotsBuffer = CreateStructuredBuffer(
				GraphBuilder, TEXT("NTMO.NeuralOut.Default"), sizeof(float), NTMOInput::Outputs,
				Defaults, sizeof(Defaults), ERDGInitialDataFlags::None);
		}

		// ---- Learned local Laplacian correction ---------------------------
		// Runs AFTER inference because it needs the 48 predicted strengths, and BEFORE
		// the LUT bake because it evaluates the curve itself.
		if (bLocalOperator)
		{
			FNTMOCorrectionCS::FParameters* P = GraphBuilder.AllocParameters<FNTMOCorrectionCS::FParameters>();
			P->NTMOLum0 = Lum[0];
			P->NTMOLum0Sampler = ClampSampler;
			P->NTMOLum1 = Lum[1];
			P->NTMOLum1Sampler = ClampSampler;
			P->NTMOLum2 = Lum[2];
			P->NTMOLum2Sampler = ClampSampler;
			P->NTMOLum3 = Lum[3];
			P->NTMOLum3Sampler = ClampSampler;
			P->NTMONeuralOut = GraphBuilder.CreateSRV(KnotsBuffer);
			P->NTMOStatsBuf = GraphBuilder.CreateSRV(StatsBuffer);
			P->NTMOStrengthScale = 1.5f;	// matches S_MAX in ntmo_local.py
			P->NTMODetailMix = FMath::Clamp(LocalSettings->LocalDetailMix, 0.0f, 1.0f);
			P->NTMOExposureScale = FMath::Clamp(LocalSettings->LocalExposureScale, 0.0f, 2.0f);
			P->NTMOClampStops = FMath::Max(0.25f, LocalSettings->LocalGainClampStops);
			P->RWNTMOCorrection = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(CorrectionTex));

			TShaderMapRef<FNTMOCorrectionCS> Ref(LocalShaderMap);
			const TShaderRef<FNTMOCorrectionCS> Sh = Ref;
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("NTMO::LocalCorrection"),
				Sh, P, FComputeShaderUtils::GetGroupCount(Sizes[0], 8));

			LocalGainTexture = CorrectionTex;
			LocalGuideTexture = Lum[0];
			LocalGainExtent = Sizes[0];
		}
	}

	// ---- Curve source, then the bake --------------------------------------
	// If the network produced a knots buffer above we use it; otherwise the CPU curve
	// is uploaded into the same buffer. Either way the shader has ONE code path: it
	// always reads the buffer.
	if (KnotsBuffer == nullptr)
	{
		// Local operator and neural path both off: upload the curve plus neutral
		// strengths so the shaders still have a legal 56-float buffer to read.
		float Defaults[NTMOInput::Outputs] = {};
		for (int32 i = 0; i < NTMOInput::Knots; ++i)
		{
			Defaults[i] = Curve.Knots[i];
		}
		KnotsBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("NTMO.NeuralOut.Fallback"), sizeof(float), NTMOInput::Outputs,
			Defaults, sizeof(Defaults), ERDGInitialDataFlags::None);
	}

	BakeLut(KnotsBuffer);

	// ---- Final tonemap pass ----------------------------------------------

	{
		FNTMOTonemapPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNTMOTonemapPS::FParameters>();
		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->NTMOEyeAdaptationBuffer = GraphBuilder.CreateSRV(EyeAdaptationBuffer);
		{
			const FLinearColor& Tint = View.FinalPostProcessSettings.SceneColorTint;
			PassParameters->NTMOSceneColorTint = FVector3f(Tint.R, Tint.G, Tint.B);
		}
		PassParameters->NTMOApplyExposure = 1;
		PassParameters->NTMOSceneColor = SceneColor.Texture;
		PassParameters->NTMOSceneColorSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		PassParameters->NTMOBloom = Bloom.IsValid() ? Bloom.Texture : SceneColor.Texture;
		PassParameters->NTMOBloomSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		PassParameters->NTMOLut = LutTexture;
		PassParameters->NTMOLutSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		// Bilinear on purpose now: the correction is a smooth log2 field produced from a
		// 4x4 learned parameter map, so there is no high-contrast ratio to protect and
		// bilinear upsampling is exactly right. This replaces the 8-tap bilateral
		// upsample the previous version needed for its gain map.
		PassParameters->NTMOLocalCorrection = LocalGainTexture;
		PassParameters->NTMOLocalCorrectionSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
		PassParameters->NTMOLocalEnabled = bLocalOperator ? 1 : 0;
		PassParameters->NTMOLutSize = (float)LutSize;
		PassParameters->NTMOLutScale = (float)(LutSize - 1) / (float)LutSize;
		PassParameters->NTMOLutOffset = 0.5f / (float)LutSize;
		PassParameters->NTMOBloomScale = Bloom.IsValid() ? CVarNTMOBloomScale.GetValueOnRenderThread() : 0.0f;
		PassParameters->NTMOOutputDevice = OutputDeviceToShaderValue(Ctx.OutputDevice);
		PassParameters->NTMOGamma = 1.0f / FMath::Max(CVarNTMOGamma.GetValueOnRenderThread(), 1.0f);
		PassParameters->RenderTargets[0] = FRenderTargetBinding(Output.Texture, Output.LoadAction);

		const FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(View.GetFeatureLevel());
		TShaderMapRef<FNTMOTonemapPS> PixelShaderRef(ShaderMap);
		const TShaderRef<FNTMOTonemapPS> PixelShader = PixelShaderRef;

		{
			static bool bLoggedPS = false;
			if (!bLoggedPS)
			{
				bLoggedPS = true;
				UE_LOG(LogTemp, Warning,
					TEXT("[NTMO] Tonemap PS diagnostic: refValid=%d rhiShader=%d shaderMap=%d featureLevel=%d"),
					PixelShader.IsValid() ? 1 : 0,
					(PixelShader.IsValid() && PixelShader.GetPixelShader() != nullptr) ? 1 : 0,
					ShaderMap != nullptr ? 1 : 0,
					(int32)View.GetFeatureLevel());
			}
		}

		// AddDrawScreenPass, not FPixelShaderUtils::AddFullscreenPass.
		//
		// With AddFullscreenPass every draw died at D3D12 PSO creation
		// ("Failed to create graphics PSO ... Pixel: <unknown>", which UE escalates to
		// the misleading "Shader compilation failures are Fatal"), even though the
		// diagnostic proved the RHI pixel shader existed and the shader map was valid.
		// AddDrawScreenPass is what Unreal's own post process passes use; it sets up the
		// screen pass viewports, vertex shader and pipeline state through the same path
		// the engine's tonemapper takes.
		AddDrawScreenPass(
			GraphBuilder,
			RDG_EVENT_NAME("NTMO::Tonemap"),
			View,
			FScreenPassTextureViewport(Output),
			FScreenPassTextureViewport(SceneColor),
			PixelShader,
			PassParameters);
	}

	return FScreenPassTexture(Output);
}
