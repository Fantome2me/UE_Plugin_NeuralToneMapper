// Copyright (c) 2026. All Rights Reserved.

#include "NTMOPolicy.h"
#include "NTMOSettings.h"

#include "HDRHelper.h"
#include "HAL/IConsoleManager.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Framework/Application/SlateApplication.h"
#include "GenericPlatform/GenericWindow.h"
#include "RHI.h"
#include "Widgets/SWindow.h"

// ---------------------------------------------------------------------------
// CVars
// ---------------------------------------------------------------------------
static TAutoConsoleVariable<int32> CVarNTMOEnable(
	TEXT("r.NTMO.Enable"),
	-1,
	TEXT("AI tone mapping master switch.\n")
	TEXT(" -1: follow Project Settings (default)\n")
	TEXT("  0: force off (Unreal's own tonemapper runs)\n")
	TEXT("  1: force on (the plugin replaces the tonemapper)\n"),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNTMOForceFallback(
	TEXT("r.NTMO.ForceFallback"),
	0,
	TEXT("Ignore the AI tone mapper and let Unreal tonemap, for A/B comparison.\n"),
	ECVF_RenderThreadSafe);

static TAutoConsoleVariable<float> CVarNTMOIntensity(
	TEXT("r.NTMO.Intensity"),
	-1.0f,
	TEXT("Override the plugin intensity in [0,1]. Negative follows project settings.\n"),
	ECVF_RenderThreadSafe);

int32 NTMO_GetRequestedEnable()
{
	if (CVarNTMOForceFallback.GetValueOnAnyThread() != 0)
	{
		return 0;
	}
	return CVarNTMOEnable.GetValueOnAnyThread();
}

// ---------------------------------------------------------------------------
// FNTMOContextProvider
// ---------------------------------------------------------------------------
namespace
{
	const TCHAR* GOutputDeviceCVar = TEXT("r.HDR.Display.OutputDevice");
	const TCHAR* GColorGamutCVar = TEXT("r.HDR.Display.ColorGamut");
	const TCHAR* GMinLuminanceLog10CVar = TEXT("r.HDR.Display.MinLuminanceLog10");
	const TCHAR* GMaxLuminanceCVar = TEXT("r.HDR.Display.MaxLuminance");

	int32 ReadIntCVar(const TCHAR* Name, int32 Default)
	{
		if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(Name))
		{
			return CVar->GetInt();
		}
		return Default;
	}

	float ReadFloatCVar(const TCHAR* Name, float Default)
	{
		if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(Name))
		{
			return CVar->GetFloat();
		}
		return Default;
	}

	void* GetGameWindowHandle()
	{
		if (!FSlateApplication::IsInitialized())
		{
			return nullptr;
		}

		TSharedPtr<SWindow> Window = FSlateApplication::Get().GetActiveTopLevelWindow();
		if (!Window.IsValid() && GEngine && GEngine->GameViewport)
		{
			Window = GEngine->GameViewport->GetWindow();
		}

		if (Window.IsValid())
		{
			TSharedPtr<FGenericWindow> Native = Window->GetNativeWindow();
			if (Native.IsValid())
			{
				return Native->GetOSWindowHandle();
			}
		}

		return nullptr;
	}
}

void FNTMOContextProvider::RefreshFromGameThread(bool bForceRefresh)
{
	const double Now = FPlatformTime::Seconds();
	if (!bForceRefresh && (Now - LastRefreshTime) < 0.5)
	{
		return;
	}
	LastRefreshTime = Now;

	FNTMOContext New;
	New.LensAttenuation = ReadFloatCVar(TEXT("r.EyeAdaptation.LensAttenuation"), 0.78f);

	bool bHadMeta = false;

	if (void* OSWindow = GetGameWindowHandle())
	{
		FHDRMetaData Meta;
		FMemory::Memzero(Meta);

		const FVector2D TopLeft(0.0, 0.0);
		const FVector2D BottomRight(1.0, 1.0);
		HDRGetMetaDataForWindow(Meta, TopLeft, BottomRight, OSWindow);

		if (Meta.MaximumLuminanceInNits > 1.0f)
		{
			New.MinLuminanceNits = Meta.MinimumLuminanceInNits;
			New.MaxLuminanceNits = Meta.MaximumLuminanceInNits;
			New.MaxFullFrameLuminanceNits = Meta.MaximumFullFrameLuminanceInNits;
			New.PaperWhiteNits = Meta.HDRPaperWhiteInNits;
			New.OutputDevice = (int32)Meta.DisplayOutputFormat;
			New.OutputGamut = (int32)Meta.DisplayColorGamut;
			New.bHDREnabled = (Meta.bHDRSupported && IsHDREnabled()) ? 1 : 0;
			bHadMeta = true;
		}
	}

	if (!bHadMeta)
	{
		// Fall back to exactly the cvars the engine's own tonemapper consumes.
		const int32 OutputDevice = ReadIntCVar(GOutputDeviceCVar, 0);
		New.OutputDevice = OutputDevice;
		New.OutputGamut = ReadIntCVar(GColorGamutCVar, 0);
		New.bHDREnabled = (OutputDevice >= 3 && OutputDevice <= 6) ? 1 : 0;

		New.MinLuminanceNits = FMath::Pow(10.0f, ReadFloatCVar(GMinLuminanceLog10CVar, -4.0f));

		// 0 means "the engine has not resolved MaxLuminance yet"; assume a
		// conservative 1000 nit HDR target / 100 nit SDR.
		const int32 MaxLum = ReadIntCVar(GMaxLuminanceCVar, 0);
		New.MaxLuminanceNits = MaxLum > 0 ? (float)MaxLum : (New.bHDREnabled ? 1000.0f : 100.0f);
		New.MaxFullFrameLuminanceNits = New.MaxLuminanceNits * 0.6f;
		New.PaperWhiteNits = New.bHDREnabled ? 203.0f : 100.0f;
	}

	// Manual override wins over both EDID and the engine cvars. This is the escape
	// hatch for displays that cannot report EDID (projectors, capture cards, virtual
	// displays) and the way to exercise the PQ / HDR path on an SDR desktop.
	if (const UNTMOSettings* Settings = GetDefault<UNTMOSettings>())
	{
		if (Settings->bOverrideDisplayMetadata)
		{
			New.MaxLuminanceNits = FMath::Max(Settings->OverridePeakNits, 10.0f);
			New.MaxFullFrameLuminanceNits = FMath::Clamp(
				Settings->OverrideFullFrameNits, 10.0f, New.MaxLuminanceNits);
			New.PaperWhiteNits = FMath::Max(Settings->OverridePaperWhiteNits, 10.0f);
			New.OutputDevice = FMath::Clamp(Settings->OverrideOutputDevice, 0, 4);
			New.bHDREnabled = (Settings->OverrideHDREnabled != 0) ? 1 : 0;
			bHadMeta = true;
		}
	}

	New.UpdateDerived();

	FScopeLock Lock(&Sync);
	New.Intensity = PendingIntensity;
	New.UserBrightnessBias = PendingBrightnessBias;
	Cached = New;
	bHadDisplayMetadata = bHadMeta;
}

void FNTMOContextProvider::UpdateUserOverrides(float InIntensity, float InBrightnessBias)
{
	FScopeLock Lock(&Sync);
	PendingIntensity = FMath::Clamp(InIntensity, 0.0f, 1.0f);
	PendingBrightnessBias = FMath::Clamp(InBrightnessBias, -3.0f, 3.0f);
	Cached.Intensity = PendingIntensity;
	Cached.UserBrightnessBias = PendingBrightnessBias;
}

FNTMOContext FNTMOContextProvider::GetContext() const
{
	FScopeLock Lock(&Sync);
	return Cached;
}

// ---------------------------------------------------------------------------
// Analytic curve policy
//
// Produces a curve with three regions, all monotone and C1 at the joints:
//   toe      - gentle shadow lift so the bottom never crushes to pure black
//   identity - middle range maps 1:1, which keeps middle grey fixed
//   shoulder - smooth exponential roll off toward the panel peak
//
// The shoulder point moves earlier as ABL gets stronger (MaxFullFrame/Max gets
// smaller), which is the part that ACES simply cannot express because it only
// sees a single peak luminance scalar.
// ---------------------------------------------------------------------------
namespace NTMPPolicy
{
	/** Peak luminance expressed in the 1.0 == 100 nits domain. */
	static float LogPeakFor(const FNTMOContext& Context)
	{
		const float PeakNits = FMath::Max(Context.MaxLuminanceNits, 50.0f);
		return FMath::Log2(PeakNits / 100.0f);
	}

	static float LogFloorFor(const FNTMOContext& Context)
	{
		const float FloorNits = FMath::Max(Context.MinLuminanceNits, 1.0e-4f);
		return FMath::Log2(FloorNits / 100.0f);
	}
}

bool FNTMOCurvePolicyAnalytic::ComputeCurve(const FNTMOContext& Context, FNTMOCurveParams& OutParams)
{
	using namespace NTMPPolicy;

	const float LogPeak = LogPeakFor(Context);
	const float LogFloor = LogFloorFor(Context);

	// ABL strength: 0 when the panel can sustain its peak, 1 when full frame peak
	// is far below the small area peak.
	const float Abl = Context.bHDREnabled
		? FMath::Clamp(1.0f - Context.ABLRatio, 0.0f, 1.0f) * ABLAwareness
		: 0.0f;

	// Where the shoulder begins, in log2. Earlier (lower) as ABL gets stronger.
	const float KneeX = FMath::Lerp(LogPeak - 0.5f, LogPeak - 3.0f, Abl);
	const float Span = FMath::Max(LogPeak - KneeX, 0.5f);

	const float ToeX = -4.0f;
	const float ToeLift = 0.20f;
	const float ToeRange = 6.0f;

	const float BrightnessBias = Context.UserBrightnessBias + Context.ExposureBias;
	const float Intensity = FMath::Clamp(Context.Intensity, 0.0f, 1.0f);

	OutParams = FNTMOCurveParams();
	OutParams.Saturation = FMath::Lerp(1.0f, Saturation, Intensity);
	OutParams.GamutCompress = GamutCompress * Intensity;
	OutParams.HighlightDesat = HighlightDesat * Intensity;
	OutParams.BlackLift = BlackLift * Intensity;

	for (int32 Index = 0; Index < 8; ++Index)
	{
		const float X = FNTMOCurveParams::KnotX0
			+ (FNTMOCurveParams::KnotX1 - FNTMOCurveParams::KnotX0) * (float)Index / 7.0f;

		float Y;
		if (X <= ToeX)
		{
			const float T = FMath::Clamp((ToeX - X) / ToeRange, 0.0f, 1.0f);
			Y = X + ToeLift * T * T;
		}
		else if (X <= KneeX)
		{
			Y = X;
		}
		else
		{
			// C1 continuous at the knee: slope is exactly 1 there and decays to 0.
			const float D = X - KneeX;
			Y = KneeX + Span * (1.0f - FMath::Exp(-D / Span));
		}

		// Global brightness preference, in stops.
		Y += BrightnessBias;

		// Blend toward identity as intensity drops: 0 gives a plain "linear + clip"
		// curve which is a useful regression baseline.
		Y = FMath::Lerp(X, Y, Intensity);

		// Never leave the panel's useful range, and keep knots strictly monotone so
		// the Fritsch-Carlson interpolation can never produce an inverted curve.
		Y = FMath::Clamp(Y, LogFloor + 0.01f, LogPeak);
		if (Index > 0)
		{
			Y = FMath::Max(Y, OutParams.Knots[Index - 1] + 1.0e-4f);
		}
		OutParams.Knots[Index] = Y;
	}

	return true;
}

FString FNTMOCurvePolicyAnalytic::GetStatusString() const
{
	return TEXT("Analytic");
}
