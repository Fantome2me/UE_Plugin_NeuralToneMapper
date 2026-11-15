// Copyright (c) 2026. All Rights Reserved.

#include "NeuralToneMapperModule.h"

#include "NTMOSceneViewExtension.h"
#include "NTMOSettings.h"

#include "HAL/IConsoleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/CoreDelegates.h"
#include "Misc/Paths.h"
#include "RenderCore.h"
#include "SceneViewExtension.h"
#include "ShaderCore.h"

#define LOCTEXT_NAMESPACE "FNeuralToneMapperModule"

static const TCHAR* GNTMOPluginName = TEXT("NeuralToneMapper");

void FNeuralToneMapperModule::StartupModule()
{
	// Map the plugin's shader folder so the global shader IMPLEMENT_GLOBAL_SHADER
	// virtual paths ("/NTMO/Private/...") resolve.
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(GNTMOPluginName);
	if (Plugin.IsValid())
	{
		const FString ShaderDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/NTMO"), ShaderDir);
	}
	else
	{
		UE_LOG(LogTemp, Error,
			TEXT("[NTMO] Could not find the NeuralToneMapper plugin. Shaders will fail to compile."));
	}

	// Create the view extension now if the engine already exists (this module can also
	// be loaded later, e.g. when the plugin is enabled at runtime), otherwise defer to
	// OnPostEngineInit. It is inert until the settings/cvar enable it either way.
	if (GEngine != nullptr)
	{
		CreateViewExtensionIfNeeded();
	}
	else
	{
		PostEngineInitHandle = FCoreDelegates::OnPostEngineInit.AddRaw(
			this, &FNeuralToneMapperModule::CreateViewExtensionIfNeeded);
	}

	RegisterConsoleCommands();

	// Do NOT query the settings CDO here. This module loads at PostConfigInit, and
	// GetDefault<UNTMOSettings>() returns null at that point -- the resulting null
	// dereference crashed the editor on startup. The enabled state is read from the
	// game thread by FNTMOSceneViewExtension::SetupViewFamily before the first render,
	// so nothing is lost by staying quiet here.
	UE_LOG(LogTemp, Log, TEXT("[NTMO] Neural Tone Mapper loaded (state is read per-frame from settings)."));
}

void FNeuralToneMapperModule::CreateViewExtensionIfNeeded()
{
	if (ViewExtension.IsValid())
	{
		return;
	}

	if (GEngine == nullptr)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[NTMO] GEngine is not available yet; the scene view extension was not created."));
		return;
	}

	ViewExtension = FSceneViewExtensions::NewExtension<FNTMOSceneViewExtension>();

	UE_LOG(LogTemp, Log, TEXT("[NTMO] Scene view extension created (GEngine present)."));
}

void FNeuralToneMapperModule::ShutdownModule()
{
	if (PostEngineInitHandle.IsValid())
	{
		FCoreDelegates::OnPostEngineInit.Remove(PostEngineInitHandle);
		PostEngineInitHandle.Reset();
	}

	UnregisterConsoleCommands();

	// The extension unregisters itself when the last reference goes away.
	ViewExtension.Reset();
}

void FNeuralToneMapperModule::RegisterConsoleCommands()
{
	IConsoleManager& Manager = IConsoleManager::Get();

	ConsoleCommands.Add(Manager.RegisterConsoleCommand(
		TEXT("NTMO.Enable"),
		TEXT("Enable the AI tone mapper."),
		FConsoleCommandDelegate::CreateStatic([]() { FNeuralToneMapperModule::SetEnabled(true); })));

	ConsoleCommands.Add(Manager.RegisterConsoleCommand(
		TEXT("NTMO.Disable"),
		TEXT("Disable the AI tone mapper (Unreal's tonemapper takes over)."),
		FConsoleCommandDelegate::CreateStatic([]() { FNeuralToneMapperModule::SetEnabled(false); })));

	ConsoleCommands.Add(Manager.RegisterConsoleCommand(
		TEXT("NTMO.Toggle"),
		TEXT("Toggle the AI tone mapper."),
		FConsoleCommandDelegate::CreateStatic([]()
		{
			FNeuralToneMapperModule::SetEnabled(!FNeuralToneMapperModule::IsEnabled());
		})));
}

void FNeuralToneMapperModule::UnregisterConsoleCommands()
{
	for (IConsoleObject* Command : ConsoleCommands)
	{
		if (Command)
		{
			IConsoleManager::Get().UnregisterConsoleObject(Command, /*bKeepState*/ false);
		}
	}
	ConsoleCommands.Reset();
}

void FNeuralToneMapperModule::SetEnabled(bool bInEnabled)
{
	UNTMOSettings::SetEnabled(bInEnabled);

	UE_LOG(LogTemp, Log, TEXT("[NTMO] AI tone mapping %s"),
		bInEnabled ? TEXT("ENABLED (Unreal tonemapper disabled)") : TEXT("DISABLED (Unreal tonemapper restored)"));
}

bool FNeuralToneMapperModule::IsEnabled()
{
	return UNTMOSettings::IsEnabled();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FNeuralToneMapperModule, NeuralToneMapper)
