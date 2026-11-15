// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

class FNTMOSceneViewExtension;

class FNeuralToneMapperModule : public IModuleInterface
{
public:
	//~ Begin IModuleInterface
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
	//~ End IModuleInterface

	/**
	 * Console command entry points, also used by the editor Build menu.
	 * Exported so the editor module can drive the switch.
	 */
	static NEURALTONEMAPPER_API void SetEnabled(bool bInEnabled);
	static NEURALTONEMAPPER_API bool IsEnabled();

private:
	void RegisterConsoleCommands();
	void UnregisterConsoleCommands();

	/**
	 * Creates the scene view extension once GEngine exists.
	 *
	 * Deferred on purpose: this module loads at PostConfigInit so that the
	 * AddShaderSourceDirectoryMapping for "/NTMO" is registered before any shader is
	 * compiled, but FSceneViewExtensions::NewExtension() requires GEngine and asserts
	 * if it is null. Building the extension during StartupModule produced
	 * "Ensure condition failed: GEngine" from SceneViewExtension.cpp the moment the
	 * plugin was enabled.
	 */
	void CreateViewExtensionIfNeeded();

	FDelegateHandle PostEngineInitHandle;

	TSharedPtr<FNTMOSceneViewExtension, ESPMode::ThreadSafe> ViewExtension;

	TArray<IConsoleObject*> ConsoleCommands;
};
