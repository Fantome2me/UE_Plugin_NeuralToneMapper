// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

/**
 * Editor module.
 *
 * Adds a checkable entry to the Level Editor's "Build" (构建) menu:
 *
 *     AI动态色调映射  [x]
 *
 * Ticking it disables Unreal's own tone mapper and enables the plugin's HDR tone
 * mapper; unticking restores Unreal's tonemapper. The state persists in the
 * project's config and is mirrored onto r.NTMO.Enable so automation and console
 * users see the same switch.
 */
class FNeuralToneMapperEditorModule : public IModuleInterface
{
public:
	//~ Begin IModuleInterface
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
	//~ End IModuleInterface

private:
	void RegisterMenus();

	static void ToggleAI();
	static void EnableAI();
	static void DisableAI();
	static ECheckBoxState GetAIState();
	static bool IsAIMenuEnabled();

	static FText GetLabel();
	static FText GetToolTip();
};
