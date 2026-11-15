// Copyright (c) 2026. All Rights Reserved.

#include "NTMOSettings.h"

#include "HAL/IConsoleManager.h"

UNTMOSettings::UNTMOSettings()
{
	CategoryName = TEXT("Plugins");
}

void UNTMOSettings::SetEnabled(bool bInEnabled)
{
	UNTMOSettings* Settings = GetMutableDefault<UNTMOSettings>();
	if (!Settings)
	{
		return;
	}

	if (Settings->bEnabled != bInEnabled)
	{
		Settings->bEnabled = bInEnabled;
		Settings->SaveConfig();

		// Mirror into a cvar so automation, -ExecCmds and the console can drive it too,
		// and so that a Shipping build (no editor module) still has a usable switch.
		if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.NTMO.Enable")))
		{
			CVar->Set(bInEnabled ? 1 : 0, ECVF_SetByProjectSetting);
		}
	}
}

bool UNTMOSettings::IsEnabled()
{
	if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.NTMO.Enable")))
	{
		// -1 means "follow settings"; anything else is an explicit override.
		const int32 Value = CVar->GetInt();
		if (Value >= 0)
		{
			return Value != 0;
		}
	}

	// Null-check the CDO. GetDefault<>() returns null when the class's default object
	// has not been created yet, which is the case during early module loading phases
	// (PostConfigInit). An unguarded dereference there crashed the editor on startup.
	const UNTMOSettings* Settings = GetDefault<UNTMOSettings>();
	return Settings != nullptr && Settings->bEnabled;
}
