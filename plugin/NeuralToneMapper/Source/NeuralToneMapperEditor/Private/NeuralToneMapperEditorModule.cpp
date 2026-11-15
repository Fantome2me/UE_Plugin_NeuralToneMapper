// Copyright (c) 2026. All Rights Reserved.

#include "NeuralToneMapperEditorModule.h"

#include "NeuralToneMapperModule.h"
#include "NTMOSettings.h"

#include "Framework/Notifications/NotificationManager.h"
#include "Misc/MessageDialog.h"
#include "ToolMenu.h"
#include "ToolMenuEntry.h"
#include "ToolMenuSection.h"
#include "ToolMenus.h"
#include "Widgets/Notifications/SNotificationList.h"

#define LOCTEXT_NAMESPACE "FNeuralToneMapperEditorModule"

static const FName GNTMOBuildMenuName = TEXT("LevelEditor.MainMenu.Build");
static const FName GNTMOMenuSection = TEXT("NTMO");
static const FName GNTMOEntryName = TEXT("NTMO_EnableAIToneMapping");

void FNeuralToneMapperEditorModule::StartupModule()
{
	// Defer until ToolMenus and the level editor are ready.
	UToolMenus::Get()->RegisterStartupCallback(
		FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FNeuralToneMapperEditorModule::RegisterMenus));
}

void FNeuralToneMapperEditorModule::ShutdownModule()
{
	if (UToolMenus::IsToolMenuUIEnabled())
	{
		UToolMenus::Get()->UnregisterOwner(this);
	}
}

void FNeuralToneMapperEditorModule::RegisterMenus()
{
	FToolMenuOwnerScoped OwnerScoped(this);

	UToolMenu* Menu = UToolMenus::Get()->ExtendMenu(GNTMOBuildMenuName);
	if (!Menu)
	{
		// Some configurations (e.g. a bare commandlet) have no level editor menu.
		return;
	}

	FToolMenuSection& Section = Menu->FindOrAddSection(
		GNTMOMenuSection,
		LOCTEXT("NTMOSection", "AI Tone Mapping"),
		FToolMenuInsert(NAME_None, EToolMenuInsertType::Last));

	// FCanExecuteAction keeps the entry greyscale until the renderer is up.
	FUIAction Action(
		FExecuteAction::CreateStatic(&FNeuralToneMapperEditorModule::ToggleAI),
		FCanExecuteAction::CreateStatic(&FNeuralToneMapperEditorModule::IsAIMenuEnabled),
		FGetActionCheckState::CreateStatic(&FNeuralToneMapperEditorModule::GetAIState));

	FToolMenuEntry Entry = FToolMenuEntry::InitMenuEntry(
		GNTMOEntryName,
		TAttribute<FText>::CreateStatic(&FNeuralToneMapperEditorModule::GetLabel),
		TAttribute<FText>::CreateStatic(&FNeuralToneMapperEditorModule::GetToolTip),
		FSlateIcon(),
		Action,
		EUserInterfaceActionType::Check);

	Entry.InsertPosition = FToolMenuInsert(NAME_None, EToolMenuInsertType::Default);

	Section.AddEntry(Entry);

	UE_LOG(LogTemp, Log, TEXT("[NTMO] Added \"AI动态色调映射\" to the Build menu."));
}

void FNeuralToneMapperEditorModule::ToggleAI()
{
	const bool bNewState = !FNeuralToneMapperModule::IsEnabled();
	bNewState ? EnableAI() : DisableAI();
}

void FNeuralToneMapperEditorModule::EnableAI()
{
	FNeuralToneMapperModule::SetEnabled(true);

	FNotificationInfo Info(LOCTEXT("NTMOEnabled", "AI 动态色调映射：已启用（Unreal 自带色调映射已关闭）"));
	Info.ExpireDuration = 3.0f;
	Info.bUseSuccessFailIcons = true;
	FSlateNotificationManager::Get().AddNotification(Info);
}

void FNeuralToneMapperEditorModule::DisableAI()
{
	FNeuralToneMapperModule::SetEnabled(false);

	FNotificationInfo Info(LOCTEXT("NTMODisabled", "AI 动态色调映射：已关闭（已恢复 Unreal 自带色调映射）"));
	Info.ExpireDuration = 3.0f;
	FSlateNotificationManager::Get().AddNotification(Info);
}

ECheckBoxState FNeuralToneMapperEditorModule::GetAIState()
{
	return FNeuralToneMapperModule::IsEnabled() ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
}

bool FNeuralToneMapperEditorModule::IsAIMenuEnabled()
{
	// Available in editor and PIE alike; the renderer decides at runtime.
	return true;
}

FText FNeuralToneMapperEditorModule::GetLabel()
{
	return LOCTEXT("NTMOLabel", "AI动态色调映射");
}

FText FNeuralToneMapperEditorModule::GetToolTip()
{
	return LOCTEXT("NTMOToolTip",
		"用插件的 HDR 色调映射替换 Unreal 自带的色调映射。\n"
		"开启后会跳过引擎的 tonemap pass 与 CombineLUTs，改用按显示器能力自适应的曲线 + 33^3 LUT。\n"
		"状态保存在项目配置中，并同步到 cvar r.NTMO.Enable。");
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FNeuralToneMapperEditorModule, NeuralToneMapperEditor)
