// Copyright (c) 2026. All Rights Reserved.

using UnrealBuildTool;

public class NeuralToneMapperEditor : ModuleRules
{
	public NeuralToneMapperEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"NeuralToneMapper",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Slate",
			"SlateCore",
			"EditorFramework",
			"UnrealEd",
			"ToolMenus",
			"LevelEditor",
			"Projects",
			"DeveloperSettings",
		});
	}
}
