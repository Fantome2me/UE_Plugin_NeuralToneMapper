// Copyright (c) 2026. All Rights Reserved.

using UnrealBuildTool;

public class NeuralToneMapper : ModuleRules
{
	public NeuralToneMapper(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"DeveloperSettings",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Projects",
			"RHI",
			"RenderCore",
			"Renderer",
			// Slate is needed to resolve the game viewport's native window handle so
			// HDRGetMetaDataForWindow can query real display capabilities.
			"Slate",
			"SlateCore",
			// NNE exposes UE::NNE::INNERuntime / INNERuntimeRDG / UNNEModelData from the
			// engine's own Runtime module. The concrete backends (NNERuntimeRDG,
			// NNERuntimeORT) are enabled as plugins and resolved at runtime, so this
			// module has no hard build dependency on them.
			"NNE",
		});
	}
}
