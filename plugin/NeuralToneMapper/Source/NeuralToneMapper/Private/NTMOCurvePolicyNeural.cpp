// Copyright (c) 2026. All Rights Reserved.

#include "NTMOCurvePolicyNeural.h"

#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "NNE.h"
#include "NNERuntime.h"
#include "UObject/Package.h"

namespace
{
	/** Human readable tensor desc, for the one-time log. */
	FString DescribeTensors(TConstArrayView<UE::NNE::FTensorDesc> Descs)
	{
		FString Out;
		for (const UE::NNE::FTensorDesc& D : Descs)
		{
			const UE::NNE::FSymbolicTensorShape& Shape = D.GetShape();
			const TConstArrayView<int32> Dims = Shape.GetData();
			Out += TEXT("[");
			for (int32 i = 0; i < Dims.Num(); ++i)
			{
				if (i > 0) { Out += TEXT(","); }
				// Symbolic (variable) dimensions are negative.
				Out += (Dims[i] < 0) ? FString(TEXT("N")) : FString::FromInt(Dims[i]);
			}
			Out += TEXT("] ");
		}
		return Out;
	}
}

FNTMOCurvePolicyNeural::FNTMOCurvePolicyNeural() = default;
FNTMOCurvePolicyNeural::~FNTMOCurvePolicyNeural() = default;

void FNTMOCurvePolicyNeural::SetModelPath(const FSoftObjectPath& InPath)
{
	if (ModelPath == InPath)
	{
		return;
	}
	ModelPath = InPath;
	bProbed = false;
	bReady = false;
	ModelInstance.Reset();
	ModelData.Reset();
	StatusText.Reset();
}

void FNTMOCurvePolicyNeural::SetModelFilePath(const FString& InPath)
{
	if (ModelFilePath == InPath)
	{
		return;
	}
	ModelFilePath = InPath;
	bProbed = false;
	bReady = false;
	ModelInstance.Reset();
	ModelData.Reset();
	StatusText.Reset();
}

void FNTMOCurvePolicyNeural::SetRuntimeName(const FString& InRuntimeName)
{
	if (RuntimeName == InRuntimeName)
	{
		return;
	}
	RuntimeName = InRuntimeName;
	bProbed = false;
	bReady = false;
	ModelInstance.Reset();
	ModelData.Reset();
	StatusText.Reset();
}

FString FNTMOCurvePolicyNeural::FindShippedModel()
{
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("NeuralToneMapper"));
	if (!Plugin.IsValid())
	{
		return FString();
	}

	const FString ModelDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Content"), TEXT("Models"));
	TArray<FString> Candidates;
	IFileManager::Get().FindFiles(Candidates, *(ModelDir / TEXT("NTMO_M_v*.onnx")), /*Files*/ true, /*Dirs*/ false);

	// Also accept the unversioned NTMO_M.onnx (version 0) so an early build still works.
	if (IFileManager::Get().FileExists(*(ModelDir / TEXT("NTMO_M.onnx"))))
	{
		Candidates.Add(TEXT("NTMO_M.onnx"));
	}

	if (Candidates.Num() == 0)
	{
		return FString();
	}

	// Pick the highest version: NTMO_M_v6.onnx beats NTMO_M_v5.onnx, and both beat
	// NTMO_M.onnx. Filename order is not enough (v10 < v9 lexically), so parse the int.
	auto VersionOf = [](const FString& Name) -> int32
	{
		int32 Best = 0;
		int32 Pos = 0;
		while ((Pos = Name.Find(TEXT("_v"), ESearchCase::IgnoreCase, ESearchDir::FromStart, Pos)) != INDEX_NONE)
		{
			Pos += 2;
			int32 Value = 0;
			int32 Digits = 0;
			while (Pos < Name.Len() && FChar::IsDigit(Name[Pos]))
			{
				Value = Value * 10 + (Name[Pos] - TEXT('0'));
				++Pos;
				++Digits;
			}
			if (Digits > 0)
			{
				Best = FMath::Max(Best, Value);
			}
		}
		return Best;
	};

	FString BestName;
	int32 BestVersion = -1;
	for (const FString& Name : Candidates)
	{
		const int32 Version = VersionOf(Name);
		if (Version > BestVersion)
		{
			BestVersion = Version;
			BestName = Name;
		}
	}

	return FPaths::ConvertRelativePathToFull(ModelDir / BestName);
}

void FNTMOCurvePolicyNeural::RefreshAvailability()
{
	bProbed = true;
	bReady = false;

	// ---- resolve the model path -------------------------------------------
	// The setting is an absolute path, which is fine for the machine that trained the
	// model but useless once the plugin folder is copied elsewhere. When the configured
	// path is empty or missing, fall back to the newest model shipped inside the plugin
	// (<Plugin>/Content/Models/NTMO_M_v*.onnx). That is what makes the folder portable:
	// drop it into Plugins/ and the AI works without touching any setting.
	FString ResolvedModelFile = ModelFilePath;
	if (ResolvedModelFile.IsEmpty() || !FPaths::FileExists(ResolvedModelFile))
	{
		const FString ShippedModel = FindShippedModel();
		if (!ShippedModel.IsEmpty())
		{
			if (!ResolvedModelFile.IsEmpty())
			{
				UE_LOG(LogTemp, Warning,
					TEXT("[NTMO] Curve model '%s' not found; using the model shipped with the ")
					TEXT("plugin instead: %s"), *ResolvedModelFile, *ShippedModel);
			}
			ResolvedModelFile = ShippedModel;
		}
	}

	if (ResolvedModelFile.IsEmpty() && ModelPath.IsNull())
	{
		StatusText = FString::Printf(TEXT("Neural: no model set (runtime '%s')"), *RuntimeName);
		if (!bLoggedMissingModel)
		{
			bLoggedMissingModel = true;
			UE_LOG(LogTemp, Log,
				TEXT("[NTMO] CurveSource = Neural, but no model is configured. Set the .onnx path in ")
				TEXT("Project Settings > Plugins > Neural Tone Mapper > Curve Model File. ")
				TEXT("Falling back to the analytic policy."));
		}
		return;
	}

	// ---- 1. runtime ------------------------------------------------------
	// GetRuntime returns a TWeakInterfacePtr, and INNERuntimeRDG lives at global scope,
	// not inside namespace UE::NNE.
	const TWeakInterfacePtr<INNERuntimeRDG> RuntimeWeak =
		UE::NNE::GetRuntime<INNERuntimeRDG>(RuntimeName);
	INNERuntimeRDG* Runtime = RuntimeWeak.Get();

	if (Runtime == nullptr)
	{
		StatusText = FString::Printf(TEXT("Neural: runtime '%s' not available"), *RuntimeName);
		if (!bLoggedMissingRuntime)
		{
			bLoggedMissingRuntime = true;
			UE_LOG(LogTemp, Warning,
				TEXT("[NTMO] NNE runtime '%s' is not available. Enable the NNERuntimeRDG plugin ")
				TEXT("(and its HLSL backend) then restart. Falling back to the analytic policy."),
				*RuntimeName);
		}
		return;
	}

	// ---- 2. model data ---------------------------------------------------
	if (!ModelData.IsValid())
	{
		if (!ResolvedModelFile.IsEmpty())
		{
			TArray<uint8> Bytes;
			if (!FFileHelper::LoadFileToArray(Bytes, *ResolvedModelFile))
			{
				StatusText = FString::Printf(TEXT("Neural: cannot read '%s'"), *ResolvedModelFile);
				UE_LOG(LogTemp, Error, TEXT("[NTMO] Cannot read curve model '%s'."), *ResolvedModelFile);
				return;
			}

			// Model data is built from the raw .onnx, so there is no asset to import.
			UNNEModelData* NewModelData = NewObject<UNNEModelData>(GetTransientPackage());
			NewModelData->Init(TEXT("onnx"), Bytes);
			ModelData.Reset(NewModelData);

			UE_LOG(LogTemp, Log, TEXT("[NTMO] Curve model loaded from '%s' (%.1f KB, file type onnx)."),
				*ResolvedModelFile, Bytes.Num() / 1024.0f);
		}
		else
		{
			UObject* Loaded = ModelPath.TryLoad();
			UNNEModelData* Casted = Cast<UNNEModelData>(Loaded);
			if (Casted == nullptr)
			{
				StatusText = FString::Printf(TEXT("Neural: '%s' is not a UNNEModelData asset"),
					*ModelPath.ToString());
				UE_LOG(LogTemp, Error, TEXT("[NTMO] '%s' did not resolve to a UNNEModelData."),
					*ModelPath.ToString());
				return;
			}
			ModelData.Reset(Casted);
		}
	}

	// ---- 3. model + instance --------------------------------------------
	TSharedPtr<UE::NNE::IModelRDG> Model = Runtime->CreateModelRDG(ModelData.Get());
	if (!Model.IsValid())
	{
		StatusText = FString::Printf(TEXT("Neural: CreateModelRDG failed for runtime '%s'"), *RuntimeName);
		UE_LOG(LogTemp, Error,
			TEXT("[NTMO] NNE runtime '%s' could not create an RDG model from this .onnx. ")
			TEXT("Most likely it uses an operator the HLSL backend does not implement -- ")
			TEXT("re-run export_onnx.py and check DEPLOYABLE."), *RuntimeName);
		return;
	}

	ModelInstance = Model->CreateModelInstanceRDG();
	if (!ModelInstance.IsValid())
	{
		StatusText = TEXT("Neural: CreateModelInstanceRDG failed");
		UE_LOG(LogTemp, Error, TEXT("[NTMO] CreateModelInstanceRDG() returned null."));
		return;
	}

	// ---- 4. shapes -------------------------------------------------------
	// FTensorShape::Make takes uint32 dimensions.
	const TArray<uint32> Dims = {
		1u, (uint32)NTMOInput::Channels, (uint32)NTMOInput::Height, (uint32)NTMOInput::Width };
	const TArray<UE::NNE::FTensorShape> Shapes = { UE::NNE::FTensorShape::Make(Dims) };

	const UE::NNE::IModelInstanceRDG::ESetInputTensorShapesStatus Status =
		ModelInstance->SetInputTensorShapes(Shapes);

	if (Status != UE::NNE::EResultStatus::Ok)
	{
		StatusText = FString::Printf(TEXT("Neural: SetInputTensorShapes failed (status %d)"),
			(int32)Status);
		UE_LOG(LogTemp, Error,
			TEXT("[NTMO] SetInputTensorShapes([1,%d,%d,%d]) failed with status %d. ")
			TEXT("The model's input does not match the training contract."),
			NTMOInput::Channels, NTMOInput::Height, NTMOInput::Width, (int32)Status);
		ModelInstance.Reset();
		return;
	}

	bReady = true;
	StatusText = FString::Printf(TEXT("Neural: ready (runtime '%s')"), *RuntimeName);

	UE_LOG(LogTemp, Log, TEXT("[NTMO] NNE model ready. runtime='%s'"), *RuntimeName);
	UE_LOG(LogTemp, Log, TEXT("[NTMO]   model inputs : %s"), *DescribeTensors(ModelInstance->GetInputTensorDescs()));
	UE_LOG(LogTemp, Log, TEXT("[NTMO]   model outputs: %s"), *DescribeTensors(ModelInstance->GetOutputTensorDescs()));
}

bool FNTMOCurvePolicyNeural::ComputeCurve(const FNTMOContext& Context, FNTMOCurveParams& OutParams)
{
	if (!bProbed)
	{
		RefreshAvailability();
	}

	// Deliberately false even when ready: the curve is produced on the GPU into a
	// buffer that the LUT bake reads. Returning true here would mean filling
	// OutParams on the CPU, which would silently bypass the network.
	return false;
}

FString FNTMOCurvePolicyNeural::GetStatusString() const
{
	return StatusText.IsEmpty() ? TEXT("Neural: idle") : StatusText;
}
