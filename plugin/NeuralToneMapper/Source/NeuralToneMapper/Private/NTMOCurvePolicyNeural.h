// Copyright (c) 2026. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NTMOPolicy.h"

#include "NNEModelData.h"
#include "NNERuntimeRDG.h"

class UNNEModelData;

/**
 * Contract between the training pipeline (ntmo_pack.py) and the renderer.
 * These MUST match: gen_dataset.py writes 48 channels and export_onnx.py bakes a
 * fixed 1x48x72x128 input. Changing either side without the other silently feeds
 * the network garbage that still produces plausible-looking knots.
 */
namespace NTMOInput
{
	// Contract v3 (NTM 1.0.3). Must match ntmo_pack.py: C_IN / C_OUT / C_IMG / LOCAL_N.
	static constexpr int32 Channels = 49;	// 4 image (PQ rgb + edge) + 13 display/scene + 16 tiles + 16 histogram
	static constexpr int32 Height   = 72;
	static constexpr int32 Width    = 128;
	static constexpr int32 Knots    = 8;	// curve knots: output[0..7]
	static constexpr int32 LocalTiles = 4;
	static constexpr int32 LocalBands = 4;	// s_base, s_mid, s_high, s_low (exposure)
	static constexpr int32 LocalParams = LocalTiles * LocalTiles * LocalBands;	// 64
	static constexpr int32 Outputs = Knots + LocalParams;						// 72
}

/**
 * Neural curve predictor.
 *
 * The model is created at RUNTIME from the raw .onnx file -- no content-browser
 * import, no UNNEModelData asset to keep in sync. That works because
 * UNNEModelData::Init(Type, Buffer, ...) is NNE_API and not editor-only
 * (Runtime/NNE/Public/NNEModelData.h:120), so NewObject + Init is enough.
 *
 * Backend: NNERuntimeRDGHlsl, which emits real RDG compute dispatches. We never
 * touch NeuralPostProcessing's runtime pass because it calls
 * SubmitAndBlockUntilGPUIdle every frame (NeuralPostProcessModelInstance.cpp:87).
 *
 * ComputeCurve() intentionally still returns false: the curve now comes from the
 * inference buffer on the GPU, not from a CPU-side struct. The renderer asks
 * IsReady() and binds GetModelInstance() into its own RDG passes.
 */
class FNTMOCurvePolicyNeural : public INTMOCurveProvider
{
public:
	FNTMOCurvePolicyNeural();
	virtual ~FNTMOCurvePolicyNeural() override;

	//~ Begin INTMOCurveProvider
	virtual ENTMOCurveProviderKind GetKind() const override { return ENTMOCurveProviderKind::Neural; }
	virtual bool ComputeCurve(const FNTMOContext& Context, FNTMOCurveParams& OutParams) override;
	virtual FString GetStatusString() const override;
	//~ End INTMOCurveProvider

	/** Soft path to an imported UNNEModelData asset (optional; .onnx path is preferred). */
	void SetModelPath(const FSoftObjectPath& InPath);

	/** Absolute path to the trained .onnx on disk. Game thread. */
	void SetModelFilePath(const FString& InPath);

	/** Name of the NNE runtime to use. NNERuntimeRDGHlsl is the zero-sync one. */
	void SetRuntimeName(const FString& InRuntimeName);

	/** True when a runtime accepted the model and an instance exists. */
	bool IsReady() const { return bReady; }

	/**
	 * Runs the one-time probe if it has not happened yet. MUST be called from the
	 * game thread (it loads a file and creates UObjects).
	 *
	 * This exists because SelectProvider() only ever asks IsReady(), and the probe
	 * used to live in ComputeCurve() -- which is never reached for the neural policy
	 * while IsReady() is still false. That deadlock meant the model was never loaded.
	 */
	void EnsureProbed() { if (!bProbed) { RefreshAvailability(); } }

	/** Render-thread usable handle. Null until IsReady(). */
	TSharedPtr<UE::NNE::IModelInstanceRDG> GetModelInstance() const { return ModelInstance; }

private:
	/** Reads the .onnx, builds UNNEModelData, creates the RDG model + instance. */
	void RefreshAvailability();

	/**
	 * Newest model shipped inside the plugin at <Plugin>/Content/Models/NTMO_M_v*.onnx.
	 * Returns an empty string when none is found.
	 */
	static FString FindShippedModel();

	FString RuntimeName = TEXT("NNERuntimeRDGHlsl");
	FSoftObjectPath ModelPath;
	FString ModelFilePath;

	/** Keeps the transient UNNEModelData alive against GC. */
	TStrongObjectPtr<UNNEModelData> ModelData;

	/** Shared to the render thread; IModelInstanceRDG::EnqueueRDG is render-thread only. */
	TSharedPtr<UE::NNE::IModelInstanceRDG> ModelInstance;

	bool bReady = false;
	bool bProbed = false;
	bool bLoggedMissingRuntime = false;
	bool bLoggedMissingModel = false;
	FString StatusText = TEXT("not initialised");
};
