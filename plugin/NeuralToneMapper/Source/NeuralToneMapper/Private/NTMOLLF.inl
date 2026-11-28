// Included by the renderer translation unit after the RDG and screen-pass headers.
static TAutoConsoleVariable<int32> CVarNTMOLLF(TEXT("r.NTMO.LLF"),1,TEXT("Enable trained spatial LUT and Laplacian reconstruction."),ECVF_RenderThreadSafe);
class FNTMOLLFGPU : public FGlobalShader
{
public:
 DECLARE_GLOBAL_SHADER(FNTMOLLFGPU);
 SHADER_USE_PARAMETER_STRUCT(FNTMOLLFGPU,FGlobalShader);
 class FStage : SHADER_PERMUTATION_INT("STAGE",6);
 using FPermutationDomain=TShaderPermutationDomain<FStage>;
 BEGIN_SHADER_PARAMETER_STRUCT(FParameters,)
 SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters,View)
 SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>,Weights)
 SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>,Eye)
 SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>,Feat)
 SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>,KV)
 SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>,VecOut)
 SHADER_PARAMETER_RDG_TEXTURE(Texture2D,A)
 SHADER_PARAMETER_RDG_TEXTURE(Texture2D,B)
 SHADER_PARAMETER_RDG_TEXTURE(Texture2D,C)
 SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>,ImageOut)
 SHADER_PARAMETER_SAMPLER(SamplerState,Bi)
 SHADER_PARAMETER(FIntPoint,Size)
 SHADER_PARAMETER(FIntPoint,ASize)
 SHADER_PARAMETER(FVector4f,SceneRect)
 SHADER_PARAMETER(FVector4f,BloomRect)
 SHADER_PARAMETER(FVector3f,Tint)
 SHADER_PARAMETER(float,BloomAmount)
 END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FNTMOLLFGPU,"/NTMO/Private/NTMOLLF.usf","Main",SF_Compute);
static FRDGTextureRef AddNTMOLLF(FRDGBuilder& G,const FSceneView& V,FScreenPassTexture Scene,FScreenPassTexture Bloom,FRDGBufferRef Eye,float BloomScale)
{
 static const float Data[]={
#include "NTMOLLFWeights.inl"
 };
 static_assert(UE_ARRAY_COUNT(Data)==46598);
 static TRefCountPtr<FRDGPooledBuffer> Cache;
 FRDGBufferRef W=Cache.IsValid()?G.RegisterExternalBuffer(Cache):CreateStructuredBuffer(G,TEXT("NTMO.LLF.Weights"),sizeof(float),UE_ARRAY_COUNT(Data),Data,sizeof(Data),ERDGInitialDataFlags::NoCopy);
 if(!Cache.IsValid()) Cache=G.ConvertToExternalBuffer(W);
 auto Rect=[](FScreenPassTexture T){const FIntPoint E=T.Texture->Desc.Extent;return FVector4f(float(T.ViewRect.Min.X)/E.X,float(T.ViewRect.Min.Y)/E.Y,float(T.ViewRect.Width())/E.X,float(T.ViewRect.Height())/E.Y);};
 auto Tex=[&](FIntPoint S){return G.CreateTexture(FRDGTextureDesc::Create2D(S,PF_FloatRGBA,FClearValueBinding::None,TexCreate_ShaderResource|TexCreate_UAV),TEXT("NTMO.LLF.Image"));};
 auto Buf=[&](int N){return G.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(4,N),TEXT("NTMO.LLF.Vector"));};
 FRDGBufferRef F=nullptr,K=nullptr;
 auto Pass=[&](int Stage,FIntPoint S,FRDGTextureRef A,FRDGTextureRef B,FRDGTextureRef C,FRDGTextureRef Out,FRDGBufferRef VO){
 auto* P=G.AllocParameters<FNTMOLLFGPU::FParameters>();P->View=V.ViewUniformBuffer;P->Size=S;P->ASize=A->Desc.Extent;P->A=A;P->B=B;P->C=C;P->Bi=TStaticSamplerState<SF_Bilinear,AM_Clamp,AM_Clamp,AM_Clamp>::GetRHI();
 P->Weights=G.CreateSRV(W);P->Eye=G.CreateSRV(Eye);P->SceneRect=Rect(Scene);P->BloomRect=Rect(Bloom.IsValid()?Bloom:Scene);P->BloomAmount=Bloom.IsValid()?BloomScale:0;
 const auto T=V.FinalPostProcessSettings.SceneColorTint;P->Tint=FVector3f(T.R,T.G,T.B);
 if(Stage==3||Stage==4)P->Feat=G.CreateSRV(F);if(Stage==4)P->KV=G.CreateSRV(K);
 if(VO)P->VecOut=G.CreateUAV(VO);if(Out)P->ImageOut=G.CreateUAV(Out);
 FNTMOLLFGPU::FPermutationDomain Perm;Perm.Set<FNTMOLLFGPU::FStage>(Stage);TShaderMapRef<FNTMOLLFGPU> Sh(GetGlobalShaderMap(V.GetFeatureLevel()),Perm);
 FComputeShaderUtils::AddPass(G,RDG_EVENT_NAME("NTMO::LLF %d %dx%d",Stage,S.X,S.Y),Sh,P,FComputeShaderUtils::GetGroupCount(S,8));};
 FIntPoint S=Scene.ViewRect.Size();TArray<FRDGTextureRef,TInlineAllocator<10>> Levels;Levels.Add(Tex(S));
 Pass(0,S,Scene.Texture,Bloom.IsValid()?Bloom.Texture:Scene.Texture,nullptr,Levels.Last(),nullptr);
 while(S.GetMax()>64){auto Prev=Levels.Last();S=FIntPoint::DivideAndRoundUp(S,2);Levels.Add(Tex(S));Pass(1,S,Prev,nullptr,nullptr,Levels.Last(),nullptr);}
 auto Low=Levels.Last();F=Buf(S.X*S.Y*16);K=Buf(64*32);Pass(2,S,Low,nullptr,nullptr,nullptr,F);Pass(3,FIntPoint(8,8),Low,nullptr,nullptr,nullptr,K);
 auto Y=Tex(S);Pass(4,S,Low,nullptr,nullptr,Y,nullptr);
 for(int i=Levels.Num()-2;i>=0;i--){S=Levels[i]->Desc.Extent;auto Next=Tex(S);Pass(5,S,Levels[i],Levels[i+1],Y,Next,nullptr);Y=Next;}
 return Y;
}
