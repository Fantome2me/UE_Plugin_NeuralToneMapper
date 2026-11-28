"""Export little-endian float32 weights to a build-time array and shared shader offsets."""
from pathlib import Path
import sys
import torch
root=Path(__file__).resolve().parent
sd=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
offsets=[]; values=[]
for name,t in sd.items():
    offsets.append('#define W_'+name.replace('.','_')+' '+str(len(values))+'u')
    values.extend(t.contiguous().flatten().tolist())
(root/'Shaders/Private/NTMOLLFOffsets.ush').write_text('\n'.join(offsets)+'\n')
fmt=lambda x: format(x,'.9e')+'f'
(root/'Source/NeuralToneMapper/Private/NTMOLLFWeights.inl').write_text('\n'.join(','.join(map(fmt,values[i:i+12]))+',' for i in range(0,len(values),12)))
