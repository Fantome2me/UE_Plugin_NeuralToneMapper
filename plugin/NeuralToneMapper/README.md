# Neural Tone Mapper 1.0.1 鈥?AI 鍔ㄦ€佽壊璋冩槧灏?(UE 5.8, Win64)

鐢ㄤ竴涓?315K 鍙傛暟鐨勭缁忕綉缁?*姣忓抚棰勬祴鑹茶皟鏇茬嚎**锛屾浛鎹?UE 鑷甫 tone mapping銆?鏇茬嚎浼氭牴鎹?*鐢婚潰鍐呭**鍜?*鏄剧ず鍣ㄧ湡瀹炶兘鍔?*锛堝嘲鍊间寒搴?/ ABL 闄愬埗锛夎嚜閫傚簲锛屾敮鎸?SDR銆丷ec.709銆丳Q (ST.2084)銆乻cRGB 杈撳嚭銆?
## 瀹夎

1. 鎶婃暣涓?`NeuralToneMapper` 鏂囦欢澶规嫹鍒颁綘鐨勫伐绋嬶細

       <浣犵殑宸ョ▼>/Plugins/NeuralToneMapper/

   鎴栨嫹鍒板紩鎿庯紙褰卞搷鎵€鏈夊伐绋嬶級锛?
       <UE_5.8>/Engine/Plugins/NeuralToneMapper/

2. 鍚姩缂栬緫鍣ㄣ€傞娆″姞杞戒細鎻愮ず鍚敤鎻掍欢锛岄€夈€屾槸銆嶏紱鎴栨墜鍔ㄥ湪
   **Edit 鈫?Plugins** 閲屽嬀閫?**Neural Tone Mapper**锛岄噸鍚紪杈戝櫒銆?
3. 纭寮曟搸鎻掍欢 **NNERuntimeRDG** 宸插惎鐢?   锛圗dit 鈫?Plugins 鎼?"NNE"锛夈€傚畠鏄?UE 鑷甫瀹為獙鎬ф彃浠讹紝鏈彃浠朵緷璧栧畠鍋?GPU 鎺ㄧ悊銆?   鏈惎鐢ㄦ椂鎻掍欢浠嶅彲宸ヤ綔锛屼絾浼氶€€鍥炶В鏋愭洸绾匡紝AI 涓嶅弬涓庛€?
## 浣跨敤

- 鑿滃崟锛?*鏋勫缓 鈫?AI鍔ㄦ€佽壊璋冩槧灏?*锛堝嬀閫?= 鍚敤锛?- 鎺у埗鍙帮細

      r.NTMO.Enable 1        鍚敤
      r.NTMO.Enable 0        鍏抽棴锛屽洖鍒?UE 鍘熺敓 tonemapper
      NTMO.Toggle            鍒囨崲
      r.NTMO.DumpKnots 1     鎵撳嵃缃戠粶瀹為檯杈撳嚭鐨?8 涓洸绾?knot锛堥獙璇佺敤锛?
- 宸ョ▼璁剧疆锛?*Project Settings 鈫?Plugins 鈫?Neural Tone Mapper**
  鍙皟 Intensity / Saturation / 楂樺厜鍘婚ケ鍜?/ 灞€閮ㄧ畻瀛愬己搴?/ 鏇茬嚎骞虫粦 / 鎺ㄧ悊闂撮殧绛夈€?
## 鍑哄巶榛樿

寮€绠卞嵆鐢細`bEnabled=True`銆乣CurveSource=Neural`锛屾ā鍨嬭嚜鍔ㄤ粠
`Content/Models/NTMO_M_v*.onnx` 閲屾寫鏈€鏂扮増鏈紝**涓嶉渶瑕侀厤缃换浣曡矾寰?*銆?
## 鐗堟湰瑕佹眰

- Unreal Engine **5.8**锛圵in64锛?- 鍥惧舰 API锛欴3D12锛堥粯璁わ級
- 闄勫甫涓や釜妯″潡鐨?Win64 Development DLL銆傝嫢浣犵殑寮曟搸 **鏋勫缓鐗堟湰鍙?*涓庢湰鍖呬笉涓€鑷达紝
  UE 浼氭彁绀洪噸鏂扮紪璇戯紱姝ゆ椂闇€瑕?Visual Studio 2019/2022 + 寮曟搸婧愮爜/瀹夎鐗堟瀯寤虹幆澧冿紝
  婧愮爜宸插湪 `Source/` 閲岋紝鍙抽敭 .uproject 鐢熸垚宸ョ▼鏂囦欢鍚庣紪璇戝嵆鍙€?
## 宸查獙璇佹寚鏍?
- 鍚屾満浣?A/B 瀵规瘮 UE 鍘熺敓 tonemapper锛氶€愬儚绱犵浉鍏?**0.974**锛屽姣斿害姣?**0.956**锛?  闃村奖/涓皟鍋忓樊 **+0.010 / +0.005 妗?*锛屾棤鍓婇《
- 鎻掍欢寮€閿€锛?*绾?-0.04 ms**锛堥粯璁ゆ帹鐞嗛棿闅?4锛涢棿闅?1 鏃?+0.81 ms锛?- HDR 鑷€傚簲锛歋DR 100nit 鈫?宄板€?100nit锛汬DR 1000nit/ABL0.4 鈫?宄板€?鈮?30nit

## 宸茬煡闄愬埗

- 灞€閮ㄧ畻瀛愭浛鎹紙鑰岄潪鍙犲姞锛塙E 鐨?Local Exposure
- 鏈疄鐜?bloom 涔嬪鐨勫畬鏁村悗澶勭悊绛変环锛堟殫瑙掓寜 1 澶勭悊锛?- 鏇茬嚎姣?N 甯ф洿鏂颁竴娆★紙瑙?Inference Interval锛夛紝骞跺缁撴灉鍋?EMA 骞虫粦

## 鎺掓煡

| 鐜拌薄 | 澶勭悊 |
|---|---|
| 鑿滃崟閲屾病鏈夎椤?| 鎻掍欢鏈惎鐢紝鎴?`NeuralToneMapperEditor.dll` 鏈姞杞斤紙鐪嬫棩蹇?`[NTMO] Added "AI鍔ㄦ€佽壊璋冩槧灏?`锛?|
| 鐢婚潰娌″彉鍖?| `r.NTMO.Enable` 鏄惁涓?1锛沗r.NTMO.LogLiveness 1` 纭姣忓抚鍦ㄨ窇 |
| AI 娌＄敓鏁堬紙璧拌В鏋愭洸绾匡級 | 鏃ュ織鎼?`NNE model ready`锛涙病鏈夊垯妫€鏌?NNERuntimeRDG 鏄惁鍚敤 |
| 杩囨洕/鍏ㄩ粦 | 鐪嬫棩蹇?`[NTMO] Curve in use` 鐨勬樉绀哄櫒涓婁笅鏂囨槸鍚﹀悎鐞?|
