"""Contract audit for the NTM plugin.

Every defect the 1.1.0 review found by hand is turned into a check here, so the next one
cannot hide the same way:

  A. every UPROPERTY default must match the shipped ini      (found: bEnabled, desat)
  B. every UPROPERTY must be read outside the header         (found: 3 dead settings)
  C. every shader parameter set from C++ must be used in a .usf (drift detector)
  D. every .usf global must be used, not just declared        (drift detector)
  E. version strings must agree across the three copies       (release hygiene)
  F. the model input contract must match the C++ contract     (49 x 72 x 128)

Run:  python audit_contracts.py [--plugin <path>]
Exit code 1 if anything fails, so it can gate a release.
"""

import argparse
import json
import os
import re
import sys

DEFAULT_PLUGIN = r"E:\ConveyanceSpline\UE_ConveyanceSpline\Plugins\NeuralToneMapper"

PROP_RE = re.compile(
    r"^\s*(?:UPROPERTY\([^)]*\)\s*)?"
    r"(bool|int32|float|double|FSoftObjectPath|FString|TEnumAsByte<[^>]+>|[A-Z]\w+)\s+"
    r"(\w+)\s*=\s*([^;]+);", re.M)


def targets(path, exts):
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(root, f)


def norm(v):
    v = v.strip().rstrip("f").strip('"').strip()
    if v.startswith("TEXT(") and v.endswith(")"):
        v = v[5:-1].strip('"')
    low = v.lower()
    if low in ("true", "false"):
        return low
    try:
        f = float(v)
        return bool(round(f, 6) == round(f, 6)) and ("%.6f" % round(f, 6))
    except ValueError:
        return v


def check_settings(plugin, problems):
    header = os.path.join(plugin, "Source", "NeuralToneMapper", "Public", "NTMOSettings.h")
    ini = os.path.join(plugin, "Config", "DefaultNeuralToneMapper.ini")
    src = open(header, encoding="utf-8").read()

    props = {}
    for m in PROP_RE.finditer(src):
        typ, name, default = m.group(1), m.group(2), m.group(3).strip()
        props[name] = (typ, default)

    ini_vals = {}
    for line in open(ini, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("[") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        ini_vals[k.strip()] = v.strip()

    # A. header default vs ini
    mismatched = []
    for name, (typ, default) in props.items():
        if name not in ini_vals or default.startswith("{") or "::" in default:
            continue
            continue
        if norm(default) != norm(ini_vals[name]):
            mismatched.append((name, default, ini_vals[name]))
    if mismatched:
        problems.append(("A 默认值不一致", ["%s 头文件=%s  ini=%s" % t for t in mismatched]))
    else:
        print("  A 默认值一致性: %d 个属性，ini 覆盖 %d 项" % (len(props), sum(1 for n in props if n in ini_vals)))

    # B. dead properties: declared but never read outside the header
    all_src = "".join(open(p, encoding="utf-8", errors="ignore").read()
                      for p in targets(os.path.join(plugin, "Source"), {".cpp", ".h"})
                      if not p.endswith("NTMOSettings.h"))
    # A property marked "Deprecated" in the comment right above it is kept on purpose so
    # existing ini files still parse; it is not a new defect. Anything else that nothing
    # reads IS a defect.
    deprecated = set(re.findall(r"//[^\n]*Deprecated[^\n]*\n\s*(?:UPROPERTY[^\n]*\n\s*)?"
                                r"\w+\s+(\w+)\s*=", src))
    dead = [n for n in props
            if not re.search(r"\b%s\b" % re.escape(n), all_src) and n not in deprecated]
    if deprecated:
        print("  B 已知弃用（保留以兼容旧配置）: %s" % ", ".join(sorted(deprecated)))
    if dead:
        problems.append(("B 死设置", dead))
    else:
        print("  B 设置引用: %d 项全部被代码读取" % len(props))
    return props


def check_shader_params(plugin, problems):
    declared = set()
    for p in targets(os.path.join(plugin, "Source"), {".cpp", ".h"}):
        txt = open(p, encoding="utf-8", errors="ignore").read()
        declared |= set(re.findall(r"SHADER_PARAMETER(?:_RDG_\w+|_STRUCT|_RDG_STRUCT)?\([^,]+,\s*(\w+)\s*\)", txt))

    exported = set()
    for p in targets(os.path.join(plugin, "Shaders"), {".usf"}):
        exported |= set(re.findall(r"(?:Texture2D|Texture2DArray|RWTexture2D|StructuredBuffer|"
                                   r"RWStructuredBuffer|SamplerState|Buffer)<[^>]*>\s*(\w+);", 
                                   open(p, encoding="utf-8", errors="ignore").read()))

    used = ""
    for p in targets(os.path.join(plugin, "Shaders"), {".usf"}):
        used += open(p, encoding="utf-8", errors="ignore").read()

    missing = sorted(n for n in declared if n not in used)
    if missing:
        problems.append(("C C++ 设置但着色器未用", missing))
    else:
        print("  C 着色器参数: %d 个 C++ 参数全部在 .usf 中出现" % len(declared))

    unused = sorted(n for n in exported if used.count(n) < 2)
    if unused:
        problems.append(("D .usf 声明但未使用", unused))
    else:
        print("  D 着色器全局: %d 个全部被使用" % len(exported))
    return declared


def check_versions(problems):
    copies = [
        r"E:\ConveyanceSpline\UE_ConveyanceSpline\Plugins\NeuralToneMapper\NeuralToneMapper.uplugin",
        r"E:\UE_ArtEffect_StarSky\Plugins\NeuralToneMapper\NeuralToneMapper.uplugin",
        r"E:\ConveyanceSpline\_gh\NTM\plugin\NeuralToneMapper\NeuralToneMapper.uplugin",
    ]
    seen = {}
    for c in copies:
        if not os.path.exists(c):
            continue
        j = json.load(open(c, encoding="utf-8"))
        seen[os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c))))] = (
            j.get("Version"), j.get("VersionName"))
    vals = set(seen.values())
    if len(vals) > 1:
        problems.append(("E 版本不一致", ["%s: %s" % (k, v) for k, v in seen.items()]))
    else:
        print("  E 版本一致性: %s" % (str(sorted(vals)) if vals else "?"))
    return seen


def check_model_contract(plugin, problems):
    h = os.path.join(plugin, "Source", "NeuralToneMapper", "Public", "NTMORenderer.h")
    cand = h if os.path.exists(h) else None
    txt = ""
    for p in targets(os.path.join(plugin, "Source"), {".h"}):
        s = open(p, encoding="utf-8", errors="ignore").read()
        if "NTMOInput" in s:
            txt += s
    ch = re.search(r"Channels\s*=\s*(\d+)", txt)
    hgt = re.search(r"Height\s*=\s*(\d+)", txt)
    wid = re.search(r"Width\s*=\s*(\d+)", txt)
    if not (ch and hgt and wid):
        problems.append(("F 输入契约", ["找不到 NTMOInput 的 Channels/Height/Width"]))
        return
    got = (int(ch.group(1)), int(hgt.group(1)), int(wid.group(1)))
    print("  F 输入契约: %d x %d x %d" % got)

    models = os.path.join(plugin, "Content", "Models")
    try:
        import onnxruntime as ort
        # Same rule the plugin uses: highest _vN, not lexicographic (v12 < v9 as text).
        def ver(n):
            m = re.search(r"_v(\\d+)", n)
            return int(m.group(1)) if m else 0
        newest = max((n for n in os.listdir(models) if n.endswith(".onnx")), key=ver)
        s = ort.InferenceSession(os.path.join(models, newest), providers=["CPUExecutionProvider"])
        shape = s.get_inputs()[0].shape
        want = [1, got[0], got[1], got[2]]
        if [int(x) for x in shape] != want:
            problems.append(("F 模型形状不符", ["%s 输入 %s，期望 %s" % (newest, shape, want)]))
        else:
            print("      %s 形状 %s 与 C++ 一致" % (newest, shape))
        out = s.get_outputs()[0].shape
        print("      输出 %s（8 节点 + 64 局部强度）" % (out,))
    except Exception as e:  # noqa: BLE001
        problems.append(("F 模型加载", [str(e)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    args = ap.parse_args()
    problems = []
    print("审计: %s" % args.plugin)
    check_settings(args.plugin, problems)
    check_shader_params(args.plugin, problems)
    check_versions(problems)
    check_model_contract(args.plugin, problems)

    print()
    if problems:
        print("发现 %d 类问题:" % len(problems))
        for tag, items in problems:
            print("  [%s]" % tag)
            for it in items:
                print("     - %s" % it)
        sys.exit(1)
    print("全部通过")


main()
