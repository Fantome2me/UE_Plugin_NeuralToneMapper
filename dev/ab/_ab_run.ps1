# Same-camera A/B driver for the Neural Tone Mapper.
#
# Launches the editor with the python harness, waits for each capture point, grabs the
# editor window, then tells the harness to advance via sentinel files. Both captures
# therefore share one camera, one sun angle and one streaming state, which is what makes
# a per-pixel comparison against Unreal's tonemapper valid.
#
# Usage:  pwsh -File _ab_run.ps1 [-OutDir <dir>]
param(
    [string]$OutDir = "E:\ConveyanceSpline\_ab",
    # Extra console commands injected at launch, e.g. "r.NTMO.Local 0,r.NTMO.Neural 0".
    # Used for cost attribution: same harness, different feature set.
    [string]$ExtraCvars = ""
)

$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W32Ab {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L,T,R,B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
}
"@

$Exe  = "D:\ue\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe"
$Proj = "E:\ConveyanceSpline\UE_ConveyanceSpline\UE_ConveyanceSpline.uproject"
$Log  = Join-Path $env:TEMP "ntmo_ab_run.txt"
$Root = "E:\ConveyanceSpline"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Get-Process | Where-Object { $_.ProcessName -match 'UnrealEditor|CrashReport' } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Remove-Item $Log -Force -ErrorAction SilentlyContinue
Remove-Item "$Root\_ab_step1.flag","$Root\_ab_step2.flag" -Force -ErrorAction SilentlyContinue
Set-Content -Path "$Root\_ab_run.flag" -Value "run" -Encoding ascii

$argList = @(
    "`"$Proj`"", "-nosplash", "-windowed", "-ResX=1600", "-ResY=900",
    "-stdout", "-FullStdOutLogOutput", "-NoAnalytics"
)
if ($ExtraCvars -ne "") {
    $argList += "-ExecCmds=`"$ExtraCvars`""
}

Start-Process -FilePath $Exe -ArgumentList $argList -WorkingDirectory "E:\ConveyanceSpline\UE_ConveyanceSpline" `
  -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

function Capture([string]$Name) {
    $p = Get-Process UnrealEditor -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { Write-Host "  editor gone, cannot capture $Name"; return $false }
    $h = $p.MainWindowHandle
    [void][W32Ab]::ShowWindow($h, 9)
    [void][W32Ab]::SetForegroundWindow($h)
    Start-Sleep -Milliseconds 800
    $r = New-Object W32Ab+RECT
    [void][W32Ab]::GetWindowRect($h, [ref]$r)
    $w = $r.R - $r.L; $ht = $r.B - $r.T
    if ($w -le 0 -or $ht -le 0) { Write-Host "  bad window rect"; return $false }
    $bmp = New-Object System.Drawing.Bitmap($w, $ht)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.L, $r.T, 0, 0, (New-Object System.Drawing.Size($w, $ht)))
    $path = Join-Path $OutDir "$Name.png"
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Write-Host "  captured $path (${w}x${ht})"
    return $true
}

function Wait-Marker([string]$Marker, [int]$TimeoutSec = 240) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $Log) {
            $hit = Select-String -Path $Log -Pattern ([regex]::Escape($Marker)) -ErrorAction SilentlyContinue
            if ($hit) { Write-Host "  marker: $Marker"; return $true }
        }
        Start-Sleep -Milliseconds 400
    }
    Write-Host "  TIMEOUT waiting for $Marker"
    return $false
}

Write-Host "Waiting for camera + plugin ON ..."
if (-not (Wait-Marker "ON_READY")) { Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force; exit 2 }
Start-Sleep -Milliseconds 500
[void](Capture "ntmo_plugin_on")

# Only now is it safe to disable the plugin: the ON capture is already on disk.
Set-Content -Path "$Root\_ab_step1.flag" -Value "go" -Encoding ascii

Write-Host "Waiting for plugin OFF ..."
if (-not (Wait-Marker "OFF_READY")) { Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force; exit 3 }
Start-Sleep -Milliseconds 500
[void](Capture "ue_tonemapper_off")

Set-Content -Path "$Root\_ab_step2.flag" -Value "go" -Encoding ascii
Start-Sleep -Seconds 2
Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "A/B capture complete -> $OutDir"
