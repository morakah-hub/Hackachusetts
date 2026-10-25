# Armstrong Studio launcher (Windows PowerShell). Start.exe and start.bat run this script.
#   powershell -ExecutionPolicy Bypass -File .\start.ps1             # set up if needed, then start
#   powershell -ExecutionPolicy Bypass -File .\start.ps1 -Port 8770  # preferred port
#   powershell -ExecutionPolicy Bypass -File .\start.ps1 -Rebuild    # rebuild the frontend (downloads Node.js)
#   powershell -ExecutionPolicy Bypass -File .\start.ps1 -NoBrowser  # do not open the browser
# First run: finds Python 3.11-3.13 (or downloads a private copy with uv if there is none), creates .venv and
# installs requirements.txt. Everything stays inside this folder; nothing is installed system-wide.
param([int]$Port = 8765, [switch]$Rebuild, [switch]$NoBrowser)
# Native tools (python, pip) write warnings to stderr; exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Tools = Join-Path $Root '.tools'
$Host.UI.RawUI.WindowTitle = 'Armstrong Studio'

function Say($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host ''
    Write-Host "  PROBLEM: $msg" -ForegroundColor Red
    Write-Host ''
    exit 1
}

Write-Host ''
Write-Host '  =========================================' -ForegroundColor DarkCyan
Write-Host '   Armstrong Studio - robot workcell setup' -ForegroundColor White
Write-Host '  =========================================' -ForegroundColor DarkCyan
Write-Host ''

# ---------------------------------------------------------------- 1. Python environment (.venv)
function Test-PyVersion($exe, [string[]]$pre) {
    try {
        $v = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $v) { return $null }
        $parts = "$v".Trim().Split('.')
        if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11 -and [int]$parts[1] -le 13) { return "$v".Trim() }
    } catch { }
    return $null
}

if (Test-Path $Py) {
    if (-not (Test-PyVersion $Py @())) {
        Say 'The existing .venv is not usable on this computer; recreating it...'
        Remove-Item -Recurse -Force $Venv
    }
}
if (-not (Test-Path $Py)) {
    $made = $false
    $candidates = @(@('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('python'), @('python3'))
    foreach ($c in $candidates) {
        $exe = $c[0]
        $pre = @($c | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $ver = Test-PyVersion $exe $pre
        if ($ver) {
            Say "Creating the Python environment with Python $ver (one time)..."
            & $exe @pre -m venv $Venv
            if ($LASTEXITCODE -eq 0 -and (Test-Path $Py)) { $made = $true; break }
        }
    }
    if (-not $made) {
        # No suitable Python: fetch uv (a single-file Python manager) and let it download a private Python 3.12.
        $Uv = Join-Path $Tools 'uv\uv.exe'
        if (-not (Test-Path $Uv)) {
            Say 'Python 3.11-3.13 was not found. Downloading a private copy (about 40 MB, one time)...'
            New-Item -ItemType Directory -Force (Join-Path $Tools 'uv') | Out-Null
            $zip = Join-Path $Tools 'uv.zip'
            try {
                Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip -ErrorAction Stop
                Expand-Archive $zip -DestinationPath (Join-Path $Tools 'uv') -Force -ErrorAction Stop
                Remove-Item $zip
            } catch {
                Fail 'Could not download Python (no internet?). Install Python 3.12 from https://www.python.org/downloads/ (tick "Add to PATH") and run Start again.'
            }
        }
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
        & $Uv venv --python 3.12 $Venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Py)) { Fail 'Could not create the Python environment.' }
        & $Uv pip install --python $Py pip | Out-Null
    }
}

& $Py -c "import mujoco, fastapi, uvicorn, scipy, PIL" 2>$null
if ($LASTEXITCODE -ne 0) {
    Say 'Installing the simulation packages (MuJoCo, FastAPI, ...). This takes a minute the first time...'
    & $Py -m pip install --disable-pip-version-check -q -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Fail 'Installing the Python packages failed. Check your internet connection and run Start again.' }
}

# ---------------------------------------------------------------- 2. Frontend (prebuilt in frontend\dist)
$Dist = Join-Path $Root 'frontend\dist\index.html'
if ($Rebuild -or -not (Test-Path $Dist)) {
    Say 'Building the browser app (downloads a private Node.js, one time)...'
    & (Join-Path $Root 'scripts\npm.ps1') ci --no-fund --no-audit
    if ($LASTEXITCODE -ne 0) { Fail 'npm ci failed.' }
    & (Join-Path $Root 'scripts\npm.ps1') run build
    if ($LASTEXITCODE -ne 0) { Fail 'The frontend build failed.' }
}

# ---------------------------------------------------------------- 3. A free port
$chosen = $null
foreach ($p in $Port..($Port + 10)) {
    if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { $chosen = $p; break }
}
if (-not $chosen) { Fail "Ports $Port-$($Port + 10) are all in use. Close other programs or use -Port <n>." }

# ---------------------------------------------------------------- 4. Run (simulation + API + app on 127.0.0.1)
Say 'Starting the simulation...'
$serveArgs = @('-u', (Join-Path $Root 'scripts\serve.py'), '--port', $chosen)
if ($NoBrowser) { $serveArgs += '--no-browser' }
& $Py @serveArgs
exit $LASTEXITCODE
