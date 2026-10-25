# Build dist\ArmStrong.exe: one self-contained Windows executable (Python, MuJoCo, the simulation, the Panda model
# and the built frontend). Users only need the exe; it serves http://127.0.0.1:8770 and opens the browser.
#   powershell -ExecutionPolicy Bypass -File .\build-exe.ps1
# Needs Python 3.12 (py launcher) and Node 24 on the build machine, not on the user's.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
$lab = Join-Path $root "backend\robotics-lab\robotics-lab"

function Invoke-Checked([scriptblock]$step, [string]$what) {
    & $step
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

if (-not (Test-Path $py)) {
    Invoke-Checked { py -3.12 -m venv (Join-Path $root "backend\.venv") } "Creating the Python environment"
    Invoke-Checked { & $py -m pip install --upgrade pip } "Upgrading pip"
}
Invoke-Checked { & $py -m pip install --disable-pip-version-check -q -r (Join-Path $lab "requirements-studio.txt") } "Installing the app requirements"
Invoke-Checked { & $py -m pip install --disable-pip-version-check -q -r (Join-Path $root "packaging\requirements-build.txt") } "Installing PyInstaller"

Push-Location (Join-Path $root "frontend")
try {
    if (-not (Test-Path node_modules)) { Invoke-Checked { npm install } "npm install" }
    Invoke-Checked { npm run build } "Building the frontend"
} finally {
    Pop-Location
}

Invoke-Checked {
    & $py -m PyInstaller --noconfirm --clean --log-level WARN `
        --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build") `
        (Join-Path $root "packaging\ArmStrong.spec")
} "PyInstaller"

$exe = Get-Item (Join-Path $root "dist\ArmStrong.exe")
Write-Host ("Built {0} ({1:N0} MB)" -f $exe.FullName, ($exe.Length / 1MB))
