# ArmStrong launcher (Windows PowerShell 5.1 or newer). start.bat runs this file.
#
# Nothing has to be installed first. If Python 3.12 or Node.js is missing, a private copy is downloaded into
# %LOCALAPPDATA%\ArmStrong\tools (no administrator rights, nothing added to PATH; delete that folder to remove it).
# First run: creates the Python environment, installs the dependencies and builds the web app (a few minutes).
# After that it serves http://127.0.0.1:8770 and opens the browser.
#
#   powershell -ExecutionPolicy Bypass -File .\start-studio.ps1 [-NoBrowser] [-Port 8770]
param(
    [switch]$NoBrowser,
    [int]$Port = 8770
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest very slow in Windows PowerShell 5.1
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$root = $PSScriptRoot
$lab = Join-Path $root "backend\robotics-lab\robotics-lab"
$venv = Join-Path $root "backend\.venv"
$py = Join-Path $venv "Scripts\python.exe"

$toolsBase = $env:LOCALAPPDATA
if (-not $toolsBase) { $toolsBase = Join-Path $env:USERPROFILE "AppData\Local" }
$tools = Join-Path $toolsBase "ArmStrong\tools"

$PythonVersion = "3.12.10"     # last 3.12 release that ships binaries; the project is verified on 3.12
$NodeMajor = 24                # the project is verified on Node 24
$NodeFallback = "v24.22.1"     # used only when nodejs.org's release list cannot be read
$NodeMinimum = 22              # an installed Node at least this new is used as is

function Say([string]$text) {
    Write-Host "[ArmStrong] $text" -ForegroundColor Cyan
}

function Get-File([string]$url, [string]$dest) {
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            return
        } catch {
            if ($attempt -eq 3) {
                throw "Could not download $url ($($_.Exception.Message)). Check the internet connection and run start.bat again."
            }
            Start-Sleep -Seconds 2
        }
    }
}

function Expand-Zip([string]$zip, [string]$dest) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
}

# ---------------------------------------------------------------------------------------------------- Python

function Find-Python {
    # Full path of a working Python 3.12 (system-wide or our private copy), or $null.
    $ErrorActionPreference = "Continue"
    $candidates = @(
        @{ Exe = "py"; Pre = @("-3.12") },
        @{ Exe = "python"; Pre = @() },
        @{ Exe = (Join-Path $tools "python\python.exe"); Pre = @() }
    )
    foreach ($c in $candidates) {
        $exe = $c.Exe
        $pre = $c.Pre
        try {
            $out = @(& $exe @pre -c "import sys; print(sys.executable); print('%d.%d' % sys.version_info[:2])" 2>$null)
        } catch {
            continue
        }
        if ($LASTEXITCODE -eq 0 -and $out.Count -ge 2 -and "$($out[-1])".Trim() -eq "3.12") {
            return "$($out[0])".Trim()
        }
    }
    return $null
}

function Install-Python {
    Say "Python 3.12 was not found. Downloading a private copy (one time) ..."
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = Join-Path $tools "python-$PythonVersion.zip"
    $tmp = Join-Path $tools "python-extract"
    $final = Join-Path $tools "python"
    Get-File "https://www.nuget.org/api/v2/package/python/$PythonVersion" $zip
    Expand-Zip $zip $tmp
    if (Test-Path $final) { Remove-Item -Recurse -Force $final }
    Move-Item (Join-Path $tmp "tools") $final
    Remove-Item -Recurse -Force $tmp
    Remove-Item -Force $zip
    $exe = Join-Path $final "python.exe"
    if (-not (Test-Path $exe)) { throw "The Python download did not unpack correctly. Run start.bat again." }
    return $exe
}

function Test-VenvRuns {
    if (-not (Test-Path $py)) { return $false }
    $ErrorActionPreference = "Continue"
    try {
        & $py -c "import sys" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-VenvPackages {
    $ErrorActionPreference = "Continue"
    try {
        & $py -c "import mujoco, numpy, scipy, osqp, yaml, PIL, fastapi, uvicorn" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------------------------------- Node.js

function Find-NodeDir {
    # Folder that holds node.exe and npm.cmd (Node $NodeMinimum or newer), or $null.
    $ErrorActionPreference = "Continue"
    $dirs = @()
    $found = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($found) { $dirs += (Split-Path $found.Source) }
    $dirs += (Join-Path $env:ProgramFiles "nodejs")
    $dirs += (Join-Path $tools "node")
    foreach ($dir in $dirs) {
        $node = Join-Path $dir "node.exe"
        if (-not ((Test-Path $node) -and (Test-Path (Join-Path $dir "npm.cmd")))) { continue }
        try {
            $version = "$(& $node --version 2>$null)"
        } catch {
            continue
        }
        if ($version -match '^v(\d+)\.' -and [int]$Matches[1] -ge $NodeMinimum) { return $dir }
    }
    return $null
}

function Install-Node {
    Say "Node.js was not found. Downloading a private copy (one time, only used to build the web app) ..."
    New-Item -ItemType Directory -Force $tools | Out-Null
    $version = $NodeFallback
    try {
        $releases = Invoke-RestMethod -Uri "https://nodejs.org/dist/index.json"
        $latest = $releases | Where-Object { $_.version -like "v$NodeMajor.*" -and ($_.files -contains "win-x64-zip") } | Select-Object -First 1
        if ($latest) { $version = $latest.version }
    } catch { }

    $name = "node-$version-win-x64"
    $base = "https://nodejs.org/dist/$version"
    $zip = Join-Path $tools "$name.zip"
    $sums = Join-Path $tools "node-SHASUMS256.txt"
    $tmp = Join-Path $tools "node-extract"
    $final = Join-Path $tools "node"

    Get-File "$base/$name.zip" $zip
    Get-File "$base/SHASUMS256.txt" $sums
    $line = Get-Content $sums | Where-Object { $_ -match ('\s' + [regex]::Escape("$name.zip") + '$') } | Select-Object -First 1
    if (-not $line) { throw "Could not check the Node.js download. Run start.bat again." }
    $expected = ($line -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Remove-Item -Force $zip, $sums
        throw "The Node.js download is damaged. Run start.bat again."
    }

    Expand-Zip $zip $tmp
    if (Test-Path $final) { Remove-Item -Recurse -Force $final }
    Move-Item (Join-Path $tmp $name) $final
    Remove-Item -Recurse -Force $tmp
    Remove-Item -Force $zip, $sums
}

# ---------------------------------------------------------------------------------------------------- main

try {
    if (-not (Test-Path (Join-Path $lab "studio.py"))) {
        throw "Cannot find backend\robotics-lab\robotics-lab\studio.py next to start.bat. Keep the whole ArmStrong folder together."
    }

    # 1. Python environment (backend\.venv). A copy made on another computer does not run here, so rebuild it.
    if (-not (Test-VenvRuns)) {
        if (Test-Path $venv) {
            Say "The Python environment in backend\.venv was made on another computer. Rebuilding it ..."
            Remove-Item -Recurse -Force $venv
        } else {
            Say "First run: setting up Python. This takes a few minutes."
        }
        $basePython = Find-Python
        if (-not $basePython) { $basePython = Install-Python }
        & $basePython -m venv $venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $py)) {
            throw "Could not create the Python environment in backend\.venv."
        }
    }

    if (-not (Test-VenvPackages)) {
        Say "Installing the simulator and web server ..."
        & $py -m pip install --disable-pip-version-check -r (Join-Path $lab "requirements-studio.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Installing the Python packages failed. Check the internet connection and run start.bat again."
        }
        if (-not (Test-VenvPackages)) {
            throw "The Python packages are installed but do not load. Install the Microsoft Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) and run start.bat again."
        }
    }

    # 2. Web app (frontend\dist). Node.js is only needed to build it.
    if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
        Say "First run: building the web app ..."
        $nodeDir = Find-NodeDir
        if (-not $nodeDir) {
            Install-Node
            $nodeDir = Find-NodeDir
            if (-not $nodeDir) { throw "Node.js could not be set up. Run start.bat again." }
        }
        $env:PATH = "$nodeDir;$env:PATH"
        $npm = Join-Path $nodeDir "npm.cmd"
        Push-Location (Join-Path $root "frontend")
        try {
            & $npm install --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw "npm install failed. Check the internet connection and run start.bat again." }
            & $npm run build
            if ($LASTEXITCODE -ne 0) { throw "Building the web app failed." }
        } finally {
            Pop-Location
        }
    }

    # 3. Run.
    $studioArgs = @("studio.py", "--port", "$Port")
    if ($NoBrowser) { $studioArgs += "--no-browser" }
    Say "Starting ArmStrong at http://127.0.0.1:$Port (close this window or press Ctrl+C to stop)."
    Push-Location $lab
    try {
        & $py @studioArgs
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    exit $exitCode
} catch {
    Write-Host ""
    Write-Host "[ArmStrong] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
