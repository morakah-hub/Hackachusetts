# Stop a running Armstrong Studio server (only processes whose command line runs scripts\serve.py).
#   powershell -ExecutionPolicy Bypass -File scripts\stop.ps1 [-Port 8765]
param([int]$Port = 8765)
$found = $false
foreach ($c in (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)"
    if ($p -and ($p.CommandLine -like '*serve.py*' -or $p.CommandLine -like '*armstrong.server*')) {
        Stop-Process -Id $p.ProcessId -Force
        Write-Host "stopped Armstrong Studio server (pid $($p.ProcessId)) on port $Port"
        $found = $true
    } elseif ($p) {
        Write-Host "port $Port is used by another program (pid $($p.ProcessId)); not touching it"
    }
}
if (-not $found) { Write-Host "no Armstrong Studio server listening on port $Port" }
