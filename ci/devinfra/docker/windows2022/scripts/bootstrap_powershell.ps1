# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script must remain compatible with Windows PowerShell 5.1, as it
# bootstraps the PowerShell 7 runtime used by every remaining image layer.
param (
  [string]$Version = '7.4.17',
  [string]$Sha256 = '882dda4d2ccbcb36f0a8a037a4fbe5a5d64f27af1c09ba90edf67b57f6f559ef'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$url = (
  'https://github.com/PowerShell/PowerShell/releases/download/v{0}/' +
  'PowerShell-{0}-win-x64.msi'
) -f $Version
$msiPath = Join-Path $env:TEMP 'PowerShell-win-x64.msi'
$logPath = Join-Path $env:TEMP 'PowerShell-install.log'

for ($attempt = 1; $attempt -le 3; $attempt++) {
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $msiPath
    break
  }
  catch {
    if ($attempt -eq 3) {
      throw
    }
    Start-Sleep -Seconds 5
  }
}

$actualHash = (Get-FileHash -LiteralPath $msiPath -Algorithm SHA256).Hash
if ($actualHash -ne $Sha256) {
  throw ('PowerShell MSI SHA-256 mismatch. Expected {0}; actual {1}.' -f `
      $Sha256, $actualHash)
}

$arguments = ('/i "{0}" /qn /norestart /log "{1}" ' +
              'ADD_PATH=1 ENABLE_PSREMOTING=0 REGISTER_MANIFEST=1 ' +
              'USE_MU=0 ENABLE_MU=0') -f $msiPath, $logPath
$process = Start-Process -FilePath 'msiexec.exe' `
  -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -notin @(0, 3010)) {
  if (Test-Path -LiteralPath $logPath) {
    Get-Content -LiteralPath $logPath | Select-Object -Last 40
  }
  throw ('PowerShell MSI failed with exit code {0}.' -f $process.ExitCode)
}

$pwshExe = Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'
if (-not (Test-Path -LiteralPath $pwshExe -PathType Leaf)) {
  throw ('PowerShell executable was not installed: {0}' -f $pwshExe)
}
$installedVersion = & $pwshExe -NoLogo -NoProfile -Command `
  '$PSVersionTable.PSVersion.ToString()'
if ($LASTEXITCODE -ne 0 -or $installedVersion -ne $Version) {
  throw ('Expected PowerShell {0}; found {1}.' -f $Version, $installedVersion)
}

Remove-Item -LiteralPath $msiPath -Force
Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue
