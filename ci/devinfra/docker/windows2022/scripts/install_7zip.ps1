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

param (
  [string]$Url = 'https://github.com/ip7z/7zip/releases/download/26.02/7z2602-x64.msi',
  [string]$Sha256 = 'db407a4f6d4999e5c7bc00ce8a882be94717b56e7fa68140fe3f12605d91643e'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing 7-Zip...'
$msiPath = Join-Path $env:TEMP '7z.msi'
$logPath = Join-Path $env:TEMP '7z-install.log'

Download-File -Url $Url -Destination $msiPath -Sha256 $Sha256

try {
  Invoke-NativeCommand -FilePath 'msiexec.exe' -ArgumentList @(
    '/i', $msiPath, '/qn', '/norestart', '/log', $logPath
  ) -SuccessExitCodes @(0, 3010)
}
catch {
  if (Test-Path -LiteralPath $logPath) {
    Get-Content -LiteralPath $logPath | Select-Object -Last 40
  }
  throw
}

$installDir = Join-Path $env:ProgramFiles '7-Zip'
$sevenZipExe = Join-Path $installDir '7z.exe'
Add-ToMachinePath -Directory $installDir
Assert-CommandVersion -FilePath $sevenZipExe -ExpectedPattern '^7-Zip '

Remove-Item -LiteralPath $msiPath -Force
Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue
