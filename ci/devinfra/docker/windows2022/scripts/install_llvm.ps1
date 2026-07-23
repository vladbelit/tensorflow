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
  [string]$Url = (
    'https://github.com/llvm/llvm-project/releases/download/' +
    'llvmorg-18.1.4/LLVM-18.1.4-win64.exe'
  ),
  [string]$Sha256 = '78d8f528a132e131b978e6b3276aa45af759a068a25c006240f59ab28df5f621',
  [string]$TargetDir = 'C:\tools\LLVM',
  [switch]$SkipPath
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing LLVM...'
$installerPath = Join-Path $env:TEMP 'LLVM-win64.exe'

Download-File -Url $Url -Destination $installerPath -Sha256 $Sha256
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Invoke-NativeCommand -FilePath '7z.exe' -ArgumentList @(
  'x', $installerPath, ('-o{0}' -f $TargetDir), '-y'
)

$binDir = Join-Path $TargetDir 'bin'
$clangExe = Join-Path $binDir 'clang.exe'
if (-not $SkipPath) {
  Add-ToMachinePath -Directory $binDir
}
Assert-CommandVersion -FilePath $clangExe -ArgumentList @('--version') `
  -ExpectedPattern '^clang version '

Remove-Item -LiteralPath $installerPath -Force
