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
  [string]$Version = '0.11.28',
  [string]$Sha256 = '0a23463216d09c6a72ff80ef5dc5a795f07dc1575cb84d24596c2f124a441b7b',
  [string]$TargetDir = 'C:\tools\uv'
)

. "$PSScriptRoot\common.ps1"

$url = (
  'https://github.com/astral-sh/uv/releases/download/{0}/' +
  'uv-x86_64-pc-windows-msvc.zip'
) -f $Version
$zipPath = Join-Path $env:TEMP 'uv.zip'

Write-Output ('Installing uv {0}...' -f $Version)
Download-File -Url $url -Destination $zipPath -Sha256 $Sha256

# The Windows archive stores its executables at the archive root.
Expand-Archive -LiteralPath $zipPath -DestinationPath $TargetDir -Force

$uvExe = Join-Path $TargetDir 'uv.exe'
Add-ToMachinePath -Directory $TargetDir
Assert-CommandVersion -FilePath $uvExe `
  -ArgumentList @('--version') -ExpectedPattern ('^uv {0}\b' -f `
    [regex]::Escape($Version))

Remove-Item -LiteralPath $zipPath -Force
