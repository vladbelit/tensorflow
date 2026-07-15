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
  [string]$PackageName = 'zulu25.34.17-ca-jdk25.0.3-win_x64.zip',
  [string]$Sha256 = 'ddd68ee54e78b6b19388f517b8a61fbed2856ae3320febe42449bac3021e2729',
  [string]$BaseUrl = 'https://cdn.azul.com/zulu/bin',
  [string]$TargetDir = 'C:\openjdk'
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing Azul Zulu JDK...'
Add-Type -AssemblyName 'System.IO.Compression.FileSystem'

$url = ('{0}/{1}' -f $BaseUrl.TrimEnd('/'), $PackageName)
$zipPath = Join-Path $env:TEMP $PackageName
$stagingDir = Join-Path $env:TEMP 'jdk-extract'

Download-File -Url $url -Destination $zipPath -Sha256 $Sha256
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $stagingDir)

$extractedDirs = @(Get-ChildItem -LiteralPath $stagingDir -Directory)
if ($extractedDirs.Count -ne 1) {
  throw ('Expected one JDK archive root, found {0}.' -f $extractedDirs.Count)
}

$parentDir = Split-Path -Parent $TargetDir
New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
Move-Item -LiteralPath $extractedDirs[0].FullName -Destination $TargetDir

$binDir = Join-Path $TargetDir 'bin'
$javaExe = Join-Path $binDir 'java.exe'
Add-ToMachinePath -Directory $binDir
$env:JAVA_HOME = $TargetDir
[Environment]::SetEnvironmentVariable('JAVA_HOME', $TargetDir, 'Machine')
Assert-CommandVersion -FilePath $javaExe -ArgumentList @('-version') `
  -ExpectedPattern '^openjdk version '

Remove-Item -LiteralPath $zipPath -Force
Remove-Item -LiteralPath $stagingDir -Recurse -Force
