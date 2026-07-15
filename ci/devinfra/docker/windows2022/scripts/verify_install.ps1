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

. "$PSScriptRoot\common.ps1"

function Assert-ResolvedCommand {
  param (
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedPath
  )

  # Get-Command returns every matching application on PATH. The first match is
  # the command that users actually invoke.
  $command = Get-Command -Name $Name -CommandType Application `
    -ErrorAction Stop | Select-Object -First 1

  $actualPath = [IO.Path]::GetFullPath($command.Source)
  $expectedFullPath = [IO.Path]::GetFullPath($ExpectedPath)
  if ($actualPath -ine $expectedFullPath) {
    throw ('{0} resolved to {1}; expected {2}.' -f `
        $Name, $actualPath, $expectedFullPath)
  }
}

Write-Output 'Verifying Windows build image...'

$pwshExe = Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'
$bashExe = 'C:\tools\msys64\usr\bin\bash.exe'
$jdkExe = 'C:\openjdk\bin\java.exe'
$gcloudCmd = 'C:\tools\google-cloud-sdk\bin\gcloud.cmd'
$expectedCommands = [ordered]@{
  'pwsh.exe' = $pwshExe
  '7z.exe' = (Join-Path $env:ProgramFiles '7-Zip\7z.exe')
  'clang.exe' = 'C:\tools\LLVM\bin\clang.exe'
  'bash.exe' = $bashExe
  'java.exe' = $jdkExe
  'bazel.exe' = 'C:\tools\bazel\bazel.exe'
  'uv.exe' = 'C:\tools\uv\uv.exe'
  'uvw.exe' = 'C:\tools\uv\uvw.exe'
  'uvx.exe' = 'C:\tools\uv\uvx.exe'
  'gcloud.cmd' = $gcloudCmd
}
foreach ($command in $expectedCommands.GetEnumerator()) {
  Assert-ResolvedCommand -Name $command.Key -ExpectedPath $command.Value
}

Assert-CommandVersion -FilePath $bashExe -ArgumentList @(
  '-lc', 'pacman -Q -- curl git patch unzip vim wget zip'
) -ExpectedPattern '^curl '

$pacmanConf = 'C:\tools\msys64\etc\pacman.conf'
$pacmanConfigs = @($pacmanConf) + @(
  Get-ChildItem -LiteralPath (Join-Path (Split-Path $pacmanConf) 'pacman.d') `
    -Filter 'mirrorlist*' -File | Select-Object -ExpandProperty FullName
)
$unsignedConfig = @(Select-String -LiteralPath $pacmanConfigs `
    -Pattern '^(?!\s*#)[^#]*SigLevel\s*=[^#]*\bNever\b')
if ($unsignedConfig) {
  throw ('MSYS2 signature verification is disabled in {0}.' -f `
      $unsignedConfig[0].Path)
}

if ($env:JAVA_HOME -ne 'C:\openjdk') {
  throw ('Unexpected JAVA_HOME: {0}' -f $env:JAVA_HOME)
}

$pythonMinors = @('3.10', '3.11', '3.12', '3.13', '3.14', '3.15')
foreach ($minor in $pythonMinors) {
  $pythonDir = 'C:\Python{0}' -f $minor
  $pythonExe = Join-Path $pythonDir 'python.exe'
  $aliasName = 'python{0}.exe' -f $minor
  $aliasPath = Join-Path $pythonDir $aliasName

  Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @(
    '-c', 'import packaging, setuptools'
  )
  Assert-ResolvedCommand -Name $aliasName -ExpectedPath $aliasPath
}

$defaultPython = 'C:\Python3.14\python.exe'
$defaultPython3 = 'C:\Python3.14\python3.exe'
Assert-ResolvedCommand -Name 'python.exe' -ExpectedPath $defaultPython
Assert-ResolvedCommand -Name 'python3.exe' -ExpectedPath $defaultPython3
Assert-CommandVersion -FilePath 'py.exe' -ArgumentList @('--version') `
  -ExpectedPattern '^Python 3\.14\.'

$defaultPip = 'C:\Python3.14\Scripts\pip.exe'
Assert-ResolvedCommand -Name 'pip' -ExpectedPath $defaultPip
Assert-CommandVersion -FilePath 'pip.exe' -ArgumentList @('--version') `
  -ExpectedPattern '^pip .+\(python 3\.14\)$'

Assert-CommandVersion -FilePath $bashExe -ArgumentList @(
  '-ic', 'alias gcloud && alias gsutil && alias bq && gcloud --version'
) -ExpectedPattern '(?s)alias gcloud=.*Google Cloud SDK'

$vsRoot = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
$vcRoot = Join-Path $vsRoot 'VC'
$clExecutables = @(Get-ChildItem -Path `
    (Join-Path $vcRoot 'Tools\MSVC\*\bin\Hostx64\x64\cl.exe') -File)
if ($clExecutables.Count -eq 0) {
  throw 'No x64-hosted MSVC compiler was found.'
}

$sdkBinRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
$sdkDirs = @(Get-ChildItem -LiteralPath $sdkBinRoot -Directory |
    Where-Object { $_.Name -like '10.0.26100.*' })
if ($sdkDirs.Count -eq 0) {
  throw 'Windows SDK 10.0.26100 was not found.'
}

Write-Output 'Windows build image verification complete.'
