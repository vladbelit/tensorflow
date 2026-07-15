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
  [string]$Version = '575.0.1',
  [string]$Sha256 = '01b091921d8691e4e206aad7e4001296d2afbd1a6906ca3c279e341de4b25b49',
  [string]$PythonPath = 'C:\Python3.14\python.exe'
)

. "$PSScriptRoot\common.ps1"

$url = (
  'https://storage.googleapis.com/cloud-sdk-release/' +
  'google-cloud-cli-{0}-windows-x86_64.zip'
) -f $Version
$zipPath = Join-Path $env:TEMP 'google-cloud-cli.zip'
# The install path must not have spaces: the MSYS2-to-cmd.exe boundary loses quoting
# when both a .cmd path and one of its arguments contain spaces e.g.,
# `gcloud storage cp path-with-spaces ...`,  resulting in failure.
$sdkRoot = 'C:\tools\google-cloud-sdk'
$sdkParent = Split-Path -Parent $sdkRoot

Write-Output ('Installing Google Cloud CLI {0}...' -f $Version)
Download-File -Url $url -Destination $zipPath -Sha256 $Sha256
Expand-Archive -LiteralPath $zipPath -DestinationPath $sdkParent

$env:CLOUDSDK_CORE_DISABLE_PROMPTS = '1'
$env:CLOUDSDK_PYTHON = $PythonPath
[Environment]::SetEnvironmentVariable('CLOUDSDK_PYTHON', $PythonPath, 'Machine')

$installBat = Join-Path $sdkRoot 'install.bat'
Invoke-NativeCommand -FilePath $installBat -ArgumentList @(
  '--quiet', '--usage-reporting', 'false', '--command-completion', 'false',
  '--path-update', 'false', '--no-compile-python'
)

$binDir = Join-Path $sdkRoot 'bin'
Add-ToMachinePath -Directory $binDir

# Global MSYS path conversion is disabled for Bazel interoperability, so the
# extensionless Google Cloud CLI launchers pass POSIX paths to Windows Python.
# These aliases select the Windows .cmd entry points instead.
$msysBashrc = 'C:\tools\msys64\.bashrc'
Add-Content -LiteralPath $msysBashrc -Value @(
  'alias gcloud=gcloud.cmd',
  'alias gsutil=gsutil.cmd',
  'alias bq=bq.cmd'
)

Assert-CommandVersion -FilePath 'cmd.exe' -ArgumentList @(
  '/d', '/s', '/c', 'gcloud.cmd --version'
) -ExpectedPattern ('Google Cloud SDK {0}' -f [regex]::Escape($Version))

Remove-Item -LiteralPath $zipPath -Force
Remove-Item Env:CLOUDSDK_CORE_DISABLE_PROMPTS
