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
  # This is the immutable target of the VS 2022 evergreen URL as of 2026-07-13.
  # The bootstrapper still selects serviced component payloads from its channel.
  [string]$Url = (
    'https://download.visualstudio.microsoft.com/download/pr/' +
    '2ae938ff-cbb6-4e4d-990c-7794a7a03745/' +
    '15bc8cfc727e099aaba861eb21a811776bcbddc603c0a3f203eba5e5a7710421/' +
    'vs_BuildTools.exe'
  ),
  [string]$Sha256 = '15bc8cfc727e099aaba861eb21a811776bcbddc603c0a3f203eba5e5a7710421',
  [string]$InstallPath = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
  [string []]$Components = @(
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    'Microsoft.VisualStudio.Component.Windows11SDK.26100',
    'Microsoft.VisualStudio.Workload.VCTools'
  )
)

. "$PSScriptRoot\common.ps1"

Write-Output 'Installing Visual Studio 2022 Build Tools...'
$installerPath = Join-Path $env:TEMP 'vs_BuildTools.exe'

Download-File -Url $Url -Destination $installerPath -Sha256 $Sha256

$arguments = @(
  '--installPath', $InstallPath,
  '--quiet', '--wait', '--norestart', '--nocache'
)
foreach ($component in $Components) {
  $arguments += @('--add', $component)
}

Invoke-NativeCommand -FilePath $installerPath -ArgumentList $arguments `
  -SuccessExitCodes @(0, 3010)

Remove-Item -LiteralPath $installerPath -Force
