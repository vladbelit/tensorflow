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
  [string]$Version = 'v1.29.0',
  [string]$Sha256 = '092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b',
  [string]$TargetDir = 'C:\tools\bazel'
)

. "$PSScriptRoot\common.ps1"

Write-Output ('Installing Bazelisk {0}...' -f $Version)
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

$url = (
  'https://github.com/bazelbuild/bazelisk/releases/download/{0}/' +
  'bazelisk-windows-amd64.exe'
) -f $Version
$exePath = Join-Path $TargetDir 'bazel.exe'

Download-File -Url $url -Destination $exePath -Sha256 $Sha256
Add-ToMachinePath -Directory $TargetDir

# Running Bazelisk would download a Bazel release selected by the working
# directory. The final image verifier checks the pinned Bazelisk binary itself
# without introducing that second, context-dependent download.
