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

$ErrorActionPreference = 'Stop'

# Windows containers cannot use the host network. Route the GCE metadata
# server through this container's default gateway instead.
$defaultRoute = Get-NetRoute -AddressFamily IPv4 `
  -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
  Sort-Object -Property RouteMetric | Select-Object -First 1

# A networkless container has no route to the metadata server, but its command
# should still run.
if ($defaultRoute) {
  $metadataPrefix = '169.254.169.254/32'
  Remove-NetRoute -AddressFamily IPv4 -DestinationPrefix $metadataPrefix `
    -Confirm:$false -ErrorAction SilentlyContinue
  New-NetRoute -AddressFamily IPv4 -DestinationPrefix $metadataPrefix `
    -InterfaceIndex $defaultRoute.InterfaceIndex `
    -NextHop $defaultRoute.NextHop | Out-Null
}

# Ignore native exit codes from route setup; only the forwarded command should
# determine the container's exit code.
$global:LASTEXITCODE = $null

# Avoid a param block: pwsh -File would bind options such as a child pwsh's
# -NoProfile to this script instead of forwarding them.
$Command = @($args)

if ($Command.Count -gt 0) {
  $exe = $Command[0]
  if ($Command.Count -gt 1) {
    $commandArguments = $Command[1..($Command.Count - 1)]
    & $exe @commandArguments
  }
  else {
    & $exe
  }
}
else {
  pwsh.exe
}

$commandSuccess = $?
$exitCode = $global:LASTEXITCODE

if ($exitCode -ne $null) {
  exit $exitCode
} elseif (-not $commandSuccess) {
  exit 1
}
