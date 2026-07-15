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
    'https://repo.msys2.org/distrib/x86_64/' +
    'msys2-base-x86_64-20260611.tar.xz'
  ),
  [string]$Sha256 = 'a2d047e8ee213c3c6a49a8de427eb1069df12207c0422ff1b3cbb5c905c34221',
  [string]$TargetDir = 'C:\tools',
  [string []]$Packages = @(
    'curl', 'git', 'patch', 'unzip', 'vim', 'wget', 'zip'
  )
)

. "$PSScriptRoot\common.ps1"

foreach ($package in $Packages) {
  if ($package -notmatch '^[a-zA-Z0-9@._+-]+$') {
    throw ('Unsafe MSYS2 package name: {0}' -f $package)
  }
}

Write-Output 'Installing MSYS2...'
$txzPath = Join-Path $env:TEMP 'msys2.tar.xz'
$tarDir = Join-Path $env:TEMP 'msys2-tar'
$msysRoot = Join-Path $TargetDir 'msys64'

New-Item -ItemType Directory -Path $tarDir -Force | Out-Null
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Download-File -Url $Url -Destination $txzPath -Sha256 $Sha256

Invoke-NativeCommand -FilePath '7z.exe' -ArgumentList @(
  'x', $txzPath, ('-o{0}' -f $tarDir), '-y'
)
$tarFiles = @(Get-ChildItem -LiteralPath $tarDir -Filter '*.tar' -File)
if ($tarFiles.Count -ne 1) {
  throw ('Expected one MSYS2 tar file, found {0}.' -f $tarFiles.Count)
}
Invoke-NativeCommand -FilePath '7z.exe' -ArgumentList @(
  'x', $tarFiles[0].FullName, ('-o{0}' -f $TargetDir), '-y'
)

$msysBin = Join-Path $msysRoot 'usr\bin'
$bashExe = Join-Path $msysBin 'bash.exe'
Add-ToMachinePath -Directory $msysRoot
Add-ToMachinePath -Directory $msysBin

# The first shell initializes the installation and its pacman keyring. MSYS2
# recommends two full upgrades so an updated runtime can be restarted before
# the remaining packages are upgraded.
Invoke-NativeCommand -FilePath $bashExe -ArgumentList @('-lc', ' ')
Invoke-NativeCommand -FilePath $bashExe -ArgumentList @(
  '-lc', 'pacman --noconfirm -Syuu'
)
Invoke-NativeCommand -FilePath $bashExe -ArgumentList @(
  '-lc', 'pacman --noconfirm -Syuu'
)

if ($Packages.Count -gt 0) {
  $packageList = $Packages -join ' '
  Invoke-NativeCommand -FilePath $bashExe -ArgumentList @(
    '-lc', ('pacman --noconfirm --needed -S -- {0}' -f $packageList)
  )
}

# Remove package archives and sync databases from the same image layer that
# populated them. Future package operations must refresh the databases.
Invoke-NativeCommand -FilePath $bashExe -ArgumentList @(
  '-lc', 'pacman --noconfirm -Scc'
)

Remove-Item -LiteralPath $txzPath -Force
Remove-Item -LiteralPath $tarDir -Recurse -Force
