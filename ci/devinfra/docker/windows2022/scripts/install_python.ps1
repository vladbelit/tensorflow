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
  [Parameter(Mandatory = $true)]
  [string]$Version,

  [Parameter(Mandatory = $true)]
  [string]$TargetDir,

  [Parameter(Mandatory = $true)]
  [string]$Sha256,

  [string]$ReleaseDirectory = $Version,  # Upstream directory for prereleases.

  [string []]$PipPackages = @(
    'packaging==26.2',
    'setuptools==83.0.0'
  ),

  [string []]$Aliases = @(),

  [switch]$IncludeFreeThreaded,  # Install the versioned free-threaded binary.

  [switch]$InstallLauncher,  # Install the shared py.exe launcher.

  [switch]$PrependToPath  # Make this the default Python on PATH.
)

. "$PSScriptRoot\common.ps1"

foreach ($alias in $Aliases) {
  if ([IO.Path]::GetFileName($alias) -ne $alias -or `
      $alias -notmatch '^python[0-9.]*\.exe$') {
    throw ('Invalid Python executable alias: {0}' -f $alias)
  }
}

Write-Output ('Installing Python {0} to {1}...' -f $Version, $TargetDir)
$url = ('https://www.python.org/ftp/python/{0}/python-{1}-amd64.exe' -f `
    $ReleaseDirectory, $Version)
$installerPath = Join-Path $env:TEMP ('python-{0}-amd64.exe' -f $Version)
$launcherValue = if ($InstallLauncher) { 1 } else { 0 }

Download-File -Url $url -Destination $installerPath -Sha256 $Sha256
$installerArguments = @(
  '/quiet',
  'InstallAllUsers=1',
  ('TargetDir={0}' -f $TargetDir),
  'PrependPath=0',
  'AssociateFiles=0',
  'CompileAll=0',
  'Include_debug=0',
  'Include_dev=1',
  'Include_doc=0',
  'Include_exe=1',
  ('Include_launcher={0}' -f $launcherValue),
  'Include_lib=1',
  'Include_pip=1',
  'Include_symbols=0',
  'Include_tcltk=0',
  'Include_test=0',
  'Include_tools=1',
  ('InstallLauncherAllUsers={0}' -f $launcherValue),
  'Shortcuts=0'
)
if ($IncludeFreeThreaded) {
  $installerArguments += 'Include_freethreaded=1'
}
Invoke-NativeCommand -FilePath $installerPath `
  -ArgumentList $installerArguments

$pythonExe = Join-Path $TargetDir 'python.exe'
Assert-CommandVersion -FilePath $pythonExe -ArgumentList @('--version') `
  -ExpectedPattern ('^Python {0}$' -f [regex]::Escape($Version))
Assert-CommandVersion -FilePath $pythonExe `
  -ArgumentList @('-m', 'pip', '--version') -ExpectedPattern '^pip '

if ($IncludeFreeThreaded) {
  $versionParts = $Version.Split('.')
  $freeThreadedExe = Join-Path $TargetDir `
    ('python{0}.{1}t.exe' -f $versionParts[0], $versionParts[1])
  Assert-CommandVersion -FilePath $freeThreadedExe -ArgumentList @('-VV') `
    -ExpectedPattern ('(?s)^Python {0}.*free-threading build' -f `
      [regex]::Escape($Version))
  Assert-CommandVersion -FilePath $freeThreadedExe `
    -ArgumentList @('-m', 'pip', '--version') -ExpectedPattern '^pip '
}

if ($PipPackages.Count -gt 0) {
  Write-Output ('Installing {0}' -f ($PipPackages -join ', '))
  $pipArguments = @(
    '-m', 'pip', 'install', '--disable-pip-version-check',
    '--no-cache-dir', '--upgrade'
  ) + $PipPackages
  Invoke-NativeCommand -FilePath $pythonExe -ArgumentList $pipArguments
}

foreach ($alias in $Aliases) {
  $aliasPath = Join-Path $TargetDir $alias
  New-Item -ItemType HardLink -Path $aliasPath -Target $pythonExe | Out-Null
}

$scriptsDir = Join-Path $TargetDir 'Scripts'
Add-ToMachinePath -Directory $TargetDir -Prepend:$PrependToPath
if ($PrependToPath) {
  # CPython's PrependPath=1 exposes both the interpreter and its console
  # scripts. Reproduce that contract only for the selected default Python.
  Add-ToMachinePath -Directory $scriptsDir -Prepend
}

Remove-Item -LiteralPath $installerPath -Force
