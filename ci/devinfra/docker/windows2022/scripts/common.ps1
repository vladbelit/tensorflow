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
$PSNativeCommandUseErrorActionPreference = $true
$ProgressPreference = 'SilentlyContinue'

# Downloads a file and verifies its required SHA-256 checksum.
function Download-File {
  param (
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$Sha256
  )

  $partialPath = $Destination + '.partial'
  Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue

  Write-Output ('Downloading {0}' -f $Url)
  Invoke-WebRequest -Uri $Url -OutFile $partialPath -MaximumRetryCount 3 `
    -RetryIntervalSec 5

  $actualHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash
  if ($actualHash -ne $Sha256) {
    $message = ("SHA-256 verification failed for '{0}'.`n" +
      "  Expected: {1}`n  Actual:   {2}") -f `
      $Destination, $Sha256.ToLowerInvariant(), `
      $actualHash.ToLowerInvariant()
    throw $message
  }

  Move-Item -LiteralPath $partialPath -Destination $Destination -Force
  Write-Output ('Verified SHA-256: {0}' -f $actualHash.ToLowerInvariant())
}

# Runs a native process without flattening its argument boundaries.
function Invoke-NativeCommand {
  param (
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string []]$ArgumentList = @(),

    [int []]$SuccessExitCodes = @(0)
  )

  Write-Output ('Running: {0} {1}' -f $FilePath, ($ArgumentList -join ' '))
  # Inspect the exit code ourselves so callers can accept codes such as the
  # MSI reboot-required result, 3010. The pipeline makes PowerShell wait for
  # GUI-subsystem executables such as msiexec.exe before reading LASTEXITCODE.
  $PSNativeCommandUseErrorActionPreference = $false
  & $FilePath @ArgumentList | Write-Output
  $exitCode = $LASTEXITCODE

  if ($SuccessExitCodes -notcontains $exitCode) {
    throw ('{0} failed with exit code {1}.' -f $FilePath, $exitCode)
  }
}

# Runs a command and checks its combined output against a regular expression.
function Assert-CommandVersion {
  param (
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string []]$ArgumentList = @(),

    [Parameter(Mandatory = $true)]
    [string]$ExpectedPattern
  )

  $PSNativeCommandUseErrorActionPreference = $false
  $output = (& $FilePath @ArgumentList 2>&1 | Out-String).Trim()
  $exitCode = $LASTEXITCODE

  if ($exitCode -ne 0) {
    throw ("{0} failed with exit code {1}.`n{2}" -f $FilePath, $exitCode, $output)
  }
  if ($output -notmatch $ExpectedPattern) {
    $message = ("Unexpected output from {0}.`n" +
      "  Expected pattern: {1}`n  Actual output:`n{2}") -f `
      $FilePath, $ExpectedPattern, $output
    throw $message
  }

  Write-Output $output
}

# Adds a directory to both the machine and current-process PATH values.
function Add-ToMachinePath {
  param (
    [Parameter(Mandatory = $true)]
    [string]$Directory,

    [switch]$Prepend
  )

  $cleanDirectory = $Directory.TrimEnd('\')
  foreach ($target in @(
      [EnvironmentVariableTarget]::Machine
      [EnvironmentVariableTarget]::Process
    )) {
    $currentPath = [Environment]::GetEnvironmentVariable('PATH', $target)
    $parts = @($currentPath -split ';' | Where-Object { $_ })
    $otherParts = @($parts | Where-Object {
        $_.TrimEnd('\') -ine $cleanDirectory
      })

    if (-not $Prepend -and $otherParts.Count -ne $parts.Count) {
      continue
    }
    if ($Prepend) {
      $updatedPath = (@($Directory) + $otherParts) -join ';'
    }
    else {
      $updatedPath = ($otherParts + @($Directory)) -join ';'
    }
    if ($updatedPath -ne $currentPath) {
      [Environment]::SetEnvironmentVariable('PATH', $updatedPath, $target)
    }
  }
}
