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

# This Dockerfile creates a Windows Server 2022 build image with:
# - PowerShell 7.4.17 (LTS)
# - Visual Studio 2022 Build Tools and the Windows 11 26100 SDK
# - LLVM/Clang 18.1.4
# - MSYS2 plus curl, git, patch, unzip, vim, wget, and zip
# - Python 3.10 through 3.15, with Python 3.14 as the PATH default
# - Azul Zulu JDK 25.0.3
# - Bazelisk 1.29.0
# - uv 0.11.28
# - Google Cloud CLI 575.0.1

# The .NET SDK base includes PowerShell 7 but is ~600 MB larger, so install
# PowerShell explicitly below.
FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';"]

# Enable long paths
RUN New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

RUN New-Item -ItemType Directory -Path C:\TEMP,C:\TMP,C:\setup -Force | Out-Null
ENV TMP=C:/TMP
ENV TEMP=C:/TEMP

COPY scripts/bootstrap_powershell.ps1 C:/setup/
RUN C:\setup\bootstrap_powershell.ps1

SHELL ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop'; $PSNativeCommandUseErrorActionPreference=$true; $ProgressPreference='SilentlyContinue';"]

# Common behavior intentionally invalidates every installer layer when changed.
COPY scripts/common.ps1 scripts/install_7zip.ps1 C:/setup/
RUN C:\setup\install_7zip.ps1

COPY scripts/install_vs_build_tools.ps1 C:/setup/
RUN C:\setup\install_vs_build_tools.ps1

COPY scripts/install_llvm.ps1 C:/setup/
RUN C:\setup\install_llvm.ps1

# Disable path conversion under MSYS - it tends to cause issues rather than solve them.
ENV MSYS_NO_PATHCONV=1
ENV MSYS2_ARG_CONV_EXCL=*
COPY scripts/install_msys2.ps1 C:/setup/
RUN C:\setup\install_msys2.ps1

COPY scripts/install_jdk.ps1 C:/setup/
RUN C:\setup\install_jdk.ps1

COPY scripts/install_bazelisk.ps1 C:/setup/
RUN C:\setup\install_bazelisk.ps1

COPY scripts/install_python.ps1 C:/setup/
RUN C:\setup\install_python.ps1 -Version '3.10.11' -Aliases @('python3.10.exe') -TargetDir 'C:\Python3.10' -Sha256 'd8dede5005564b408ba50317108b765ed9c3c510342a598f9fd42681cbe0648b'
RUN C:\setup\install_python.ps1 -Version '3.11.9' -Aliases @('python3.11.exe') -TargetDir 'C:\Python3.11' -Sha256 '5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde'
RUN C:\setup\install_python.ps1 -Version '3.12.10' -Aliases @('python3.12.exe') -TargetDir 'C:\Python3.12' -Sha256 '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
RUN C:\setup\install_python.ps1 -Version '3.13.14' -Aliases @('python3.13.exe') -TargetDir 'C:\Python3.13' -Sha256 'c54d9b9bbb8a36e6489363ddd01139707fd781d72f1f9e90c7ec65d0061368e0'
RUN C:\setup\install_python.ps1 -Version '3.14.6' -Aliases @('python3.exe', 'python3.14.exe') -PrependToPath -TargetDir 'C:\Python3.14' -InstallLauncher -Sha256 '14b3e9a710a3fcf0bd9b55ab6b60412bd91227563f813fc49040cabc0209e0bd'
RUN C:\setup\install_python.ps1 -Version '3.15.0b3' -Aliases @('python3.15.exe') -TargetDir 'C:\Python3.15' -ReleaseDirectory '3.15.0' -Sha256 '8002dbdf7277964a3c50f5deef090dfcd0769647a35aec97999f63d5f2903d38'
ENV PY_PYTHON=3.14

COPY scripts/install_uv.ps1 C:/setup/
RUN C:\setup\install_uv.ps1

ENV CLOUDSDK_PYTHON=C:/Python3.14/python.exe
COPY scripts/install_gcloud.ps1 C:/setup/
RUN C:\setup\install_gcloud.ps1

# Bazel environment variables use forward slashes to avoid Dockerfile escaping;
# Windows tools accept both separator forms.
ENV BAZEL_LLVM=C:/tools/LLVM
ENV BAZEL_SH=C:/tools/msys64/usr/bin/bash.exe
ENV BAZEL_VS="C:/Program Files/Microsoft Visual Studio/2022/BuildTools"
ENV BAZEL_VC="C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC"

# Map T: to C:\drive_t. Internal CI uses T:, but Docker volume mounts require
# a real C: path. The mapping lets `docker exec` consume unmodified T:\ paths;
# `docker cp` does not honor it.
RUN New-Item -ItemType directory -Path C:\drive_t; \
    New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\DOS Devices' -Name 'T:' -Value '\??\C:\drive_t' -PropertyType String;

COPY scripts/entrypoint.ps1 scripts/verify_install.ps1 C:/setup/
RUN Copy-Item -Path C:\setup\common.ps1,C:\setup\entrypoint.ps1,C:\setup\verify_install.ps1 -Destination C:\tools; C:\tools\verify_install.ps1; Remove-Item -LiteralPath C:\setup -Recurse -Force

# With no command, the entrypoint opens PowerShell. Callers that need cmd.exe or
# MSYS2 Bash pass that executable explicitly; see the adjacent README.
ENTRYPOINT ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\\tools\\entrypoint.ps1"]
