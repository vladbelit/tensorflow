# Windows 2022 Docker Image and Toolchain

This image is used by the TensorFlow/XLA Windows 2022 Bazel configurations. Keep
the Dockerfile and checked-in Bazel C++ toolchain snapshots in sync: the toolchain
files hard-code the Visual Studio install path, MSVC toolset directory, Windows
SDK include/lib directories, and LLVM path.

## Current Pins

- Base image: `mcr.microsoft.com/windows/servercore:ltsc2022@sha256:e000e9a1712065a0218447c20ae19984b447fa741d11cf64696b8a1172fcd7da`
- Visual Studio installer: Build Tools bootstrapper, overrideable through `VS_BUILDTOOLS_URL`
- Visual Studio install path: `C:\Program Files\Microsoft Visual Studio\2022\BuildTools`
- C++ workload: `Microsoft.VisualStudio.Workload.VCTools`
- MSVC component: `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`
- MSVC toolset selected by Bazel: `BAZEL_VC_FULL_VERSION=14.44.35207`
- Windows SDK component: `Microsoft.VisualStudio.Component.Windows11SDK.22621`
- Windows SDK path expected by the toolchain: `10.0.22621.0`
- LLVM: `C:\tools\LLVM`, installed from LLVM 18.1.4
- Checked-in Windows 2022 toolchain: update to a new dated snapshot generated
  from this image.

`Microsoft.VisualStudio.Component.VC.Tools.x86.x64` is intentionally omitted. It
means the latest MSVC v143 x64/x86 toolset and can silently move the installed
default compiler. Avoid `--includeRecommended` for the same reason: the Build
Tools C++ workload lists the latest MSVC toolset and latest Windows SDK as
recommended dependencies.

## Updating The Image

1. Pick the new Windows Server Core base image digest.

   ```powershell
   docker buildx imagetools inspect mcr.microsoft.com/windows/servercore:ltsc2022
   ```

   Update the `FROM` digest intentionally. Windows base image updates usually
   include OS security fixes, so this should be a scheduled maintenance action,
   not an accidental tag drift.

2. Pick the Visual Studio Build Tools release.

   The Dockerfile accepts `VS_BUILDTOOLS_URL` as a build argument. The default is
   the Visual Studio 2022 release bootstrapper. For a more reproducible release,
   point this argument at an internally mirrored bootstrapper or a bootstrapper
   from a fixed Visual Studio layout.

3. Pick the MSVC toolset.

   Use a versioned component such as:

   ```text
   Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64
   ```

   Then set `BAZEL_VC_FULL_VERSION` to the exact directory that Visual Studio
   installs under `VC\Tools\MSVC`, for example `14.44.35207`.

4. Pick the Windows SDK.

   Keep `Microsoft.VisualStudio.Component.Windows11SDK.22621` for the initial
   Dockerfile update. If the SDK changes, regenerate the toolchain snapshot in
   the final image and update the Bazel configs at the same time. Do not update
   only the Dockerfile: the generated toolchain embeds paths under
   `Windows Kits\10\Include\<sdk>` and `Windows Kits\10\Lib\<sdk>`.

5. Build and publish the image.

   Use the repository's normal image build/publish flow. After publishing, update
   the `container-image` digest and `cache-silo-key` in:

   ```text
   tensorflow/tools/toolchains/win2022/BUILD
   third_party/xla/tools/toolchains/win2022/BUILD
   ```

## Updating The Bazel Toolchain

Use the existing checked-in snapshot only if the Docker image values still match
the snapshot exactly. With the conventional Build Tools install path, create a
new dated snapshot instead of reusing the existing `20260322` snapshot, because
that snapshot was generated for the Community install path.

- Visual Studio path: `C:\Program Files\Microsoft Visual Studio\2022\BuildTools`
- MSVC toolset: `14.44.35207`
- Windows SDK: `10.0.22621.0`
- LLVM path: `C:\tools\LLVM`

Create a new dated snapshot when any of those paths or versions change, including
the move from the Community path to the Build Tools path.

The helper CLI in this directory handles the repeatable parts of the refresh:

```powershell
python ci\devinfra\docker\windows2022\refresh_toolchain.py --help
```

The examples below assume the current directory is the TensorFlow repository
root.

Run generation inside the final Windows image, not on a developer host. Host
generation can capture the wrong SDK, LLVM path, temp directory, PATH entries, or
Visual Studio installation path.

Generate both WORKSPACE-mode and Bzlmod-mode `local_config_cc` repos with fixed
output bases:

```powershell
python ci\devinfra\docker\windows2022\refresh_toolchain.py generate `
  --repo-root third_party\xla `
  --output-dir C:\tmp\tf-win2022-toolchain `
  --bazel-vs "C:\Program Files\Microsoft Visual Studio\2022\BuildTools" `
  --bazel-vc "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC" `
  --bazel-llvm C:\tools\LLVM `
  --bazel-vc-full-version 14.44.35207 `
  --add-llvm-to-path
```

The command writes logs under `<output-dir>\logs` and copies generated repos
under `<output-dir>\generated`:

```text
generated\workspace_local_config_cc
generated\bzlmod_bazel_tools_local_config_cc
generated\bzlmod_bazel_tools_local_config_cc_toolchains
```

The Bzlmod repo name is intentionally the canonical extension repo name:

```text
@@bazel_tools~cc_configure_extension~local_config_cc
```

Compare the WORKSPACE and Bzlmod generated repos before choosing a source:

```powershell
python ci\devinfra\docker\windows2022\refresh_toolchain.py compare `
  C:\tmp\tf-win2022-toolchain\generated\workspace_local_config_cc `
  C:\tmp\tf-win2022-toolchain\generated\bzlmod_bazel_tools_local_config_cc
```

For the current Windows 2022 toolchain, only these generated files are normally
copied:

```text
BUILD
windows_cc_toolchain_config.bzl
armeabi_cc_toolchain_config.bzl
builtin_include_directory_paths_clangcl
builtin_include_directory_paths_msvc
```

These generated files are normally intentionally omitted:

```text
REPO.bazel
WORKSPACE
get_env.bat
builtin_include_directory_paths_mingw
msys_gcc_installation_error.bat
vc_installation_error_arm.bat
vc_installation_error_arm64.bat
```

Stage the useful files into new dated snapshots:

```powershell
$date = "20260507"
python ci\devinfra\docker\windows2022\refresh_toolchain.py stage `
  --source C:\tmp\tf-win2022-toolchain\generated\workspace_local_config_cc `
  --destination tensorflow\tools\toolchains\win2022\$date `
  --destination third_party\xla\tools\toolchains\win2022\$date
```

If the destination already exists, pass `--force` to overwrite the five staged
files. The command does not delete the destination directory.

Then inspect the staged `*.bzl` diffs against the previous snapshot. Existing
TensorFlow/XLA snapshots may carry local compatibility edits, including explicit
`cc_common` loads and small feature/flag differences in
`windows_cc_toolchain_config.bzl`; preserve those edits unless validation proves
the raw generated files are correct.

Update dated references after the new snapshot is staged:

```powershell
python ci\devinfra\docker\windows2022\refresh_toolchain.py rewrite-refs `
  --old 20241118 `
  --new $date `
  --file .bazelrc `
  --file third_party\xla\tensorflow.bazelrc
```

Do not copy a developer-host snapshot into the Docker/RBE toolchain as-is. Check
that the generated files contain the intended values:

- Visual Studio path: `C:\Program Files\Microsoft Visual Studio\2022\BuildTools`
- MSVC toolset: `14.44.35207`
- Windows SDK: `10.0.22621.0`, unless deliberately bumped
- LLVM path: `C:\tools\LLVM`
- Temp/path details appropriate for the image, for example `C:\TMP`

## Validation

Before relying on the image in CI:

1. Confirm the image contains the expected MSVC `cl.exe` path.
2. Confirm the image contains the expected Windows SDK include/lib paths.
3. Confirm LLVM is installed at `C:\tools\LLVM`.
4. Run a small local Windows Bazel C++ build with `--config=windows_x86_cpu_2022`.
5. Run the RBE Windows config that consumes the published image.
6. Update the RBE image digest and cache key with the Dockerfile/toolchain change.

Primary Microsoft references checked on 2026-05-07:

- `https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2022`
- `https://learn.microsoft.com/en-us/windows/apps/windows-sdk`
