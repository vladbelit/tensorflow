# Windows Server 2022 build image

This directory builds the Windows image used by TensorFlow build and test
jobs. It starts from Windows Server Core and bootstraps a versioned,
SHA-verified PowerShell LTS installation before running the remaining scripts.

## Build and verify

Docker must be running in Windows-container mode on a host compatible with
Windows Server 2022 containers. The explicit process-isolation setting keeps
the build mode consistent across Windows Server and Windows client hosts. From
the TensorFlow repository root, run:

```powershell
docker build `
  --isolation=process `
  --tag tensorflow-windows2022:dev `
  ci/devinfra/docker/windows2022
```

The final build layer runs a verification script that sanity-checks the state
of the installed tooling. It can also be run against an existing image:

```powershell
docker run --rm tensorflow-windows2022:dev `
  C:\tools\verify_install.ps1
```

With no command, the entrypoint opens PowerShell. Select other shells
explicitly:

```powershell
docker run --rm -it tensorflow-windows2022:dev
docker run --rm tensorflow-windows2022:dev cmd.exe /d /c python --version
docker run --rm tensorflow-windows2022:dev `
  C:\tools\msys64\usr\bin\bash.exe -c 'python --version'
```

At container startup, the entrypoint routes the GCE metadata server address
through the container's default gateway when one is available. Networkless
containers skip this setup.

## Transitional Windows toolchains

TODO(belitskiy): Remove the compatibility stack once all CI consumers use the
new compiler stack.

The image temporarily carries two complete compiler stacks so the runner and
RBE image digests can move before TensorFlow/XLA/JAX select the new toolchain:

- The compatibility stack retains Visual Studio Community/MSVC 14.42.34433,
  Windows SDK 10.0.22621.0, and LLVM 18.1.4 at the paths captured by the
  `20241118` toolchain snapshot.
- The current stack provides Visual Studio Build Tools/MSVC 14.44.35207,
  Windows SDK 10.0.26100.0, and LLVM 19.1.7 under
  `C:\tools\LLVM-19.1.7`.

Unqualified `clang` and `BAZEL_LLVM` intentionally continue to select LLVM 18
for compatibility.
Dated Bazel toolchain snapshots select all compiler, header, library, and SDK
paths explicitly, so the LLVM 19 directory does not need to be on the machine
PATH.

## Python layout

Python 3.10 through 3.15 are installed under matching dotted directories such
as `C:\Python3.12`; these paths are part of the CI interface. Each interpreter
has a `python3.x.exe` alias. Python 3.14 is the default and provides `python`,
`python3`, `py`, and `pip`. Other Python Scripts directories remain off PATH
to avoid ambiguous unqualified commands.

## Cache and reproducibility

Each installer is copied immediately before its `RUN` instruction. This keeps
downloads and cleanup in one layer while allowing a late script edit to reuse
expensive earlier installations. A change to `common.ps1` intentionally
invalidates every installer layer.

Inspect image history and size with:

```powershell
docker history tensorflow-windows2022:dev
docker image inspect tensorflow-windows2022:dev `
  --format '{{.Id}} {{.Size}}'
```

Versioned downloads require checked-in SHA-256 hashes. Four inputs remain
intentionally serviced or rolling:

- The LTSC 2022 Server Core base tag receives Windows servicing.
- The two hashed Visual Studio bootstrappers select serviced VS 2022
  components.
- The fixed MSYS2 archive is upgraded from live signed repositories.

## Updating a tool

1. Select a versioned upstream artifact instead of a moving download URL.
2. Download it and calculate its checksum:

   ```powershell
   Invoke-WebRequest -Uri $url -OutFile $file
   (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
   ```

3. Update the version or URL and its checksum together. Update the verifier
   only if a command or path contract changes. Python updates must retain the
   dotted CI directory layout.
4. Parse every PowerShell script and run `git diff --check`.
5. Build the image, run the verifier, and exercise PowerShell, cmd, CI-style
   Bash, and interactive Bash entry paths.
6. Record the build command, image ID, size, and smoke-test results.

For Visual Studio, hash the final target of the evergreen bootstrapper URL;
this gates bootstrapper changes but does not freeze its serviced component
channel. For MSYS2, do not disable package signature verification to work
around initialization or mirror failures.

## Publishing and consuming an image

A local build does not affect CI. Consumers pin published image digests in
`ci/official/envs/windows_x86_2022` and
`tensorflow/tools/toolchains/win2022/BUILD`.

The dated Bazel toolchain snapshots under
`tensorflow/tools/toolchains/win2022` hard-code the Visual Studio path, MSVC
toolset, Windows SDK, and LLVM paths. If any of those change, generate and
validate a matching snapshot and update `.bazelrc` as part of the image
rollout. When updating the RBE image digest, also change its `cache-silo-key`
so results from different toolchain images cannot share a remote cache silo.
