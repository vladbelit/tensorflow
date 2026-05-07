"""Helpers for refreshing TensorFlow's Windows 2022 Bazel toolchain."""

import argparse
import filecmp
import os
from pathlib import Path
import shutil
import subprocess
import sys


TOOLCHAIN_FILES = (
    'BUILD',
    'windows_cc_toolchain_config.bzl',
    'armeabi_cc_toolchain_config.bzl',
    'builtin_include_directory_paths_clangcl',
    'builtin_include_directory_paths_msvc',
)

OMITTED_GENERATED_FILES = (
    'REPO.bazel',
    'WORKSPACE',
    'get_env.bat',
    'builtin_include_directory_paths_mingw',
    'msys_gcc_installation_error.bat',
    'vc_installation_error_arm.bat',
    'vc_installation_error_arm64.bat',
)

BZLMOD_LOCAL_CONFIG_CC = (
    'external',
    'bazel_tools~cc_configure_extension~local_config_cc',
)

BZLMOD_LOCAL_CONFIG_CC_TOOLCHAINS = (
    'external',
    'bazel_tools~cc_configure_extension~local_config_cc_toolchains',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate, compare, and stage Windows @local_config_cc snapshots.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    generate = subparsers.add_parser(
        'generate',
        help='Generate WORKSPACE and Bzlmod local_config_cc repos.',
    )
    generate.add_argument('--repo-root', type=Path, default=Path.cwd())
    generate.add_argument('--output-dir', type=Path, required=True)
    generate.add_argument('--bazel', default='bazel')
    generate.add_argument(
        '--workspace-output-base',
        type=Path,
        help='Defaults to <output-dir>/_bazel_output_base_workspace.',
    )
    generate.add_argument(
        '--bzlmod-output-base',
        type=Path,
        help='Defaults to <output-dir>/_bazel_output_base_bzlmod_cc_configure.',
    )
    generate.add_argument(
        '--bazel-vs',
        default=r'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
    )
    generate.add_argument(
        '--bazel-vc',
        default=r'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC',
    )
    generate.add_argument('--bazel-llvm', default=r'C:\tools\LLVM')
    generate.add_argument('--bazel-sh', default='')
    generate.add_argument('--bazel-vc-full-version', default='14.44.35207')
    generate.add_argument('--add-llvm-to-path', action='store_true')
    generate.set_defaults(func=generate_command)

    compare = subparsers.add_parser(
        'compare',
        help='Compare the toolchain-useful files in two generated repos.',
    )
    compare.add_argument('left', type=Path)
    compare.add_argument('right', type=Path)
    compare.set_defaults(func=compare_command)

    stage = subparsers.add_parser(
        'stage',
        help='Copy the toolchain-useful files into one or more dated snapshots.',
    )
    stage.add_argument('--source', type=Path, required=True)
    stage.add_argument(
        '--destination',
        type=Path,
        action='append',
        required=True,
        help='Snapshot directory to update. Repeat for TF and vendored XLA.',
    )
    stage.add_argument('--force', action='store_true')
    stage.add_argument('--dry-run', action='store_true')
    stage.set_defaults(func=stage_command)

    rewrite_refs = subparsers.add_parser(
        'rewrite-refs',
        help='Rewrite win2022/<old> references to win2022/<new>.',
    )
    rewrite_refs.add_argument('--old', required=True)
    rewrite_refs.add_argument('--new', required=True)
    rewrite_refs.add_argument('--file', type=Path, action='append', required=True)
    rewrite_refs.add_argument('--dry-run', action='store_true')
    rewrite_refs.set_defaults(func=rewrite_refs_command)

    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def assert_child_path(parent: Path, child: Path) -> None:
    parent = resolved(parent)
    child = resolved(child)
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f'Refusing to touch path outside {parent}: {child}') from exc


def reset_child_dir(parent: Path, child: Path) -> None:
    assert_child_path(parent, child)
    if child.exists():
        shutil.rmtree(child)
    child.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(source: Path, destination: Path, parent: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f'Generated repository not found: {source}')
    reset_child_dir(parent, destination)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def run_bazel(
    args: list[str],
    repo_root: Path,
    env: dict[str, str],
    log_path: Path,
) -> int:
    with log_path.open('w', encoding='utf-8') as log_file:
        result = subprocess.run(
            args,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def build_generation_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env['BAZEL_VS'] = args.bazel_vs
    env['BAZEL_VC'] = args.bazel_vc
    env['BAZEL_LLVM'] = args.bazel_llvm
    if args.bazel_sh:
        env['BAZEL_SH'] = args.bazel_sh
    if args.bazel_vc_full_version:
        env['BAZEL_VC_FULL_VERSION'] = args.bazel_vc_full_version
    if args.add_llvm_to_path:
        env['PATH'] = str(Path(args.bazel_llvm) / 'bin') + os.pathsep + env['PATH']
    return env


def generate_command(args: argparse.Namespace) -> int:
    repo_root = resolved(args.repo_root)
    output_dir = resolved(args.output_dir)
    logs_dir = output_dir / 'logs'
    generated_dir = output_dir / 'generated'
    workspace_output_base = resolved(
        args.workspace_output_base or output_dir / '_bazel_output_base_workspace'
    )
    bzlmod_output_base = resolved(
        args.bzlmod_output_base or output_dir / '_bazel_output_base_bzlmod_cc_configure'
    )

    logs_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    env = build_generation_env(args)

    workspace_log = logs_dir / 'workspace_sync.log'
    workspace_args = [
        args.bazel,
        f'--output_base={workspace_output_base}',
        'sync',
        '--configure',
        '--only=local_config_cc',
    ]
    workspace_exit = run_bazel(workspace_args, repo_root, env, workspace_log)
    if workspace_exit != 0:
        raise RuntimeError(
            'WORKSPACE local_config_cc generation failed with exit code '
            f'{workspace_exit}; see {workspace_log}'
        )

    bzlmod_log = logs_dir / 'bzlmod_local_config_cc_fetch.log'
    bzlmod_args = [
        args.bazel,
        f'--output_base={bzlmod_output_base}',
        'fetch',
        '--config=bzlmod',
        '@@bazel_tools~cc_configure_extension~local_config_cc//...',
    ]
    bzlmod_exit = run_bazel(bzlmod_args, repo_root, env, bzlmod_log)

    workspace_source = workspace_output_base / 'external' / 'local_config_cc'
    bzlmod_source = bzlmod_output_base.joinpath(*BZLMOD_LOCAL_CONFIG_CC)
    bzlmod_toolchains_source = bzlmod_output_base.joinpath(
        *BZLMOD_LOCAL_CONFIG_CC_TOOLCHAINS
    )
    if not bzlmod_source.exists():
        raise FileNotFoundError(
            'Bzlmod local_config_cc was not generated; see '
            f'{bzlmod_log} and {bzlmod_output_base}'
        )

    copy_tree_contents(
        workspace_source,
        generated_dir / 'workspace_local_config_cc',
        generated_dir,
    )
    copy_tree_contents(
        bzlmod_source,
        generated_dir / 'bzlmod_bazel_tools_local_config_cc',
        generated_dir,
    )
    if bzlmod_toolchains_source.exists():
        copy_tree_contents(
            bzlmod_toolchains_source,
            generated_dir / 'bzlmod_bazel_tools_local_config_cc_toolchains',
            generated_dir,
        )

    paths_file = generated_dir / 'generation_paths.txt'
    paths_file.write_text(
        '\n'.join(
            [
                f'REPO_ROOT={repo_root}',
                f'WORKSPACE_OUTPUT_BASE={workspace_output_base}',
                f'WORKSPACE_LOCAL_CONFIG_CC={workspace_source}',
                f'WORKSPACE_EXIT_CODE={workspace_exit}',
                f'BZLMOD_OUTPUT_BASE={bzlmod_output_base}',
                f'BZLMOD_LOCAL_CONFIG_CC={bzlmod_source}',
                f'BZLMOD_LOCAL_CONFIG_CC_TOOLCHAINS={bzlmod_toolchains_source}',
                f'BZLMOD_EXIT_CODE={bzlmod_exit}',
                '',
            ]
        ),
        encoding='utf-8',
    )

    print(f'Generated WORKSPACE repo: {generated_dir / "workspace_local_config_cc"}')
    if bzlmod_source.exists():
        print(
            'Generated Bzlmod repo: '
            f'{generated_dir / "bzlmod_bazel_tools_local_config_cc"}'
        )
    if bzlmod_exit != 0:
        print(
            f'Bzlmod fetch exited non-zero after generation; inspect {bzlmod_log}',
            file=sys.stderr,
        )
    return 0


def compare_command(args: argparse.Namespace) -> int:
    left = resolved(args.left)
    right = resolved(args.right)
    had_difference = False
    for relative in TOOLCHAIN_FILES:
        left_file = left / relative
        right_file = right / relative
        if not left_file.exists() or not right_file.exists():
            print(f'MISSING {relative}: {left_file.exists()} {right_file.exists()}')
            had_difference = True
        elif filecmp.cmp(left_file, right_file, shallow=False):
            print(f'SAME    {relative}')
        else:
            print(f'DIFF    {relative}')
            had_difference = True
    return 1 if had_difference else 0


def stage_command(args: argparse.Namespace) -> int:
    source = resolved(args.source)
    missing = [name for name in TOOLCHAIN_FILES if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            'Source is missing toolchain files: ' + ', '.join(missing)
        )

    destinations = [resolved(destination_arg) for destination_arg in args.destination]
    for destination in destinations:
        if destination.exists() and not args.force:
            raise FileExistsError(
                f'{destination} exists; pass --force to overwrite toolchain files'
            )

    for destination in destinations:
        print(f'Staging toolchain files into {destination}')
        if not args.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        for relative in TOOLCHAIN_FILES:
            source_file = source / relative
            destination_file = destination / relative
            print(f'  {source_file} -> {destination_file}')
            if not args.dry_run:
                shutil.copy2(source_file, destination_file)

    print('Omitted generated files:')
    for relative in OMITTED_GENERATED_FILES:
        print(f'  {relative}')
    print(
        'Review *.bzl diffs against the previous snapshot before committing; '
        'TensorFlow/XLA snapshots may carry local compatibility edits.'
    )
    return 0


def rewrite_refs_command(args: argparse.Namespace) -> int:
    old = f'win2022/{args.old}'.encode('utf-8')
    new = f'win2022/{args.new}'.encode('utf-8')
    changed = False

    for file_arg in args.file:
        path = resolved(file_arg)
        data = path.read_bytes()
        count = data.count(old)
        if count == 0:
            print(f'No refs to rewrite in {path}')
            continue
        print(f'Rewriting {count} ref(s) in {path}')
        changed = True
        if not args.dry_run:
            path.write_bytes(data.replace(old, new))

    return 0 if changed else 1


def main() -> int:
    args = parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
