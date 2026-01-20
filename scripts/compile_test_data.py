import os
import subprocess
import argparse

src_root = "test_data"
dst_root = "test_data_artifacts"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Defines which cairo files to compile")
args = parser.parse_args()

subprocess.run(
    [
        "cargo",
        "build",
        "--package",
        "test-utils",
    ],
    check=True,
)


def get_dst_path(src_path):
    rel_path = os.path.relpath(src_path, start=src_root)
    dst_path = os.path.join(dst_root, rel_path)
    dst_path, ext = os.path.splitext(dst_path)

    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    return dst_path


def compile_cairo_project(src_path):
    dst_path = get_dst_path(src_path)

    print(f"compiling project {src_path} into {dst_path}")
    subprocess.run(
        [
            "target/debug/compile-cairo-project",
            src_path,
            dst_path + ".sierra.json",
            dst_path + ".sierra",
        ],
        check=True,
    )


def compile_cairo_tests(src_path, starknet=False):
    dst_path = get_dst_path(src_path)

    print(f"compiling tests {src_path} into {dst_path}")
    args = [
        "target/debug/compile-cairo-tests",
        src_path,
        dst_path + ".tests.json",
    ]
    if starknet:
        args.append("--starknet")
    subprocess.run(args, check=True)


def compile_cairo_contract(src_path):
    dst_path = get_dst_path(src_path)

    print(f"compiling contract {src_path} into {dst_path}")
    subprocess.run(
        [
            "target/debug/compile-cairo-contract",
            src_path,
            dst_path + ".contract.json",
        ],
        check=True,
    )


def walk(subdir, f):
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(src_root, subdir), followlinks=True
    ):
        if os.path.isfile(os.path.join(dirpath, "cairo_project.toml")):
            f(dirpath)
            dirnames.clear()
        else:
            for filename in filenames:
                if filename.endswith(".cairo"):
                    filepath = os.path.join(dirpath, filename)
                    f(filepath)

 
if args.mode == "sierra-emu":
    walk("programs/debug_utils", lambda p: compile_cairo_project(p))
else:
    walk("programs", lambda p: compile_cairo_project(p))
    walk("contracts", lambda p: compile_cairo_contract(p))
    walk("tests", lambda p: compile_cairo_tests(p, starknet=False))
    walk("tests_starknet", lambda p: compile_cairo_tests(p, starknet=True))
