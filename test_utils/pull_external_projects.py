import tempfile
import subprocess
import sys
import shutil
import os

if len(sys.argv) < 2:
    print("Expected Cairo revision as first argument")
    exit(1)
revision = sys.argv[1]

with tempfile.TemporaryDirectory() as root:
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/starkware-libs/cairo",
            "--revision",
            revision,
            root,
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    shutil.rmtree("corelib", ignore_errors=True)
    shutil.copytree(
        os.path.join(root, "corelib"),
        "corelib",
    )
    shutil.rmtree("test_data/tests_starknet/bug_samples", ignore_errors=True)
    shutil.copytree(
        os.path.join(root, "tests/bug_samples"),
        "test_data/tests_starknet/bug_samples",
    )
    shutil.rmtree("test_data/tests_starknet/cairo_level_tests", ignore_errors=True)
    shutil.copytree(
        os.path.join(root, "crates/cairo-lang-starknet/cairo_level_tests"),
        "test_data/tests_starknet/cairo_level_tests",
    )
