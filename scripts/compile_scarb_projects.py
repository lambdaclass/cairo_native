import os
import subprocess
import shutil
import glob

src_root = "test_data"
dst_root = "test_data_artifacts"


def get_dst_path(src_path):
    rel_path = os.path.relpath(src_path, start=src_root)
    dst_path = os.path.join(dst_root, rel_path)
    dst_path, ext = os.path.splitext(dst_path)

    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    return dst_path


def compile_scarb_project(src_path):
    dst_path = get_dst_path(src_path)

    print(f"compiling project {src_path} into {dst_path}")
    subprocess.run(
        ["scarb", "build"],
        cwd=src_path,
        check=True,
    )

    os.makedirs(dst_path, exist_ok=True)
    for file in glob.glob(os.path.join(src_path, "target", "dev", "*.sierra.json")):
        shutil.copy(file, dst_path)


for dirpath, dirnames, filenames in os.walk(
    os.path.join(src_root, "scarb"), followlinks=True
):
    if os.path.isfile(os.path.join(dirpath, "Scarb.toml")):
        compile_scarb_project(dirpath)
        dirnames.clear()
