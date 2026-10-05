"""Configure the shared 3dgs.ipynb for local MPS or Google Colab CUDA."""

from pathlib import Path
import importlib
import importlib.util
import sys


def configure_runtime(
    project_root="/content/drive/MyDrive/course_3dgs",
    github_ref="main",
):
    """Return (device, data_root, in_colab) without changing local setup."""
    try:
        from google.colab import drive
    except ImportError:
        import torch
        return torch.device("mps"), Path("."), False

    import json
    import subprocess
    from urllib.parse import quote
    from urllib.request import urlopen

    # Preserve Colab's preinstalled CUDA PyTorch.
    dependencies = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "torchmetrics": "torchmetrics>=1.4,<2",
    }
    missing = [
        package for module, package in dependencies.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Select Runtime → Change runtime type → GPU, reconnect, "
            "and run the adapter cell again."
        )
    device = torch.device("cuda")

    drive.mount("/content/drive")
    data_root = Path(project_root) / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_root}. Pass project_root to configure_runtime."
        )

    # Resolve once, so util.py and rendering.py belong to the same commit.
    repository = "vinneyto/course_3dgs"
    ref_url = (
        f"https://api.github.com/repos/{repository}/commits/"
        f"{quote(github_ref, safe='')}"
    )
    with urlopen(ref_url, timeout=30) as response:
        source_commit = json.load(response)["sha"]
    sources = {}
    for name in ("util.py", "rendering.py"):
        url = f"https://raw.githubusercontent.com/{repository}/{source_commit}/{name}"
        with urlopen(url, timeout=30) as response:
            sources[name] = response.read()

    code_root = Path("/content/course_3dgs_modules")
    code_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (code_root / name).write_bytes(source)
    code_path = str(code_root)
    if code_path in sys.path:
        sys.path.remove(code_path)
    sys.path.insert(0, code_path)
    # Rerunning after a ref change must not use cached or old Drive modules.
    for name in ("rendering", "util"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("GitHub modules:", github_ref, source_commit)
    print("Device:", device)
    print("Data root:", data_root)
    return device, data_root, True
