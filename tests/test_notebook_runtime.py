"""Test environment routing without requiring PyTorch, a GPU, or Drive access."""

import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notebook_runtime", ROOT / "notebook_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fake_modules(colab=False, cuda=True):
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    google = types.ModuleType("google")
    google.__path__ = []
    modules = {"torch": torch, "google": google, "google.colab": None}
    if colab:
        colab_module = types.ModuleType("google.colab")
        colab_module.drive = types.SimpleNamespace(mount=Mock())
        google.colab = colab_module
        modules["google.colab"] = colab_module
    return modules


class NotebookRuntimeTests(unittest.TestCase):
    def test_local_keeps_mps_relative_paths_and_performs_no_external_setup(self):
        with patch.dict(sys.modules, fake_modules()), \
             patch("subprocess.check_call") as install, \
             patch("urllib.request.urlopen") as download:
            self.assertEqual(runtime.configure_runtime(), ("mps", Path("."), False))
            install.assert_not_called()
            download.assert_not_called()

    def test_colab_requires_gpu_before_mounting_drive(self):
        modules = fake_modules(colab=True, cuda=False)
        with patch.dict(sys.modules, modules), \
             patch("importlib.util.find_spec", return_value=object()), \
             patch("urllib.request.urlopen") as download:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                runtime.configure_runtime()
            modules["google.colab"].drive.mount.assert_not_called()
            download.assert_not_called()

    def test_missing_drive_data_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, fake_modules(colab=True)), \
                 patch("importlib.util.find_spec", return_value=object()), \
                 patch("urllib.request.urlopen") as download:
                with self.assertRaisesRegex(FileNotFoundError, "project_root"):
                    runtime.configure_runtime(project_root=directory)
                download.assert_not_called()

    def test_colab_downloads_same_commit_modules_and_refreshes_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            (project / "data").mkdir(parents=True)
            code = Path(directory) / "modules"
            modules = fake_modules(colab=True)
            modules.update({"util": types.ModuleType("util"),
                            "rendering": types.ModuleType("rendering")})
            urls = []

            def download(url, timeout):
                self.assertEqual(timeout, 30)
                urls.append(url)
                if "/commits/" in url:
                    return io.BytesIO(b'{"sha": "commit123"}')
                return io.BytesIO(f"# {url.rsplit('/', 1)[-1]}".encode())

            def path(value):
                return code if str(value) == "/content/course_3dgs_modules" else Path(value)

            def find_spec(module):
                return None if module == "torchmetrics" else object()

            with patch.dict(sys.modules, modules), \
                 patch.object(runtime, "Path", side_effect=path), \
                 patch("importlib.util.find_spec", side_effect=find_spec), \
                 patch("urllib.request.urlopen", side_effect=download), \
                 patch("subprocess.check_call") as install, \
                 patch.object(sys, "path", list(sys.path)):
                for _ in range(2):
                    result = runtime.configure_runtime(project, github_ref="feat/test")
                    self.assertEqual(result, ("cuda", project / "data", True))
                    self.assertNotIn("util", sys.modules)
                    self.assertNotIn("rendering", sys.modules)
                self.assertEqual(sys.path[0], str(code))
                self.assertEqual(sys.path.count(str(code)), 1)
                install.assert_called_with([
                    sys.executable, "-m", "pip", "install", "-q", "torchmetrics>=1.4,<2",
                ])
            self.assertTrue(urls[0].endswith("/commits/feat%2Ftest"))
            self.assertTrue(urls[1].endswith("/commit123/util.py"))
            self.assertTrue(urls[2].endswith("/commit123/rendering.py"))
            self.assertEqual((code / "util.py").read_text(), "# util.py")
            modules["google.colab"].drive.mount.assert_called_with("/content/drive")

    def test_first_cell_bootstraps_colab_adapter_from_selected_ref(self):
        notebook = json.loads((ROOT / "3dgs.ipynb").read_text())
        source = "".join(notebook["cells"][0]["source"])
        adapter = b'def configure_runtime(github_ref):\n return "cuda", "drive-data", True\n'
        with patch.dict(sys.modules, fake_modules(colab=True)), \
             patch("urllib.request.urlopen", return_value=io.BytesIO(adapter)) as download:
            namespace = {}
            exec(compile(source, "adapter-cell", "exec"), namespace)
        self.assertEqual(namespace["device"], "cuda")
        self.assertTrue(namespace["IN_COLAB"])
        self.assertIn("/main/notebook_runtime.py", download.call_args.args[0])

    def test_notebook_python_syntax(self):
        notebook = json.loads((ROOT / "3dgs.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")


if __name__ == "__main__":
    unittest.main()
