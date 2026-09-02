"""Static checks for the compact Colab experiment runner."""

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "colab"
    / "run_experiment.ipynb"
)


def _load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_code_cells_are_valid_python():
    notebook = _load()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_colab_runner_locks_the_requested_protocol():
    source = "\n".join(
        "".join(cell["source"]) for cell in _load()["cells"]
    )
    assert '["mrl", "ese", "mipic", "gsr"]' in source
    assert '["bert", "tinybert_6l", "bgem3", "qwen3_0.6b"]' in source
    assert "[Research Space]/[ICLR] Embedding MRL" in source
    assert "data.train_file=train/final_data.csv" in source
    assert "data.max_train_samples=null" in source
    assert "eval.split=test" in source
    assert "eval.every_epoch=false" in source
    assert "git\", \"fetch" in source
    assert "merged_all.csv" not in source
