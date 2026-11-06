"""Execute the notebook's smoke path in a real kernel and verify saved artifacts.

Usage: .venv/bin/python scripts/verification/verify_mae_notebook.py [--resume RUN_DIRECTORY]
The full pilot is never selected by this runner.
"""
import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=Path)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
(root / "local/scratch").mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(root / "notebooks/mae/mae_baseline.ipynb", as_version=4)
assert 'MODE = "smoke"' in notebook.cells[1].source
if args.resume:
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "RESUME_RUN = None", f"RESUME_RUN = {str(args.resume.resolve())!r}", 1)
if args.cpu:
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "# Avoid CPU thread overhead", 'DEVICE = torch.device("cpu")\n# Avoid CPU thread overhead', 1)

notebook.cells.append(nbformat.v4.new_code_cell(r'''
# Additional artifact and resume checks, executed only by the verification runner.
assert MODE == "smoke"
assert len(diagnostics_df) == 4 * CFG["n_folds"] * len(LABELS)
assert len(metrics_df) == 4 * len(LABELS)
assert len(summary) == 4
assert len(history_df) == CFG["epochs"]
for name, features in representations.items():
    saved_predictions = pd.read_csv(RUN_DIR / f"oof_{name}.csv")
    assert saved_predictions.StudyInstanceUID.tolist() == labeled_ids
    for fold, (_, val_idx) in enumerate(folds):
        fitted = torch.load(RUN_DIR / f"probe_{name}_fold{fold}.pt", weights_only=True)
        x = torch.from_numpy(features[val_idx]).double()
        x = (x - fitted["mean"]) / fitted["scale"]
        predictions = (x @ fitted["weights"].T + fitted["biases"]).sigmoid().numpy()
        np.testing.assert_allclose(predictions, saved_predictions.loc[val_idx, LABELS], rtol=1e-9, atol=1e-12)

# A completed epoch checkpoint must reproduce the next optimizer step, including
# masks and Adam moments. This operates on throwaway models, never saved weights.
checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
outer_rng = capture_rng()
resume_a, resume_b = SmallMAE(CFG).to(DEVICE), SmallMAE(CFG).to(DEVICE)
resume_a.load_state_dict(checkpoint["model"])
resume_b.load_state_dict(checkpoint["model"])
opt_a, opt_b = make_optimizer(resume_a), make_optimizer(resume_b)
opt_a.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
opt_b.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
results = []
for net, opt in [(resume_a, opt_a), (resume_b, opt_b)]:
    restore_rng(checkpoint["rng"])
    net.train()
    opt.zero_grad(set_to_none=True)
    loss, _, mask = net(x_check)
    loss.backward()
    opt.step()
    results.append((float(loss.detach().cpu()), mask.cpu()))
assert results[0][0] == results[1][0]
assert torch.equal(results[0][1], results[1][1])
for a, b in zip(resume_a.parameters(), resume_b.parameters()):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
restore_rng(outer_rng)
del resume_a, resume_b, opt_a, opt_b
verification["saved_probes_reproduce_oof"] = True
verification["checkpoint_next_step_reproducible"] = True
(RUN_DIR / "verification.json").write_text(json.dumps(verification, indent=2))
print("Saved probes reproduce all OOF predictions; optimizer/RNG checkpoint resume verified.")
print("VERIFIED_RUN_DIR=" + str(RUN_DIR.resolve()))
'''))

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def announce(cell_index, **kwargs):
    print(f"Executing cell {cell_index}", flush=True)


client = NotebookClient(notebook, km=manager, timeout=300,
                        resources={"metadata": {"path": str(root)}},
                        on_cell_start=announce)
output = root / "local/scratch" / "mae_baseline_smoke.executed.ipynb"
try:
    client.execute()
finally:
    nbformat.write(notebook, output)
    print(f"Executed notebook: {output}", flush=True)
    for cell in notebook.cells:
        for entry in cell.get("outputs", []):
            if entry.output_type == "stream" and "VERIFIED_RUN_DIR=" in entry.text:
                print(entry.text, flush=True)
