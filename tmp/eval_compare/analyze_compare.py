import json
from pathlib import Path

hf = json.loads(Path("tmp/eval_compare/TOFU_SUMMARY_hf_ref.json").read_text())
ft = json.loads(Path("tmp/eval_compare/TOFU_SUMMARY_modal_ref.json").read_text())

print("metric\thf_ref\tmodal_ft\tdelta(ft-hf)")
for k in sorted(set(hf) | set(ft)):
    hv = hf.get(k)
    fv = ft.get(k)
    if isinstance(hv, (int, float)) and isinstance(fv, (int, float)):
        print(f"{k}\t{hv:.12g}\t{fv:.12g}\t{(fv-hv):.12g}")
    else:
        print(f"{k}\t{hv}\t{fv}\tNA")

print("\n--- trainer_state ---")
state = json.loads(Path("tmp/eval_compare/trainer_state.json").read_text())
print("num_train_epochs:", state.get("num_train_epochs"))
print("max_steps:", state.get("max_steps"))
print("global_step:", state.get("global_step"))
print("epoch:", state.get("epoch"))
print("best_metric:", state.get("best_metric"))

hist = state.get("log_history", [])
rows = [r for r in hist if "epoch" in r and ("loss" in r or "eval_loss" in r)]
print("\nrecent epoch/loss rows:")
for r in rows[-30:]:
    out = {k: r.get(k) for k in ["epoch", "step", "loss", "eval_loss", "learning_rate"] if k in r}
    print(out)
