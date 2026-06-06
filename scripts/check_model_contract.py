"""Assert minimum F2 on a saved model artifact. Used by CI."""
import pickle
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ci_model.pkl"
with open(model_path, "rb") as f:
    art = pickle.load(f)

f2 = float(art.get("f2") or art.get("f2_test") or 0)
assert f2 >= 0.40, f"F2 too low on sample: {f2:.4f} (bar: 0.40)"
print(f"Contract passed: F2={f2:.4f} >= 0.40")
