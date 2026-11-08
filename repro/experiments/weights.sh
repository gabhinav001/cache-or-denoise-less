# Regenerate the per-cell substitution weights that every attenuated arm uses.
#
# This matters for review: inputs/weights/*.json are inputs to the closed-loop
# experiments, so shipping them alone lets a reviewer USE our coefficients but
# not check how they were obtained. The claim being verified here is that they
# are a closed-form least-squares solution fit on training episodes only -- NOT
# swept, and never fit on anything the closed loop is scored on.
#
# Needs no GPU: it reads the recorded Gram matrices. Runs anywhere.
TITLE="Refit the per-cell substitution weights (no GPU)"
SECTION="inputs"
DEFAULT_SEEDS="refit"
NEEDS_SIM=no
GATE=no
ANALYSIS=none
NO_GPU=yes

run_one() {
  local tag="$1" out="$2" src dst name
  for src in "$REPO"/results/residual/fw_families.npz \
             "$REPO"/results/residual/fw_families_idm.npz \
             "$REPO"/results/residual/pi05_families.npz; do
    [ -e "$src" ] || continue
    case "$(basename "$src")" in
      fw_families.npz)     name=fw ;;
      fw_families_idm.npz) name=idm ;;
      pi05_families.npz)   name=pi05 ;;
    esac
    dst="$out/shrink_${name}.json"
    ( cd "$REPO/probes" && python3 export_shrink.py "$src" "$dst" tap1 ) || return 1
    python3 - "$dst" "$REPO/inputs/weights/shrink_${name}.json" <<'PYCMP'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
a, b = new["coef"], old["coef"]
keys = set(a) & set(b)
d = max(abs(a[k][0] - b[k][0]) for k in keys) if keys else float("inf")
print(f"  {sys.argv[1].split('/')[-1]}: {len(keys)} cells, "
      f"max |refit - shipped| = {d:.2e}  "
      f"{'MATCHES' if d < 1e-9 else 'DIFFERS'}")
print(f"  fit episodes identical: {new['fit_episodes'] == old['fit_episodes']}")
PYCMP
  done
}
