# Residual geometry and the predictor families, measured under full compute with
# no cache applied -- so the result is a property of the model, not of any cache.
# Deterministic given the episode set, so a meaningful repeat is a DIFFERENT
# MACHINE, not another run here. BACKBONE=fw|pi05|dz.
TITLE="Residual geometry and predictor families"
SECTION="03/04"
BACKBONE="${BACKBONE:-fw}"
# The shipped artefacts are the primary measurement and a repeat on a SECOND
# MACHINE -- this measurement is deterministic given its episode set, so a repeat
# on the same host proves nothing. Their names are not uniform (Fast-WAM's repeat
# shipped as fw_families_repro52.npz, the other two as *_families_repro.npz), so
# the tag selects the shipped name rather than being appended to it. Run "primary"
# on one host and "repro" on another.
DEFAULT_SEEDS="primary repro"
NEEDS_SIM=no
GATE=no
ANALYSIS=residual

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT="${CKPT:-libero_uncond_2cam224.pt}"

# Recorded episodes this experiment replays. Overridable: the path was
# hardcoded to the machine it was first run on and documented nowhere, so a
# reviewer holding the videos anywhere else could not run it and got no
# warning until the probe failed.
VIDEO_ROOT="${VIDEO_ROOT:-/workspace/libero}"

# Fast-WAM also shipped fw_families_idm.npz, the same measurement on the
# checkpoint the closed-loop arms actually run, and fw_families_idm_taps7.npz,
# that same replay recording cache ages 5-7 as well. Collect them with
#   ./repro/run_experiment.sh mechanism --seeds idm
#   ./repro/run_experiment.sh mechanism --seeds idm7
# Each tag selects the deployment checkpoint and episodes24_idm.json by itself;
# both setup/onstart_fw_families.sh downloads. Nothing needs to be exported.
_mech_name() {   # <backbone> <tag> -> the shipped artefact name
  case "$1:$2" in
    fw:primary)   echo fw_families ;;
    fw:repro)     echo fw_families_repro52 ;;
    fw:idm)       echo fw_families_idm ;;
    # The same deployment-checkpoint replay, recording taps 5-7 as well. Written
    # under a new name so the shipped eight-column artefact survives beside it:
    # both remain readable, and a reader can compare the two layouts directly.
    fw:idm7)      echo fw_families_idm_taps7 ;;
    fw:repeat16)  echo t1_seed0 ;;
    pi05:primary) echo pi05_families ;;
    pi05:repro)   echo pi05_families_repro ;;
    dz:primary)   echo dz_families ;;
    dz:repro)     echo dz_families_repro ;;
    *)            echo "$1_families_$2" ;;
  esac
}

# Which checkpoint a tag loads. preflight reads this too: declaring the mapping
# only inside run_one meant preflight verified CKPT, passed a host holding just
# the unconditional file, and let "--seeds idm7" fail minutes later inside model
# loading. One definition, both readers.
ckpt_for_seed() {
  case "$1" in
    idm|idm7) echo libero_optional_idm_2cam224.pt ;;
    *)        echo "$CKPT" ;;
  esac
}

# An explicit MANIFEST from the environment still wins, but it has to be captured
# before run_one can overwrite it -- which is how the leak below arose.
MANIFEST_OVERRIDE="${MANIFEST:-}"

# Which recorded-episode list a tag replays. Declared here for the same reason as
# ckpt_for_seed, and for one of its own: MANIFEST used to be assigned inside
# run_one WITHOUT `local`, so it survived the iteration that set it and the next
# tag inherited it. "--seeds 'idm7 primary'" therefore collected fw_families.npz
# -- the primary geometry artefact this paper reports -- from the 24-episode idm
# list instead of episodes50.json, with nothing in the output saying so.
manifest_for_seed() {
  [ -n "$MANIFEST_OVERRIDE" ] && { echo "$MANIFEST_OVERRIDE"; return; }
  case "$1" in
    idm|idm7) echo episodes24_idm.json ;;
    repeat16) echo episodes16_t1.json ;;
    *)        echo episodes50.json ;;
  esac
}

run_one() {
  local tag="$1" out="$2" name
  name=$(_mech_name "$BACKBONE" "$tag")
  # Checkpoint and episode list both come from the tag. Both were once implicit:
  # the header documented "CKPT=... --seeds idm" while the invocation below
  # hardcoded the unconditional checkpoint and ignored it, silently producing an
  # unconditional-checkpoint artefact under an idm name; and t1_seed0's
  # 16-episode list lived only inside the artefact's own meta. Both lists are now
  # committed (episodes24_idm.json, episodes16_t1.json), and both resolvers are
  # pure functions of the tag, so neither depends on a variable surviving.
  local ck mf
  ck=$(ckpt_for_seed "$tag")
  mf=$(manifest_for_seed "$tag")
  case "$BACKBONE" in
    fw)
      cd "$WAM_ROOT" && python probes/fw_families.py \
        ckpt="$CKPT_DIR/$ck" model.redirect_common_files=false \
        +PROBE.manifest="$REPO/inputs/manifests/$mf" \
        +PROBE.video_root="$VIDEO_ROOT" \
        +PROBE.stride=10 +PROBE.proprio_seed=0 \
        +PROBE.out="$out/${name}.npz" ;;
    pi05)
      cd "$WAM_ROOT" && PI_OUT="$out/${name}.npz" python probes/pi05_families.py ;;
    dz)
      cd "$WAM_ROOT" && DZ_OUT="$out/${name}.npz" \
        DZ_BLOCK_STRIDE=4 DZ_STEP_STRIDE=2 DZ_NTOK=128 \
        DZ_EPISODES="$REPO/inputs/manifests/dz_episodes20.json" python probes/dz_families.py ;;
  esac
}
