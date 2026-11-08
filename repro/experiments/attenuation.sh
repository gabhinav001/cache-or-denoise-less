# Attenuating the cached residual raises closed-loop success at identical compute.
# Repeats vary the SEED, which reseeds both the simulator's initial states and the
# denoiser's noise: success moves up to 10 points across seeds, so one run cannot
# support a claim about an effect that size.
#
# WHAT THIS CAN AND CANNOT REPRODUCE
#
# The nine shipped runs in results/closedloop/ were collected before this harness
# existed, ad hoc, at settings that differ from each other: 10, 20 and 30 episodes
# per arm, three configs or five, ten tasks or five. They are enumerated below so
# a fresh host re-collects the same nine configurations under the same filenames,
# rather than twelve similar-looking runs the analysis would pool into a different
# number.
#
# Four of them cannot be reproduced exactly. None of the nine artefacts records a
# seed -- the field did not exist when they were written -- so for five the seed
# is read off the filename (q_*_seed7, t2_seed7, t2_seed11) and for the other four
# it is simply unknown. Those four are marked seed=? below and run at 42.
#
# What that costs is worth stating plainly. A reviewer re-running this gets the
# same configurations and, on the five recoverable seeds, comparable episode
# outcomes; on the other four they get a valid measurement of the same quantity at
# a different draw. The pooled tau=8 effect should replicate -- it is +16.0 points
# with all nine runs positive, far outside the seed-to-seed spread -- but the
# artefacts will not match episode for episode, and no amount of care here can
# recover a seed nobody wrote down. Runs collected since carry a seed and a
# provenance block; see probes/provenance.py.
TITLE="Attenuated cached residual, closed loop (tau=8)"
SECTION="01"

# name:suite:tasks:episodes:seed:configs   ("?" marks a seed that was never recorded)
RUNS="${RUNS:-\
q_libero10_seed7:libero_10:10:10:7:full,c3ache:8,c3ache:8:s \
q_object_seed7:libero_object:10:10:7:full,c3ache:8,c3ache:8:s \
q_spatial_seed7:libero_spatial:10:10:7:full,c3ache:8,c3ache:8:s \
t2_seed7:libero_goal:10:20:7:full,c3ache:8,c3ache:8:s \
t2_seed11:libero_goal:10:20:11:full,c3ache:8,c3ache:8:s \
suite_object:libero_object:10:10:?:full,c3ache:4,c3ache:4:s,c3ache:8,c3ache:8:s \
suite_spatial:libero_spatial:10:10:?:full,c3ache:4,c3ache:4:s,c3ache:8,c3ache:8:s \
tau48:libero_goal:10:20:?:full,c3ache:4,c3ache:4:s,c3ache:8,c3ache:8:s \
suite_libero10_n300a:libero_10:0-5:30:?:full,c3ache:8,c3ache:8:s \
suite_libero10_n300b:libero_10:5-10:30:?:full,c3ache:8,c3ache:8:s}"

DEFAULT_SEEDS="$RUNS"
GATE=yes
ANALYSIS=closedloop
CKPT=libero_optional_idm_2cam224.pt

# suite_libero10_n300 shipped as a merge of two machines covering disjoint task
# ranges: its meta records one shard's 5 tasks while the file holds all ten, 900
# rows. Re-collecting only the first shard would silently reproduce half the task
# grid the headline pools, so this runs BOTH halves as a task range and merges
# them at the end -- the merged file is what the analysis reads, and the halves
# are left beside it as evidence.

run_one() {   # <name:suite:tasks:episodes:seed:configs> <outdir>
  local spec="$1" out="$2" name suite tasks eps seed cfgs
  name=${spec%%:*};            spec=${spec#*:}
  suite=${spec%%:*};           spec=${spec#*:}
  tasks=${spec%%:*};           spec=${spec#*:}
  eps=${spec%%:*};             spec=${spec#*:}
  seed=${spec%%:*};            cfgs=${spec#*:}
  local first=0
  case "$tasks" in *-*) first=${tasks%%-*}; tasks=${tasks#*-};; esac
  if [ "$seed" = "?" ]; then
    seed=42
    echo "  -- $name: seed was never recorded for this artefact; running at 42." >&2
    echo "     Configuration matches; episode outcomes will not." >&2
  fi
  echo "  -- $name ($suite, tasks $first..$((tasks-1)), $eps episodes, seed $seed)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite="$suite" +PROBE.first_task="$first" \
    +PROBE.n_tasks="$tasks" +PROBE.n_episodes="$eps" \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

# The shipped suite_libero10_n300 is one file holding all ten tasks. Its two
# halves are collected above as separate runs; merge_l10.py asserts the task sets
# are disjoint and the configs identical before joining them, and refuses rather
# than producing a file that silently double-counts a task.
post_collect() {
  local out="$1" a="$1/suite_libero10_n300a.json" b="$1/suite_libero10_n300b.json"
  # DRY_RUN asks what this experiment would write. The merge is a second writer,
  # invisible to a scan of the probe command lines, so declare it: MERGED_OUT is
  # the artefact the analysis reads, INTERMEDIATE names the shards behind it so
  # they are not mistaken for outputs nothing ships.
  if [ -n "${DRY_RUN:-}" ]; then
    echo "MERGED_OUT=$out/suite_libero10_n300.json"
    echo "INTERMEDIATE=$a"
    echo "INTERMEDIATE=$b"
    return 0
  fi
  [ -s "$a" ] && [ -s "$b" ] || { echo "  -- halves missing, nothing to merge" >&2; return 0; }
  echo "  -- merging the two libero_10 task ranges" >&2
  python3 "$REPO/probes/merge_l10.py" "$a" "$b" "$out/suite_libero10_n300.json" || return 1
  # the halves stay on disk as evidence, but out of the analysis directory: the
  # variance reporter globs *.json, and counting a task twice would move the mean
  mkdir -p "$out/shards" && mv "$a" "$b" "$out/shards/"
}
