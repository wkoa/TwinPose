# Rust algorithm library and CLI

The local `feat/rust-migration` branch ports the algorithm and command-line data
pipeline from upstream commit `fc684f1e2f487fac1e4005b8520b02a79b6458a2`.
The original C++ source, renderer, assets and license are retained as references.
Rust runs its own implementation: there is no C/C++ FFI, native OpenCV dependency,
CMake invocation, detector/model download or GUI dependency in the Cargo build.
The user requested the algorithm library and CLI only; the original standalone
Ink3D/SDL/ImGui visualization and its seven debug modes are outside this port.

## Build and run

From this repository root, with a Rust toolchain (declared minimum 1.88; actually
validated on macOS arm64 with Rust 1.97.0):

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check

cargo run --release --locked -- run \
  --dataset dataset/shelf/dataset.json \
  --algorithm dataset/shelf/algorithm.json \
  --output /tmp/twinpose-shelf.jsonl
```

The output path must not exist. `--limit N` restricts the number of processed
frames. JSONL contains the original detection `frame_id`, sparse `poses`, sparse
`ground_truth`, per-ground-truth-person metrics, wall-clock `algorithm_ms` and
`rejected_geometry`. Output is flushed before success. A failure returns nonzero
and writes its reason to stderr; an interrupted/failed batch can leave a partial
JSONL prefix, which must not be treated as a completed run. Raw input files are
only opened for reading. Dataset-relative paths are resolved relative to the
configuration file, just as in the upstream JSON importer.

For a calibrated external producer:

```sh
twinpose stream --algorithm dataset/shelf/algorithm.json < frames.jsonl > poses.jsonl
```

Each input line has this shape (two or more views required):

```json
{"frame_id":300,"views":[{"camera":{"k":[[1000,0,500],[0,1000,400],[0,0,1]],"r":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,0]},"joints":{"0":[{"point":[500,400],"conf":0.9,"uid":1}]}},{"camera":{"k":[[1000,0,500],[0,1000,400],[0,0,1]],"r":[[1,0,0],[0,1,0],[0,0,1]],"t":[-1,0,0]},"joints":{"0":[{"point":[300,400],"conf":0.9,"uid":1}]}}]}
```

Joint arrays hold view-local, one-based person candidates internally; `uid` must
be unique within a view. Frame IDs must strictly increase. Empty observations
produce empty poses, never the last pose. A new process or `Tracker::reset()`
starts an independent sequence. `max_person_num=0` derives a per-frame candidate
limit for streams; set an explicit fixed limit for a stable stream configuration.
A pose ID is the **current frame's association rank**, not a persistent identity.
The temporal option adjusts scores using the previous frame; it does not establish
a global person-ID contract. Add `"use_temporal_information": true` to algorithm
JSON to enable it (default false, matching the upstream application).

Input camera convention is `p_camera = R * p_world + t`, `p_pixel ~ K*p_camera`;
outputs stay in the dataset camera/world coordinate system. Upstream's negative
ray direction is preserved (triangulation uses infinite lines). No cheirality test,
scene-axis display remap, calibration, exposure synchronization, unit conversion
or metric-scale verification is invented. Algorithm distances expect **meters**;
callers must supply coordinates in consistent units. Panoptic's raw positions are
retained just as upstream: convert a centimeter-based source explicitly before
mixing it with meter-based detections/cameras. Metrics labeled `mpjpe_mm` assume
meter inputs; they do not certify a dataset's physical scale.

## Source mapping

| Original source | Rust replacement |
|---|---|
| `Common.h`, `Math.h`, `Triangulation.h`, Ink vector/matrix operations | `rust/src/model.rs`, `geometry.rs`: owned cameras/views, rays, f32 math, weighted/unweighted least-squares intersection |
| `PoseBuilder.h`, `AnomalyDetector.h` | `pose.rs`: pair candidates, spatial-grid clustering, confidence/epipolar scores, bone-length statistics, association forest, per-person bone graphs, temporal penalties, bottom-up branch resolution |
| `PackedIntArray.h`, `DisjointSet.h`, `EdgeSet.h` | typed 16-camera byte arrays, owned disjoint sets, index-based graph edges; no pointer casts or raw branch pointers |
| `OpenPoseBody25.h`, `Conversion.h` | BODY25 topology; Coco17/Coco18/Coco19/Shelf14/OptiTrack21/Skel19 ground-truth mappings; Coco17 2D neck/hip construction |
| six files under `src/loaders/` | `dataset.rs`: 4DA/Campus/OSD cameras; MMPose/OpenPose/4DA/OSD detections; 4DA/Shelf/Campus/Panoptic annotations |
| `Correction.h`, `Synchronization.h`, detection scaling and camera rectification | explicit Shelf/4DA correction, sync-point index mapping, detection resolution scale, Rodrigues and Brown five-coefficient rectified K; no automatic coordinate/unit conversion |
| `Evaluation.h`, algorithm execution/history in `Main.cpp`, `Timer.h` | `evaluation.rs`, `Tracker`, CLI JSONL and `Instant` timings; 30-frame bone history and optional previous-pose scoring |

`Tracker` records bone statistics before dataset correction and keeps corrected
poses for temporal scoring, matching the original application. History changes
only after successful processing. At most 16 views and 255 candidates per joint
per view are supported. Joint subsets can be sparse and must be listed children
before parents. Default hard bone limits are the upstream 15-joint limits; other
BODY25 bone types have no learned hard prior by default. Library callers can
provide explicit `AnomalyDetector` values to `pose::build` for another skeleton.

Intentional safety/portability differences:

- Singular triangulations are rejected and counted, not emitted as NaN/zero poses.
  Invalid cameras, pixels, scores, thresholds, truncated text, and mismatched
  MMPose cross-camera frame IDs fail explicitly. Heatmap confidence scores above
  1 are valid (the bundled MMPose data contains them); they are not clamped.
- Rust orders clusters and tied candidates/associations by explicit per-camera
  candidate IDs. Upstream relies on C++ `unordered_map` traversal and unspecified
  sorting ties. Its fallback skeleton choices can therefore differ; this port
  does **not** promise bit-for-bit parity with every C++ standard library.
- Bone variance is clamped at zero to avoid a negative roundoff variance becoming
  NaN. Missing annotations remain absent. MPJPE is null when any evaluated GT
  joint lacks a prediction; an absent pose is never reported as zero-mm error.
  PCK/PCP use available GT denominators. These are explicit metric changes from
  the original evaluator; do not compare aggregates without matching this policy.
- Raw frames/annotations are retained without fabricated timestamps. A missing
  Panoptic annotation file is an empty annotation frame, matching that loader's
  convention. An incomplete detection view is an error.

## Reproducible checks

`cargo test --locked` exercises geometry/degeneracy, two-person association,
sparse joints, invalid parameters, empty/failed frames, independent histories,
confidence overflow, mappings/corrections/synchronization, actual bundled Shelf
input, all loader families, CLI ordering/parse errors, and overwrite protection.

The differential tool compiles the original algorithm headers (including their
original arithmetic), with only the header-only Ink math and original Ray
constructor. That C++ executable is **validation-only** and is never linked by
Cargo. The optional camera check requires NumPy/OpenCV in the validation Python,
not in the Rust application:

```sh
python3 rust/tools/compare_reference.py --limit 301 --opencv-check \
  --canonical-order --report /tmp/twinpose-shelf-parity.json
python3 rust/tools/compare_reference.py --limit 301 --temporal \
  --opencv-check --canonical-order --report /tmp/twinpose-shelf-temporal.json
python3 rust/tools/compare_reference.py \
  --dataset dataset/4da/seq4/dataset.json --algorithm dataset/4da/algorithm.json \
  --limit 120 --temporal --opencv-check --canonical-order \
  --report /tmp/twinpose-4da-seq4-parity.json
```

`--canonical-order` is deliberately explicit: it normalizes only the C++ candidate,
cluster and association iteration order before the original remaining stages.
It does not modify upstream files, relax tolerances, replace the C++ algorithm,
or change thresholds. Omitting it checks the original unspecified order and can
fail; failures are reported with mismatching frames and retained in an adjacent
`.mismatches.json`. Person permutations are matched for geometry comparison,
because rank IDs are not persistent identities. The exporter feeds identical
calibrated observations to both implementations; this proves the algorithm port,
not independent correctness of every loader. The optional OpenCV matrix check
independently verifies the Rust calibration calculation.

Observed macOS arm64 results (2026-10-25), tolerance `1e-4` dataset length units:

| Canonical reference | Frames | Compared joints | Maximum joint distance | K element error |
|---|---:|---:|---:|---:|
| Shelf, temporal off | 301 | 12,954 | 7.67225e-5 | 0 |
| Shelf, temporal on | 301 | 12,954 | 7.67225e-5 | 0 |
| 4DA seq4, temporal on | 120 | 5,400 | 4.09802e-6 | 0 |

Initial unmodified-order checks matched Shelf's first 10 frames within 1.54e-6,
but 117/301 Shelf frames differed (including person count at frame index 125;
maximum matched-pose joint difference 0.631546 dataset units).
The unmodified-order 4DA seq4 check had 36/120 differing frames, max distance
0.071117 dataset units. Canonical ordering resolved those tested discrepancies;
these unmodified checks are not recorded as passing parity tests.

The upstream bundled `dataset/4da/seq2/gt.txt` declares 13,127 frames but ends
inside the declared range. The strict reader reports truncated text; no source
repair or missing-ground-truth fabrication was performed. Use intact seq4/Shelf
for the supplied commands.

Not validated: original GUI, Linux/Windows native builds, actual devices, physical
pose accuracy, long-running live throughput, persistent person IDs, new detector
models or Studio integration. The library is an optional submodule experiment;
Studio does not import or start it automatically. This local branch has not been
pushed; a Studio gitlink to its local commit requires publishing that commit to an
authorized remote before another machine can fetch it.
