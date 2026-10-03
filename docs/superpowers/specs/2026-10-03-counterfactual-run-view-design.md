# Counterfactual Run View

The server log fails before model loading: `cail.jsonl:13` references a parent absent from
the current smoke training subset. Generation resumed twelve current requests from a cache
that also contains historical requests. This run does not establish whether GPU OOM is fixed.

Keep the existing append-only generator cache. Add an optional `--view-output` that publishes
only the current unique requests in request order after successful generation/resume.
Validate selected cache rows against the current intervention spec, generator identity,
prompt version, and validation schema. Reject malformed cache JSON, duplicate selected IDs,
missing selected rows, or mismatched identity/spec before publishing. Use a same-directory
temporary file, fsync, and atomic replace; failure preserves the previous view and cache.
Include invalid generations in the view so A6 keeps its existing semantics.

Add `--parent-limit` to generation, counting processed JSONL records before skipping long
cases or proposing interventions. `run.sh` passes `DATA_LIMIT`, matching the training subset
for CMDL's multiple defendants as well as CAIL. The view feeds all training and prediction
commands; the master cache is never passed to them by the pipeline.

Use `counterfactuals/<generator>/views/<mode>/` for current views. Smoke training and
evaluation use `runs/<generator>/smoke/length-<length>/`; main/matrix retain existing paths
and checkpoint resume. A repeated smoke run archives the exact prior experiment/seed
directory into a timestamped sibling before writing fresh results, preserving old artifacts.
Smoke does not select checkpoints and always exercises a new optimizer step.

Typed experiments require at least one eligible pair after validation, known-target filtering,
and ablation-type routing. Fail before loading weights if none remain. Unknown parent IDs
remain strict errors; they indicate wrong manual inputs rather than being silently discarded.

Tests reproduce a master cache containing twelve current rows plus an orphan thirteenth row,
verify byte-preserving cache reuse and exact view publication, reject malformed/duplicate/
incompatible cache rows while preserving the old view, exercise the real mock CLI twice,
align processed-parent limits, and check smoke/main command paths and resume behavior.
