# Benchmark specification registry

`registry.json` is the ordered index for all task records under `tasks/`. Each
task record is self-contained and follows `schema.json`: dataset identity and
availability, split-use permissions, metrics, sample/call protocol, provenance
requirements, paper compatibility, and the dated project status.

The registry contains no raw examples, row-level gold labels or predictions,
credentials, home-directory paths, or licensed dataset locations. Canonical
label vocabularies and their order are protocol metadata required to bind a
classifier head. Dataset roots are always paths relative to
`EVAL_DATASETS_DIR`; raw and row-level material remains outside Git.

Validate the committed registry and its exact alignment with `node run.mjs list`:

```bash
python3 scripts/check_benchmark_registry.py
python3 scripts/check_benchmark_registry.py --selftest
python3 scripts/check_benchmark_registry.py --print-hashes
```

`--print-hashes` emits a canonical bundle commitment plus one canonical hash per
task. A trainer must store the bundle hash and the selected task-spec hash in
its preregistration and run artifact; merely reading an unvalidated JSON file
does not bind a run to this registry.

The four IMHI specialist-screen profiles additionally preregister SHA-256
commitments of the raw train/dev file bytes. A screen must compare its observed
source-file hashes with those commitments before creating jobs; this freezes
the authorized local split identity without publishing any row-level data.

The standard-library-only checker rejects missing fields, duplicate task keys,
unknown metrics, any training or selection use on a normalized test role,
unsafe paths/secret-like values, ambiguous DeepSeek aliases, and an EmoBench
paper protocol other than 400 questions × 4 option orders × 5 repetitions per
task (8,000 calls each; 16,000 calls for EA + EU).

Update discipline:

1. Change a task record only with a dated evidence update elsewhere in the
   project ledger; a status string is not proof that a run completed.
2. Freeze model, prompt, parser, dataset, and sample provenance before any test
   confirmation. Test may never select a model, prompt, parser, threshold, or
   training checkpoint.
3. An `unversioned_local_copy` must remain `frozen: false`; create and record a
   non-sensitive manifest commitment before starting a new scored run. For a
   supervised screen, commit the exact train/dev source-file hashes in its
   profile and reject any runtime mismatch.
4. `paper_compatibility` describes the present evidence/protocol boundary, not
   whether a point estimate happens to exceed a published number.
