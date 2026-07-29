import { createHash } from 'node:crypto';

export const PSYSUICIDE_V2_PARTITION = Object.freeze({
  schema_version: 1,
  partition_id: 'psysuicide-train-opt-holdout-v2',
  seed: 'psysuicide-model-optimization-v2-2026-07-28',
  holdout_fraction: 0.2,
  selection_rule: 'within each label, sort by sha256(seed + NUL + canonical-row-sha256), then take floor(label_n * holdout_fraction)',
});

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalRow(row) {
  return JSON.stringify({
    idx: String(row.idx),
    labels: row.labels,
    text: String(row.text),
  });
}

function commitment(digests) {
  return sha256([...digests].sort().join('\n'));
}

export function partitionPsySuicideTrain(rows, labels, options = {}) {
  const spec = { ...PSYSUICIDE_V2_PARTITION, ...options };
  if (!Array.isArray(rows) || !rows.length) throw new Error('PsySUICIDE train rows are empty');
  if (!Array.isArray(labels) || !labels.length) throw new Error('PsySUICIDE labels are empty');
  if (!(spec.holdout_fraction > 0 && spec.holdout_fraction < 0.5)) {
    throw new Error('holdout_fraction must be greater than 0 and less than 0.5');
  }

  const allowed = new Set(labels);
  const retained = [];
  let droppedMultiLabel = 0;
  let droppedUnknownLabel = 0;
  for (const row of rows) {
    if (!Array.isArray(row.labels) || row.labels.length !== 1) {
      droppedMultiLabel += 1;
      continue;
    }
    const label = row.labels[0];
    if (!allowed.has(label)) {
      droppedUnknownLabel += 1;
      continue;
    }
    const rowDigest = sha256(canonicalRow(row));
    retained.push({
      row,
      label,
      rowDigest,
      rankDigest: sha256(`${spec.seed}\0${rowDigest}`),
    });
  }

  if (new Set(retained.map((entry) => entry.rowDigest)).size !== retained.length) {
    throw new Error('duplicate canonical PsySUICIDE train rows detected');
  }

  const holdoutEntries = [];
  const optimizationEntries = [];
  const perLabel = {};
  for (const label of labels) {
    const group = retained
      .filter((entry) => entry.label === label)
      .sort((a, b) => a.rankDigest.localeCompare(b.rankDigest) || a.rowDigest.localeCompare(b.rowDigest));
    if (!group.length) throw new Error(`PsySUICIDE train label has no retained rows: ${label}`);
    const holdoutCount = Math.max(1, Math.floor(group.length * spec.holdout_fraction));
    const holdout = group.slice(0, holdoutCount);
    const optimization = group.slice(holdoutCount);
    holdoutEntries.push(...holdout);
    optimizationEntries.push(...optimization);
    perLabel[label] = {
      retained: group.length,
      optimization: optimization.length,
      holdout: holdout.length,
    };
  }

  const manifest = {
    schema_version: spec.schema_version,
    partition_id: spec.partition_id,
    seed: spec.seed,
    holdout_fraction: spec.holdout_fraction,
    selection_rule: spec.selection_rule,
    retained_single_label_rows: retained.length,
    optimization_rows: optimizationEntries.length,
    holdout_rows: holdoutEntries.length,
    dropped_multi_label_rows: droppedMultiLabel,
    dropped_unknown_label_rows: droppedUnknownLabel,
    per_label: perLabel,
    all_rows_commitment_sha256: commitment(retained.map((entry) => entry.rowDigest)),
    optimization_commitment_sha256: commitment(optimizationEntries.map((entry) => entry.rowDigest)),
    holdout_commitment_sha256: commitment(holdoutEntries.map((entry) => entry.rowDigest)),
  };

  return {
    manifest,
    optimizationRows: optimizationEntries.map((entry) => entry.row),
    holdoutRows: holdoutEntries.map((entry) => entry.row),
  };
}
