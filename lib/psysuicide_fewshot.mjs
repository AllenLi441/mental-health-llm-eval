import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { partitionPsySuicideTrain } from './psysuicide_partition.mjs';

export const BALANCED_RETRIEVAL_SPEC = Object.freeze({
  version: 'psysuicide-balanced-retrieval-v1',
  candidates_per_label: 128,
  examples_per_label: 1,
  max_example_chars: 480,
  similarity: 'character-trigram-dice-with-small-shortness-tiebreak',
  exact_text_exclusion: true,
  source_partition: 'optimizationRows only; frozen holdoutRows are never candidates',
});

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function normalize(text) {
  return String(text).normalize('NFKC').toLowerCase().replace(/\s+/g, '');
}

function trigrams(text) {
  const normalized = normalize(text);
  if (normalized.length < 3) return new Set([normalized]);
  const out = new Set();
  for (let index = 0; index <= normalized.length - 3; index += 1) {
    out.add(normalized.slice(index, index + 3));
  }
  return out;
}

function dice(left, right) {
  if (!left.size && !right.size) return 1;
  let overlap = 0;
  const [small, large] = left.size <= right.size ? [left, right] : [right, left];
  for (const gram of small) if (large.has(gram)) overlap += 1;
  return (2 * overlap) / (left.size + right.size || 1);
}

function exemplarText(text) {
  const value = String(text).trim();
  if (value.length <= BALANCED_RETRIEVAL_SPEC.max_example_chars) return value;
  return `${value.slice(0, BALANCED_RETRIEVAL_SPEC.max_example_chars)}…`;
}

export function buildBalancedRetriever(datasetRoot, labels) {
  const trainPath = join(datasetRoot, 'PsySUICIDE', 'repo', 'train.json');
  const rows = JSON.parse(readFileSync(trainPath, 'utf8'));
  const { optimizationRows, manifest } = partitionPsySuicideTrain(rows, labels);
  const pools = new Map();
  for (const label of labels) {
    const candidates = optimizationRows
      .filter((row) => row.labels?.[0] === label)
      .map((row) => ({
        text: String(row.text),
        rank: sha256(`${row.idx}\0${row.text}`),
      }))
      .sort((a, b) => a.rank.localeCompare(b.rank))
      .slice(0, BALANCED_RETRIEVAL_SPEC.candidates_per_label)
      .map((entry) => ({ ...entry, grams: trigrams(entry.text) }));
    if (!candidates.length) throw new Error(`balanced retrieval has no optimization candidate for ${label}`);
    pools.set(label, candidates);
  }

  return {
    manifest,
    retrieve(queryText) {
      const query = String(queryText);
      const queryGrams = trigrams(query);
      return labels.map((label) => {
        const ranked = pools.get(label)
          .filter((candidate) => candidate.text !== query)
          .map((candidate) => ({
            ...candidate,
            score: dice(queryGrams, candidate.grams)
              + 0.002 * (1 - Math.min(candidate.text.length, 500) / 500),
          }))
          .sort((a, b) => b.score - a.score || a.rank.localeCompare(b.rank));
        if (!ranked.length) throw new Error(`balanced retrieval exhausted candidates for ${label}`);
        return { label, text: exemplarText(ranked[0].text) };
      });
    },
  };
}

export function syntheticBalancedExamples(labels) {
  return labels.map((label) => ({
    label,
    text: `仅用于离线接口自检的合成占位句（类别：${label}）`,
  }));
}
