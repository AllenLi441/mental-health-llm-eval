#!/usr/bin/env node
/**
 * Freeze an untouched, deterministic, label-stratified holdout from the official
 * PsySUICIDE train split. The public artifact contains only aggregate counts and
 * cryptographic commitments: never text, source ids, row-level labels, or membership.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { labels } from '../tasks/psysuicide.mjs';
import {
  PSYSUICIDE_V2_PARTITION,
  partitionPsySuicideTrain,
} from '../lib/psysuicide_partition.mjs';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function parseArgs(argv) {
  const args = {
    datasetRoot: process.env.EVAL_DATASETS_DIR || '',
    output: join(ROOT, 'reports', 'psysuicide-v2-train-holdout.commitment.json'),
    verify: false,
    selftest: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--dataset-root') args.datasetRoot = argv[++i];
    else if (key === '--output') args.output = resolve(argv[++i]);
    else if (key === '--verify') args.verify = true;
    else if (key === '--selftest') args.selftest = true;
    else throw new Error(`unknown flag: ${key}`);
  }
  return args;
}

function buildArtifact(trainPath) {
  const raw = readFileSync(trainPath);
  const rows = JSON.parse(raw.toString('utf8'));
  const { manifest } = partitionPsySuicideTrain(rows, labels);
  return {
    ...manifest,
    source: 'licensed PsySUICIDE official train split; raw rows excluded from public repository',
    dataset_file_sha256: sha256(raw),
    publishing_boundary: [
      'public: this aggregate commitment artifact and portable partition code',
      'private: raw text, source ids, row-level labels, partition membership, predictions, and model outputs',
    ],
    freeze_policy: 'commit and push this artifact before inspecting optimization rows, changing prompts, training models, or evaluating the holdout',
  };
}

function selftest() {
  const rows = [];
  for (const [labelIndex, label] of labels.entries()) {
    for (let i = 0; i < 10; i += 1) {
      rows.push({ idx: `${labelIndex}-${i}`, labels: [label], text: `synthetic-${labelIndex}-${i}` });
    }
  }
  rows.push({ idx: 'multi', labels: [labels[0], labels[1]], text: 'synthetic-multi' });
  rows.push({ idx: 'unknown', labels: ['unknown'], text: 'synthetic-unknown' });
  const first = partitionPsySuicideTrain(rows, labels);
  const second = partitionPsySuicideTrain([...rows].reverse(), labels);
  if (JSON.stringify(first.manifest) !== JSON.stringify(second.manifest)) {
    throw new Error('partition is not invariant to source row order');
  }
  if (first.manifest.holdout_rows !== labels.length * 2) throw new Error('unexpected holdout count');
  if (first.manifest.optimization_rows !== labels.length * 8) throw new Error('unexpected optimization count');
  if (first.manifest.dropped_multi_label_rows !== 1 || first.manifest.dropped_unknown_label_rows !== 1) {
    throw new Error('unexpected dropped-row counts');
  }
  const publicText = JSON.stringify(first.manifest);
  if (publicText.includes('synthetic-') || publicText.includes('"idx"')) {
    throw new Error('public commitment leaks row contents or source ids');
  }
  const altered = rows.map((row, i) => i === 0 ? { ...row, text: `${row.text}-changed` } : row);
  const changed = partitionPsySuicideTrain(altered, labels);
  if (changed.manifest.all_rows_commitment_sha256 === first.manifest.all_rows_commitment_sha256) {
    throw new Error('row mutation did not change the dataset commitment');
  }
  console.log('PsySUICIDE v2 partition selftest PASS: deterministic, stratified, mutation-sensitive, aggregate-only');
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.selftest) {
    selftest();
    return;
  }
  if (!args.datasetRoot) {
    throw new Error('set EVAL_DATASETS_DIR or pass --dataset-root; no dataset path is embedded in the public artifact');
  }
  const trainPath = join(resolve(args.datasetRoot), 'PsySUICIDE', 'repo', 'train.json');
  const artifact = buildArtifact(trainPath);
  const rendered = `${JSON.stringify(artifact, null, 2)}\n`;
  if (args.verify) {
    if (!existsSync(args.output)) throw new Error(`commitment artifact does not exist: ${args.output}`);
    if (readFileSync(args.output, 'utf8') !== rendered) {
      throw new Error('committed PsySUICIDE v2 train/holdout artifact does not match the licensed dataset');
    }
    console.log(`PsySUICIDE v2 partition verify PASS: retained=${artifact.retained_single_label_rows} optimization=${artifact.optimization_rows} holdout=${artifact.holdout_rows}`);
    return;
  }
  if (existsSync(args.output)) {
    throw new Error(`refusing to overwrite frozen commitment artifact: ${args.output}; use --verify`);
  }
  writeFileSync(args.output, rendered, { flag: 'wx' });
  console.log(`PsySUICIDE v2 partition frozen: retained=${artifact.retained_single_label_rows} optimization=${artifact.optimization_rows} holdout=${artifact.holdout_rows}`);
  console.log(`aggregate commitment -> ${args.output}`);
}

main();
