#!/usr/bin/env python3
"""Replay recorded successful submissions to audit Arrow evidence transport."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from veripy_agent.contracts import TaskSpec
from veripy_agent.training.agentic import sha, write_once
from veripy_agent.verifiers.dispatcher import verify_task_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = {row['extra_info']['task_id']: row for row in pq.read_table(args.train).to_pylist()}
    counts = Counter()
    records = []
    for path in sorted(args.receipts.glob('*.json')):
        receipt = json.loads(path.read_text())
        if not receipt['won'] or receipt['family'] not in {'filter_aggregate', 'missing_values', 'multi_constraint_decision'}:
            continue
        spec = TaskSpec.model_validate(rows[receipt['task_id']]['env_kwargs']['task'])
        submission = json.loads(receipt['steps'][-1]['result']['stdout'])
        result, evidence = verify_task_submission(spec, submission)
        counts[f"{receipt['family']}|answer={result.passed}|evidence={evidence}"] += 1
        records.append({'path': str(path), 'sha256': sha(path), 'answer_passed': result.passed,
            'evidence_passed': evidence, 'historical_evidence_reward': receipt['steps'][-1]['reward_components']['evidence']})
    write_once(args.output, {'schema': 'veripy.evidence_transport_replay', 'train_sha256': sha(args.train),
        'verifier_sha256': sha(Path('src/veripy_agent/verifiers/dispatcher.py')), 'counts': dict(counts),
        'records': records, 'scope': 'recorded_submissions_against_serialized_tasks_not_model_learning'})
    print(json.dumps(dict(counts), sort_keys=True))


if __name__ == '__main__':
    main()
