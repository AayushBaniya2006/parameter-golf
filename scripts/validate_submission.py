# scripts/validate_submission.py
import json
import os
import sys


def validate(record_dir, allow_fallback=False):
    required = ['train_gpt.py', 'submission.json', 'README.md', 'requirements.txt',
                'train_seed42.log', 'train_seed314.log', 'train_seed999.log']
    missing = [f for f in required if not os.path.exists(os.path.join(record_dir, f))]
    assert not missing, f"missing: {missing}"
    with open(os.path.join(record_dir, 'submission.json')) as f:
        meta = json.load(f)
    for k in ('author', 'github_id', 'val_bpb', 'seeds', 'seed_results', 'compliance'):
        assert k in meta, f"submission.json missing key: {k}"
    assert len(meta['seeds']) >= 3, "need at least 3 seeds"
    for seed, r in meta['seed_results'].items():
        assert 'val_bpb' in r, f"seed {seed} missing val_bpb"
        assert 'artifact_bytes' in r, f"seed {seed} missing artifact_bytes"
        assert r['artifact_bytes'] < 16_000_000, f"seed {seed} artifact > 16MB"
    gate = 1.081 if allow_fallback else 1.075
    gate_label = "fallback gate 1.081" if allow_fallback else "stretch gate 1.075"
    assert meta['val_bpb'] <= gate, f"val_bpb {meta['val_bpb']} above {gate_label}"
    code_bytes = os.path.getsize(os.path.join(record_dir, 'train_gpt.py'))
    assert code_bytes < 100_000, f"train_gpt.py unreasonably large: {code_bytes}"
    print(f"OK: {record_dir}  val_bpb={meta['val_bpb']:.5f}  gate={gate_label}  code={code_bytes} bytes")


if __name__ == '__main__':
    args = sys.argv[1:]
    allow_fallback = '--allow-fallback' in args
    args = [a for a in args if a != '--allow-fallback']
    if len(args) != 1:
        sys.exit("usage: validate_submission.py [--allow-fallback] <record_dir>")
    validate(args[0], allow_fallback=allow_fallback)
