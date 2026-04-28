# scripts/validate_submission.py
import json
import os
import sys


def validate(record_dir):
    required = ['train_gpt.py', 'submission.json', 'README.md', 'requirements.txt',
                'train_seed42.log', 'train_seed314.log', 'train_seed999.log']
    missing = [f for f in required if not os.path.exists(os.path.join(record_dir, f))]
    assert not missing, f"missing: {missing}"
    meta = json.load(open(os.path.join(record_dir, 'submission.json')))
    for k in ('author', 'github_id', 'val_bpb', 'seeds', 'seed_results', 'compliance'):
        assert k in meta, f"submission.json missing key: {k}"
    assert len(meta['seeds']) >= 3, "need at least 3 seeds"
    for seed, r in meta['seed_results'].items():
        assert r['artifact_bytes'] < 16_000_000, f"seed {seed} artifact > 16MB"
    assert meta['val_bpb'] <= 1.075, f"val_bpb {meta['val_bpb']} above fallback gate"
    code_bytes = os.path.getsize(os.path.join(record_dir, 'train_gpt.py'))
    assert code_bytes < 100_000, f"train_gpt.py unreasonably large: {code_bytes}"
    print(f"OK: {record_dir}  val_bpb={meta['val_bpb']:.5f}  code={code_bytes} bytes")


if __name__ == '__main__':
    validate(sys.argv[1])
