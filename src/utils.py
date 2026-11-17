import hashlib
import json


def stable_seed(*parts):
    """Order-independent mapping from explicit context to a uint32 seed."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()[:4], 'big')


def tie_key(patient_id):
    return hashlib.sha256(str(patient_id).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
