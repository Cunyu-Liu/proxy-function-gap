#!/usr/bin/env python3
import argparse
import binascii
import hashlib
import json
import os
import struct
import ssl
import time
import urllib.request
import zlib
from pathlib import Path

import certifi

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def get_range(url, start, end, attempts=6):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "scenario-gb1-v1/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180, context=SSL_CONTEXT) as response:
                if response.status != 206:
                    raise RuntimeError(f"Range request returned HTTP {response.status}, expected 206")
                data = response.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"Short range response: {len(data)} != {end-start+1}")
            return data
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intermediate-ca")
    args = parser.parse_args()
    if args.intermediate_ca:
        SSL_CONTEXT.load_verify_locations(cafile=args.intermediate_ca)
    config = json.loads(Path(args.config).read_text())
    url = config["source_url"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tail_start_request = urllib.request.Request(url, headers={"Range": "bytes=-65536", "User-Agent": "scenario-gb1-v1/1.0"})
    with urllib.request.urlopen(tail_start_request, timeout=180, context=SSL_CONTEXT) as response:
        if response.status != 206:
            raise RuntimeError(f"Tail range returned HTTP {response.status}")
        tail = response.read()
        content_range = response.headers["Content-Range"]
        total_size = int(content_range.rsplit("/", 1)[1])
    tail_start = total_size - len(tail)
    print(json.dumps({"stage": "tail", "zip_bytes": total_size}), flush=True)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("EOCD not found")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
    central_size, central_offset = eocd[5], eocd[6]
    if central_offset == 0xFFFFFFFF or central_size == 0xFFFFFFFF:
        raise RuntimeError("ZIP64 central directory is not supported")
    central = get_range(url, central_offset, central_offset + central_size - 1)
    print(json.dumps({"stage": "central_directory", "bytes": len(central)}), flush=True)
    target = config["source_member"].encode()
    entry = None
    offset = 0
    while offset < len(central):
        fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"Bad central header at {offset}")
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        name = central[offset + 46:offset + 46 + name_len]
        if name == target:
            entry = fields
            break
        offset += 46 + name_len + extra_len + comment_len
    if entry is None:
        raise RuntimeError(f"Missing ZIP member {target!r}")
    method, crc_expected = entry[4], entry[7]
    compressed_size, uncompressed_size, local_offset = entry[8], entry[9], entry[16]
    local = get_range(url, local_offset, local_offset + 4095)
    fields = struct.unpack_from("<4s5H3L2H", local, 0)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("Bad local header")
    data_start = local_offset + 30 + fields[9] + fields[10]
    compressed = get_range(url, data_start, data_start + compressed_size - 1)
    print(json.dumps({"stage": "compressed_member", "bytes": len(compressed), "uncompressed_bytes": uncompressed_size}), flush=True)
    if method == 8:
        raw = zlib.decompress(compressed, -15)
    elif method == 0:
        raw = compressed
    else:
        raise RuntimeError(f"Unsupported compression method {method}")
    if len(raw) != uncompressed_size:
        raise RuntimeError("Uncompressed size mismatch")
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    if crc != crc_expected:
        raise RuntimeError("CRC mismatch")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config["source_sha256"] or len(raw) != config["source_bytes"]:
        raise RuntimeError(f"Frozen source mismatch: bytes={len(raw)}, sha256={digest}")
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_bytes(raw)
    os.replace(temp, output)
    print(json.dumps({"output": str(output), "bytes": len(raw), "sha256": digest, "crc32": f"{crc:08x}", "zip_member": target.decode()}, indent=2))


if __name__ == "__main__":
    main()
