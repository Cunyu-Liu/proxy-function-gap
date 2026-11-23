import ast
import io
import struct
import zipfile


def _header(descr, shape):
    if shape:
        shape_text = "(" + ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"
    else:
        shape_text = "()"
    text = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_text + ", }"
    prefix = b"\x93NUMPY\x01\x00"
    padding = (64 - (len(prefix) + 2 + len(text) + 1) % 64) % 64
    header = (text + " " * padding + "\n").encode("latin1")
    return prefix + struct.pack("<H", len(header)) + header


def write_npy(handle, descr, shape, data):
    handle.write(_header(descr, shape))
    handle.write(data)


def unicode_array(values):
    values = list(values)
    width = max(1, max((len(value) for value in values), default=1))
    payload = "".join(value + "\x00" * (width - len(value)) for value in values).encode("utf-32-le")
    return f"<U{width}", (len(values),), payload


def uint8_array(values):
    return "|u1", (len(values),), bytes(values)


def float64_array(values):
    return "<f8", (len(values),), b"".join(struct.pack("<d", float(value) if value not in (None, "") else float("nan")) for value in values)


def bool_array(value):
    return "|b1", (), bytes([1 if value else 0])


def one_hot(sequences):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    sequences = list(sequences)
    payload = bytearray()
    for sequence in sequences:
        if len(sequence) != 50:
            raise ValueError("UTR50 required")
        for base in sequence:
            one_hot = [0, 0, 0, 0]
            one_hot[mapping[base]] = 1
            payload.extend(one_hot)
    return "|u1", (len(sequences), 50, 4), bytes(payload)


def write_npz(path, arrays):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, (descr, shape, data) in arrays.items():
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            with archive.open(entry, "w") as handle:
                write_npy(handle, descr, shape, data)


def read_npy(handle):
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("Not an NPY file")
    major, minor = handle.read(2)
    if major != 1:
        raise ValueError("Unsupported NPY version")
    header_length = struct.unpack("<H", handle.read(2))[0]
    metadata = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
    shape = metadata["shape"] if isinstance(metadata["shape"], tuple) else (metadata["shape"],)
    return metadata["descr"], shape, handle.read()


def read_npz(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as handle:
                result[info.filename.removesuffix(".npy")] = read_npy(handle)
    return result


def decode_unicode(descr, shape, data):
    width = int(descr[2:])
    count = shape[0]
    raw = data.decode("utf-32-le")
    return [raw[index * width:(index + 1) * width].rstrip("\x00") for index in range(count)]


def decode_float64(data):
    return [value[0] for value in struct.iter_unpack("<d", data)]
