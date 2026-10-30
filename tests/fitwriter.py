"""Minimal FIT encoder (test-only): writes `hrv` messages (global 78).

Enough of the FIT protocol to round-trip beat intervals through a real
decoder (fitdecode): 14-byte header with CRC, one definition message,
N data messages, trailing file CRC.
"""

import struct

_CRC_TABLE = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
              0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]


def fit_crc(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def write_hrv_fit(path, rr_seconds, per_msg=5):
    recs = bytearray()
    # file_id (global 0): field 0 type enum=4 (activity)
    recs += bytes([0x41, 0, 0]) + struct.pack("<HB", 0, 1) + bytes([0, 1, 0x00])
    recs += bytes([0x01, 4])
    # hrv definition: local 0, global 78, field 0 'time' uint16[per_msg] (base 0x84)
    recs += bytes([0x40, 0, 0]) + struct.pack("<HB", 78, 1) + bytes([0, 2 * per_msg, 0x84])
    vals = [round(r * 1000) for r in rr_seconds]
    for i in range(0, len(vals), per_msg):
        chunk = vals[i:i + per_msg]
        chunk += [0xFFFF] * (per_msg - len(chunk))  # invalid padding
        recs += bytes([0x00]) + struct.pack("<" + "H" * per_msg, *chunk)
    header = struct.pack("<BBHI4s", 14, 0x10, 2132, len(recs), b".FIT")
    header += struct.pack("<H", fit_crc(header))
    body = header + bytes(recs)
    with open(path, "wb") as f:
        f.write(body + struct.pack("<H", fit_crc(body)))
