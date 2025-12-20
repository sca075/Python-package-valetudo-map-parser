#!/usr/bin/env python3
"""Debug binary data to find the correct robot position and angle."""

import os
import struct
import sys


# Add the SCR directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "SCR")))

from valetudo_map_parser.config.rand25_parser import RRMapParser


def hex_dump(data: bytes, start: int = 0, length: int = 64) -> str:
    """Create a hex dump of binary data."""
    result = []
    for i in range(0, min(length, len(data) - start), 16):
        offset = start + i
        hex_part = " ".join(
            f"{data[offset + j]:02x}" if offset + j < len(data) else "  "
            for j in range(16)
        )
        ascii_part = "".join(
            chr(data[offset + j])
            if offset + j < len(data) and 32 <= data[offset + j] <= 126
            else "."
            for j in range(16)
            if offset + j < len(data)
        )
        result.append(f"{offset:08x}: {hex_part:<48} |{ascii_part}|")
    return "\n".join(result)


def find_robot_blocks(payload: bytes):
    """Find all robot position and path blocks in the payload."""
    print("Searching for robot position (type 8) and path (type 3) blocks...")

    offset = 0x14  # Start after header
    robot_blocks = []

    while offset < len(payload) - 8:
        try:
            type_ = struct.unpack("<H", payload[offset : offset + 2])[0]
            hlength = struct.unpack("<H", payload[offset + 2 : offset + 4])[0]
            length = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]

            print(
                f"Block at offset {offset:04x}: type={type_}, hlength={hlength}, length={length}"
            )

            if type_ == 8:  # Robot position block
                robot_blocks.append((offset, hlength, length))
                print("  *** ROBOT POSITION BLOCK FOUND ***")

                # Show hex dump of the block
                print("  Block header + data:")
                print(hex_dump(payload, offset, hlength + length))
            elif type_ == 3:  # Path block
                print("  *** PATH BLOCK FOUND ***")

                # Show hex dump of the block
                print("  Block header + data:")
                print(
                    hex_dump(payload, offset, min(hlength + length, 128))
                )  # Limit to 128 bytes

                # Try to extract path angle
                if offset + 16 + 4 <= len(payload):
                    path_angle = struct.unpack(
                        "<I", payload[offset + 16 : offset + 20]
                    )[0]
                    print(f"  Path current_angle: {path_angle}")
                print()

                # Try different parsing approaches
                print("\n  PARSING ATTEMPTS:")

                # Original approach (16-bit at offset 8, 12)
                if offset + 8 + 4 <= len(payload):
                    x1 = struct.unpack("<H", payload[offset + 8 : offset + 10])[0]
                    y1 = struct.unpack("<H", payload[offset + 12 : offset + 14])[0]
                    angle1 = (
                        struct.unpack("<i", payload[offset + 16 : offset + 20])[0]
                        if offset + 20 <= len(payload)
                        else 0
                    )
                    print(
                        f"    Original (16-bit at 8,12): x={x1}, y={y1}, angle={angle1}"
                    )

                # Roborock approach (32-bit at block_data + 0, 4, 8)
                block_data_start = offset + 8
                if block_data_start + 12 <= len(payload):
                    x2 = struct.unpack(
                        "<I", payload[block_data_start : block_data_start + 4]
                    )[0]
                    y2 = struct.unpack(
                        "<I", payload[block_data_start + 4 : block_data_start + 8]
                    )[0]
                    angle2 = (
                        struct.unpack(
                            "<I", payload[block_data_start + 8 : block_data_start + 12]
                        )[0]
                        if block_data_start + 12 <= len(payload)
                        else 0
                    )
                    print(
                        f"    Roborock (32-bit at 0,4,8): x={x2}, y={y2}, angle={angle2}"
                    )

                    # Apply Roborock angle normalization
                    if angle2 > 0xFF:
                        normalized_angle2 = (angle2 & 0xFF) - 256
                    else:
                        normalized_angle2 = angle2
                    print(f"    Roborock normalized angle: {normalized_angle2}")

                # Try other offsets and data types
                for test_offset in [0, 4, 8, 12, 16, 20]:
                    if block_data_start + test_offset + 4 <= len(payload):
                        test_val = struct.unpack(
                            "<I",
                            payload[
                                block_data_start + test_offset : block_data_start
                                + test_offset
                                + 4
                            ],
                        )[0]
                        test_val_signed = struct.unpack(
                            "<i",
                            payload[
                                block_data_start + test_offset : block_data_start
                                + test_offset
                                + 4
                            ],
                        )[0]
                        print(
                            f"    32-bit at offset {test_offset}: {test_val} (0x{test_val:08x}) signed: {test_val_signed}"
                        )

                        # Check if this could be an angle (reasonable range)
                        if -360 <= test_val_signed <= 360:
                            print(
                                f"      *** POSSIBLE ANGLE: {test_val_signed} degrees ***"
                            )
                        if 0 <= test_val <= 360:
                            print(f"      *** POSSIBLE ANGLE: {test_val} degrees ***")

                print()

            # Move to next block
            offset = offset + hlength + length

        except (struct.error, IndexError) as e:
            print(f"Error at offset {offset:04x}: {e}")
            break

    return robot_blocks


def main():
    """Main function."""
    payload_file = "map_data_20250728_194519.bin"

    if not os.path.exists(payload_file):
        print(f"Payload file {payload_file} not found!")
        return

    with open(payload_file, "rb") as f:
        payload = f.read()

    print(f"Loaded payload: {len(payload)} bytes")
    print(f"Header: {payload[:20].hex()}")
    print()

    # Find robot blocks
    robot_blocks = find_robot_blocks(payload)

    print(f"\nFound {len(robot_blocks)} robot position blocks")

    # Compare with current parser
    print("\n" + "=" * 60)
    print("CURRENT PARSER RESULTS:")
    print("=" * 60)

    parser = RRMapParser()
    result = parser.parse_data(payload, pixels=False)

    if result:
        print(f"Robot: {result.get('robot', 'NOT FOUND')}")
        print(f"Robot angle: {result.get('robot_angle', 'NOT FOUND')}")
        print(f"Charger: {result.get('charger', 'NOT FOUND')}")
    else:
        print("Parser failed!")


if __name__ == "__main__":
    main()
