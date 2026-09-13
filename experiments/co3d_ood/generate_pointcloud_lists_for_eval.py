#!/usr/bin/env python3
"""
Scan input folder for .ply files and write their absolute paths to output file.
"""
import argparse
import os
from pathlib import Path


def find_ply_files(input_folder):
    """Find all .ply files in input folder recursively."""
    ply_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.abspath(os.path.join(root, file)))
    return sorted(ply_files)


def main(input_folder, output_file):
    """Scan for .ply files and write absolute paths to output file."""
    print(f"Scanning {input_folder} for .ply files...")

    ply_files = find_ply_files(input_folder)

    print(f"Found {len(ply_files)} .ply files")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to output file
    with open(output_file, 'w') as f:
        for ply_path in ply_files:
            f.write(f"{ply_path}\n")

    print(f"Wrote {len(ply_files)} paths to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pointcloud list for evaluation")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing .ply files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output text file to write paths")

    args = parser.parse_args()
    main(args.input_folder, args.output_file)
