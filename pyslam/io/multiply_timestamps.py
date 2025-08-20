#!/usr/bin/python3

import re
import argparse


def multiply_timestamps(input_file, output_file, timestamp_scale):
    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as f:
        for line in lines:
            items = line.split()
            timestamp = float(items[0])
            multiplied_timestamp = timestamp * float(timestamp_scale)
            print(f"multiplied_timestamp: {multiplied_timestamp}")
            new_line = " ".join([str(multiplied_timestamp)] + items[1:])
            f.write(new_line + "\n")


"""
usage: 
python multiply_timestamps.py input.txt output.txt timestampscale
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiply timestamps in a text file by one.")
    parser.add_argument("input_file", help="The input file.")
    parser.add_argument("output_file", help="The output file.")
    parser.add_argument("timestamp_scale", help="The scale for timestamps")
    args = parser.parse_args()

    multiply_timestamps(args.input_file, args.output_file, args.timestamp_scale)
