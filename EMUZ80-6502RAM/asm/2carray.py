#!/usr/local/bin/python3

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("no input file name.")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("--") : 
                    continue
                elif line.startswith("DEPTH") :
                    inf["DEPTH"] = int(line.split("=")[1])
                elif line.startswith("DEPTH") :
                    inf["WIDTH"] = int(line.split("=")[1])
                elif line.startswith("DEPTH") :
                    inf["ADDRESS_RADIX"] = int(line.split("=")[1])
                elif line.startswith("DEPTH") :
                    inf["ADDRESS_RADIX"] = int(line.split("=")[1])
                elif line.startswith("DEPTH") :
                    inf["DATA_RADIX"] = int(line.split("=")[1])
                elif line.startswith("CONTENT BEGIN") :
                    continue
                left, right = line.split(":")
                print(left, right)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)
    