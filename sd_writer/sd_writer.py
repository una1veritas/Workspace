#!/usr/bin/env python3
'''
Created on 2025/04/20

@author: sin
'''
import subprocess
import sys

def run_subprocess(args : list):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            #timeout=10,
            #check=True,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE
        )
        # Check if the command was successful
        if result.returncode != 0:
            print(f"Failed to run {args}.")
            print(f"Error: {result.stderr.strip()}")
        return result.returncode
    # except subprocess.CalledProcessError as e:
    #     print("Command failed:")
    #     print(e.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out.")
    except FileNotFoundError:
        print("Error: command {args} does not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Prompt the user for the disk identifier
    #disk_id = input("Enter the disk identifier to unmount (e.g., disk2, disk2s1): ").strip()
    image_file = "rpi3+4-bullseye-6.1.21-sd16.img"
    if len(sys.argv) == 2 :
        print(f'Use the defau;t imagefile "{image_file}".')
        disk_id = sys.argv[1]
    elif len(sys.argv) == 3 :
        image_file = sys.argv[2]
        disk_id = sys.argv[1]
        print(f'Write {image_file} into {disk_id}.')
    else:
        exit(1)
    # Unmount the disk
    returncode = run_subprocess(["diskutil", "umountDisk", f"/dev/{disk_id}"])
    # Check if the command was successful
    if returncode != 0:
        print(f"Failed to unmount {disk_id}.")
        exit(1)
    
    ddcmd = ["sudo", "dd", f"if={image_file}", f"of=/dev/{disk_id}", "status=progress", "bs=1024"]
    returncode = run_subprocess(ddcmd)
        # Check if the command was successful
    if returncode != 0:
        print(f"Failed to copy to {disk_id}.")
        exit(1)

