#!/usr/bin/env sh
# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set -eu

image=$1
device=$2
block=$3
confirm=$4

if [ -z "$device" ]; then
    echo "NN_SDCARD_DEVICE is required (example: /dev/sdb)."
    exit 2
fi
if [ ! -b "$device" ]; then
    echo "Not a block device: $device"
    exit 2
fi
if [ "$(lsblk -dnro TYPE "$device")" != "disk" ]; then
    echo "Refusing to flash a partition; select the whole SD card disk."
    exit 2
fi
if lsblk -nrpo MOUNTPOINT "$device" | grep -q '[^[:space:]]'; then
    echo "Refusing to flash: $device or one of its partitions is mounted."
    lsblk -o NAME,TYPE,SIZE,MOUNTPOINT "$device"
    exit 2
fi
if [ "$confirm" != "YES" ]; then
    echo "Flash check passed for $device."
    lsblk -o NAME,MODEL,TYPE,SIZE,MOUNTPOINT "$device"
    echo "Reconfigure with -DNN_FLASH_CONFIRM=YES to write $image at block $block."
    exit 2
fi

echo "Writing $image to $device, 512-byte block $block..."
sudo dd if="$image" of="$device" bs=512 seek="$block" conv=notrunc,fsync status=progress
echo "Flash completed."
