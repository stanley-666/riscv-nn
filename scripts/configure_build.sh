#!/usr/bin/env sh
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <linux-pk|baremetal> <cpu|vector|gemmini|all> <testbench> [profile]"
    exit 2
fi

platform=$1
backend=$2
testbench=$3
profile=${4:-}
cmake_bin=${CMAKE:-/usr/bin/cmake}

case "$platform" in
linux-pk)
    if [ -z "$profile" ]; then
        if [ "$backend" = "cpu" ] || [ "$backend" = "gemmini" ]; then
            profile=default
        else
            profile=zvl128b
        fi
    fi
    build_dir="build/cmake/linux-pk-${testbench}-${backend}-${profile}"
    "$cmake_bin" -S . -B "$build_dir" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-linux-gnu.cmake \
        -DNN_PLATFORM=linux-pk -DNN_BACKEND="$backend" \
        -DNN_TESTBENCH="$testbench" -DNN_CONFIG="$profile" \
        -DNN_LINK_MODE=static -DNN_AUTO_VECTORIZE=OFF
    ;;
baremetal)
    if [ -z "$profile" ]; then
        if [ "$backend" = "gemmini" ]; then
            profile=GEMMINI
        else
            profile=V128D128B
        fi
    fi
    build_dir="build/cmake/baremetal-${testbench}-${backend}-${profile}"
    "$cmake_bin" -S . -B "$build_dir" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-baremetal.cmake \
        -DNN_PLATFORM=baremetal -DNN_BACKEND="$backend" \
        -DNN_TESTBENCH="$testbench" -DNN_HARDWARE_CONFIG="$profile" \
        -DNN_AUTO_VECTORIZE=OFF
    ;;
*)
    echo "Unsupported platform: $platform"
    exit 2
    ;;
esac

"$cmake_bin" --build "$build_dir" --parallel
echo "Ninja build directory: $build_dir"
