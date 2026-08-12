#!/usr/bin/env sh
# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 linux-pk <cpu|vector|gemmini|all> <testbench> [profile]" >&2
    echo "       $0 baremetal <cpu|vector|gemmini> <testbench> [profile]" >&2
    exit 2
fi

platform=$1
backend=$2
testbench=$3
profile=${4:-}
cmake_bin=${CMAKE:-/usr/bin/cmake}

case "$backend" in
vector)
    auto_vectorize=ON
    ;;
cpu|gemmini|all)
    auto_vectorize=OFF
    ;;
*)
    echo "Unsupported backend: $backend" >&2
    exit 2
    ;;
esac

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
        -DNN_LINK_MODE=static -DNN_AUTO_VECTORIZE="$auto_vectorize"
    ;;
baremetal)
    if [ "$backend" = "all" ]; then
        echo "Unsupported backend for baremetal: all" >&2
        echo "Use cpu, vector, or gemmini." >&2
        exit 2
    fi
    if [ -z "$profile" ]; then
        if [ "$backend" = "gemmini" ]; then
            profile=GEMMINI
        elif [ "$backend" = "cpu" ]; then
            profile=cpu
        else
            profile=V128D128B
        fi
    fi
    build_dir="build/cmake/baremetal-${testbench}-${backend}-${profile}"
    "$cmake_bin" -S . -B "$build_dir" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-baremetal.cmake \
        -DNN_PLATFORM=baremetal -DNN_BACKEND="$backend" \
        -DNN_TESTBENCH="$testbench" -DNN_HARDWARE_CONFIG="$profile" \
        -DNN_AUTO_VECTORIZE="$auto_vectorize"
    ;;
*)
    echo "Unsupported platform: $platform" >&2
    exit 2
    ;;
esac

"$cmake_bin" --build "$build_dir" --parallel
echo "Ninja build directory: $build_dir"
