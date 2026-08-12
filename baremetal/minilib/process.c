/* SPDX-FileContributor: Person: Stanley Lee */
/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

extern void _exit(int status);

void exit(int status) {
    _exit(status);
}
