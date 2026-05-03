"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os


def permissive_makedirs(dir):
    """Make a directory, including any missing parent directories, with 777 permissions."""

    # Unset the current process's umask so intermediate directories aren't created with default permissions
    os.umask(0)

    # Create the directory, without raising an error if it already exists, with permissions 777
    os.makedirs(dir, exist_ok=True, mode=0o777)
