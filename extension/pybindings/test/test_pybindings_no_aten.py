# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest

import _C as runtime

import numpy as np


class PybindingsNoAtenTest(unittest.TestCase):
    def test_import_does_not_load_torch(self) -> None:
        self.assertFalse(runtime._uses_aten)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("executorch.exir", sys.modules)

    def test_tensor_from_numpy_and_list(self) -> None:
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_array_equal(runtime.Tensor(array).numpy(), array)
        np.testing.assert_array_equal(
            runtime.Tensor([[1, 2], [3, 4]], dtype=np.int32).numpy(),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
