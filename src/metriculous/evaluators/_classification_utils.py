from typing import Union, Sequence

import numpy as np

NORMALIZATION_ABS_TOLERANCE = 1e-5
NORMALIZATION_REL_TOLERANCE = 1e-5


def check_normalization(probabilities: Union[np.ndarray, Sequence[float]], axis: int):
    np.testing.assert_allclose(
        np.sum(probabilities, axis=axis),
        desired=1.0,
        rtol=NORMALIZATION_REL_TOLERANCE,
        atol=NORMALIZATION_ABS_TOLERANCE,
    )
