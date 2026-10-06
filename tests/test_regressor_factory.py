"""Tests for the backend-neutral SymbolicModel regressor factory."""

import sys
import unittest

import numpy as np

from symtorch import SymbolicModel


class FakeRegressor:
    """Minimal regressor that records its parameters and fit data."""

    def __init__(self, params):
        self.params = params
        self.fit_args = None

    def fit(self, X, y, **kwargs):
        self.fit_args = (X, y, kwargs)
        return self

    def get_best(self):
        return {"equation": "x0", "loss": 0.0}


class FakeRegressorFactory:
    """Create and record fake regressors."""

    def __init__(self):
        self.regressors = []

    def __call__(self, **params):
        regressor = FakeRegressor(params)
        self.regressors.append(regressor)
        return regressor


class RegressorFactoryTests(unittest.TestCase):
    """Verify injected regressors do not load the default PySR backend."""

    def setUp(self):
        self.inputs = np.arange(20, dtype=float).reshape(5, 4)
        self.assert_backend_modules_not_loaded()

    def assert_backend_modules_not_loaded(self):
        loaded = set(sys.modules)
        self.assertFalse(any(name == "pysr" or name.startswith("pysr.") for name in loaded))
        self.assertFalse(
            any(name == "juliacall" or name.startswith("juliacall.") for name in loaded)
        )

    def test_callable_scalar_output_uses_injected_factory(self):
        factory = FakeRegressorFactory()

        def scalar_output(x):
            return x[:, 0] ** 2 + x[:, 1]

        model = SymbolicModel(
            scalar_output,
            block_name="fake_scalar",
            regressor_factory=factory,
        )
        result = model.distill(self.inputs, sr_params={"niterations": 0})

        self.assertIs(model.regressor_factory, factory)
        self.assertEqual(result, {0: factory.regressors[0]})
        self.assertEqual(len(factory.regressors), 1)
        self.assertEqual(factory.regressors[0].params["niterations"], 0)
        np.testing.assert_array_equal(factory.regressors[0].fit_args[0], self.inputs)
        np.testing.assert_array_equal(
            factory.regressors[0].fit_args[1],
            scalar_output(self.inputs),
        )
        self.assert_backend_modules_not_loaded()

    def test_selected_output_uses_one_injected_regressor(self):
        factory = FakeRegressorFactory()

        def multi_output(x):
            return np.column_stack([x[:, 0] ** 2, x[:, 1] - x[:, 2]])

        model = SymbolicModel(
            multi_output,
            block_name="fake_selected_output",
            regressor_factory=factory,
        )
        result = model.distill(
            self.inputs,
            output_dim=1,
            sr_params={"niterations": 0},
        )

        self.assertIs(result, factory.regressors[0])
        self.assertEqual(list(model.pysr_regressor), [1])
        self.assertEqual(len(factory.regressors), 1)
        np.testing.assert_array_equal(
            factory.regressors[0].fit_args[1],
            multi_output(self.inputs)[:, 1],
        )
        self.assert_backend_modules_not_loaded()


if __name__ == "__main__":
    unittest.main()
