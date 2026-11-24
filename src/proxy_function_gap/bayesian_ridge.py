import numpy as np


def _features_2d(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("features must be a finite 2D array with at least one column")
    return array


def _labels_1d(values, size):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("labels must be finite and match the feature rows")
    return array


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class BayesianRidge:
    def __init__(
        self,
        prior_mean,
        prior_covariance,
        noise_variance,
        mean,
        covariance,
        train_features,
        train_labels,
    ):
        self.prior_mean = np.array(prior_mean, dtype=np.float64, copy=True)
        self.prior_covariance = np.array(prior_covariance, dtype=np.float64, copy=True)
        self.noise_variance = float(noise_variance)
        self._mean = np.array(mean, dtype=np.float64, copy=True)
        self._covariance = np.array(covariance, dtype=np.float64, copy=True)
        self._train_features = np.array(train_features, dtype=np.float64, copy=True)
        self._train_labels = np.array(train_labels, dtype=np.float64, copy=True)

    @classmethod
    def fit(cls, features, labels, prior_mean=None, prior_covariance=None, noise_variance=1.0):
        features = _features_2d(features)
        labels = _labels_1d(labels, len(features))
        dimension = features.shape[1]
        if prior_mean is None:
            prior_mean = np.zeros(dimension)
        else:
            prior_mean = np.asarray(prior_mean, dtype=np.float64)
        if prior_mean.shape != (dimension,) or not np.isfinite(prior_mean).all():
            raise ValueError("prior_mean must be finite and match the feature dimension")
        if prior_covariance is None:
            prior_covariance = np.eye(dimension)
        else:
            prior_covariance = np.asarray(prior_covariance, dtype=np.float64)
        if (
            prior_covariance.shape != (dimension, dimension)
            or not np.isfinite(prior_covariance).all()
            or not np.allclose(prior_covariance, prior_covariance.T, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("prior_covariance must be finite, square and symmetric")
        noise_variance = float(noise_variance)
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        prior_precision = _spd_inverse(prior_covariance)
        precision = prior_precision + features.T @ features / noise_variance
        covariance = _spd_inverse(precision)
        right_hand_side = prior_precision @ prior_mean + features.T @ labels / noise_variance
        mean = covariance @ right_hand_side
        return cls(
            prior_mean,
            prior_covariance,
            noise_variance,
            mean,
            covariance,
            features,
            labels,
        )

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    @property
    def train_features(self):
        return self._train_features.copy()

    @property
    def train_labels(self):
        return self._train_labels.copy()

    @property
    def observation_count(self):
        return len(self._train_labels)

    def refit(self):
        return self.fit(
            self._train_features,
            self._train_labels,
            self.prior_mean,
            self.prior_covariance,
            self.noise_variance,
        )

    def update(self, feature, label):
        feature = np.asarray(feature, dtype=np.float64)
        if feature.shape != (self._mean.size,) or not np.isfinite(feature).all():
            raise ValueError("feature must be finite and match the model dimension")
        label = float(label)
        if not np.isfinite(label):
            raise ValueError("label must be finite")
        covariance_feature = self._covariance @ feature
        denominator = self.noise_variance + feature @ covariance_feature
        mean = self._mean + covariance_feature * (label - feature @ self._mean) / denominator
        covariance = self._covariance - np.outer(covariance_feature, covariance_feature) / denominator
        covariance = (covariance + covariance.T) / 2.0
        return BayesianRidge(
            self.prior_mean,
            self.prior_covariance,
            self.noise_variance,
            mean,
            covariance,
            np.vstack((self._train_features, feature)),
            np.append(self._train_labels, label),
        )

    def predict(self, features, include_noise=True):
        features = _features_2d(features)
        if features.shape[1] != self._mean.size:
            raise ValueError("feature dimension does not match the model")
        mean = features @ self._mean
        variance = np.einsum("ij,jk,ik->i", features, self._covariance, features)
        variance = np.maximum(variance, 0.0)
        if include_noise:
            variance = variance + self.noise_variance
        return mean, variance

    def sample_weights(self, count, rng):
        _positive_integer(count, "count")
        factor = np.linalg.cholesky(self._covariance)
        normal = rng.standard_normal((count, self._mean.size))
        return self._mean + normal @ factor.T


def _spd_inverse(matrix):
    factor = np.linalg.cholesky(matrix)
    identity = np.eye(len(matrix))
    return np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
