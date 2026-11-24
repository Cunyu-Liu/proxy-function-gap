import math

import numpy as np
import torch

from .paper3_headroom import fit_model
from .paper3_headroom_v4 import regret_analytic_ladder_cuda
from .trainer.cuda import configure_cuda


def tiny_model_and_features(config):
    runtime = config.runtime(1.0, 0.04)
    development_x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    development_y = np.array([0.2, -0.1, 0.3, 0.05])
    features = np.array([[1.0, 0.2], [0.1, 1.0], [0.8, 0.7]])
    model, training = fit_model(development_x, development_y, runtime)
    return model, features, training


def scalar_independent_expected_max(mean, variance, nodes, weights):
    sigma = np.sqrt(variance)
    expected = 0.0
    for i in range(len(mean)):
        for node, weight in zip(nodes, weights):
            value = mean[i] + sigma[i] * node
            log_product = 0.0
            for j in range(len(mean)):
                if i == j:
                    continue
                z = (value - mean[j]) / sigma[j]
                cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
                log_product += math.log(max(cdf, 1e-300))
            expected += weight * value * math.exp(log_product)
    return expected


def exhaustive_scores(model, features, noise_variance, count=64):
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    scores = []
    deployments = []
    for feature in features:
        query_mean, query_variance = model.predict(feature[None, :], include_noise=True)
        regrets = []
        choices = []
        for node in nodes:
            fantasy = query_mean[0] + np.sqrt(query_variance[0]) * node
            updated = model.update(feature, fantasy)
            means, variances = updated.predict(features, include_noise=False)
            expected_max = scalar_independent_expected_max(means, variances, nodes, weights)
            choice = int(np.argmax(means))
            choices.append(choice)
            regrets.append(expected_max - means[choice])
        scores.append(-float(np.dot(weights, regrets)))
        deployments.append(choices)
    return np.asarray(scores), deployments


def two_candidate_closed_form_scores(model, features, noise_variance, count=64):
    evaluation = features[:2]
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    scores = []
    for feature in features:
        query_mean, query_variance = model.predict(feature[None, :], include_noise=True)
        regrets = []
        for node in nodes:
            fantasy = query_mean[0] + np.sqrt(query_variance[0]) * node
            updated = model.update(feature, fantasy)
            means, variances = updated.predict(evaluation, include_noise=False)
            choice = int(np.argmax(means))
            other = 1 - choice
            delta = means[choice] - means[other]
            sigma = math.sqrt(max(variances[choice] + variances[other], 0.0))
            regret = sigma / math.sqrt(2 * math.pi) * math.exp(-0.5 * (delta / sigma) ** 2) - delta * 0.5 * math.erfc(delta / (math.sqrt(2) * sigma))
            regrets.append(regret)
        scores.append(-float(np.dot(weights, regrets)))
    return np.asarray(scores)


def nested_independent_mc_cuda(model, features, config):
    device, dtype, gpu = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, config.validation_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.validation_seed)
    mean = torch.as_tensor(model.mean, device=device, dtype=dtype)
    covariance = torch.as_tensor(model.covariance, device=device, dtype=dtype)
    evaluation = torch.as_tensor(features, device=device, dtype=dtype)
    evaluation_mean = evaluation @ mean
    evaluation_variance = torch.einsum("ij,jk,ik->i", evaluation, covariance, evaluation).clamp_min(0)
    query_covariance = evaluation @ covariance @ evaluation.T
    label_count = config.validation_label_mc_samples
    posterior_count = config.validation_posterior_mc_samples
    z_half = torch.randn(label_count // 2, device=device, dtype=dtype, generator=generator)
    label_z = torch.cat((z_half, -z_half))
    epsilon_half = torch.randn((posterior_count // 2, len(features)), device=device, dtype=dtype, generator=generator)
    epsilon = torch.cat((epsilon_half, -epsilon_half), dim=0)
    scores = []
    standard_errors = []
    for index in range(len(features)):
        cross = query_covariance[:, index]
        predictive_variance = query_covariance[index, index] + float(config.runtime(1.0, 0.04).noise_variance)
        predictive_scale = torch.sqrt(predictive_variance)
        updated_mean = evaluation_mean[None, :] + cross[None, :] * label_z[:, None] / predictive_scale
        updated_variance = (evaluation_variance - cross.square() / predictive_variance).clamp_min(torch.finfo(dtype).tiny)
        updated_sigma = torch.sqrt(updated_variance)
        deployed_mean = updated_mean.max(dim=1).values
        label_regrets = []
        for start in range(0, label_count, 16):
            stop = min(start + 16, label_count)
            outcomes = (
                updated_mean[start:stop, None, :]
                + updated_sigma[None, None, :] * epsilon[None, :, :]
            )
            per_label = (outcomes.max(dim=2).values - deployed_mean[start:stop, None]).mean(dim=1)
            label_regrets.append(per_label)
        regrets = torch.cat(label_regrets)
        scores.append(-regrets.mean())
        standard_errors.append(regrets.std(unbiased=True) / math.sqrt(label_count))
    scores_tensor = torch.stack(scores)
    errors_tensor = torch.stack(standard_errors)
    torch.cuda.synchronize(device)
    if not scores_tensor.is_cuda or not torch.isfinite(scores_tensor).all() or not torch.isfinite(errors_tensor).all():
        raise RuntimeError("invalid nested independent CUDA Monte Carlo")
    return scores_tensor, errors_tensor, {
        "device": str(device), "gpu": gpu, "cpu_fallback": False,
        "label_mc_samples": label_count, "posterior_mc_samples": posterior_count,
        "antithetic_labels": True, "antithetic_posterior": True,
        "posterior_dependence": "diagonal_independent_fantasy_posterior_approximation",
        "standard_error_definition": "nominal_standard_error_over_antithetic_label_regrets",
    }


def run_numerical_validation(config):
    model, features, training = tiny_model_and_features(config)
    analytic, metadata = regret_analytic_ladder_cuda(model, features, features, config, 0.04)
    analytic_np = analytic.detach().cpu().numpy()
    exhaustive, deployments = exhaustive_scores(model, features, 0.04)
    closed_form = two_candidate_closed_form_scores(model, features, 0.04)
    analytic_two, _ = regret_analytic_ladder_cuda(model, features, features[:2], config, 0.04)
    analytic_two_np = analytic_two.detach().cpu().numpy()
    mc, standard_errors, mc_metadata = nested_independent_mc_cuda(model, features, config)
    mc_np = mc.detach().cpu().numpy()
    se_np = standard_errors.detach().cpu().numpy()
    exhaustive_error = np.abs(analytic_np - exhaustive)
    closed_form_error = np.abs(analytic_two_np - closed_form)
    mc_error = np.abs(analytic_np - mc_np)
    if np.max(exhaustive_error) > 1e-8 or np.max(closed_form_error) > 1e-8:
        raise RuntimeError("analytic regret failed exhaustive or two-candidate closed-form validation")
    if np.any(mc_error > np.maximum(0.02, 5 * se_np)):
        raise RuntimeError("analytic regret failed frozen high-sample Monte Carlo validation")
    return {
        "schema_version": "paper3-headroom-v4-numerical-validation-v1",
        "development_features": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        "development_labels": [0.2, -0.1, 0.3, 0.05],
        "evaluation_features": features.tolist(), "prior_variance": 1.0, "noise_variance": 0.04,
        "training_device": training["training_device"],
        "analytic_scores": analytic_np.tolist(), "analytic_metadata": metadata,
        "exhaustive_scores": exhaustive.tolist(),
        "exhaustive_deployment_choices_by_label_node": deployments,
        "exhaustive_absolute_errors": exhaustive_error.tolist(),
        "two_candidate_analytic_scores": analytic_two_np.tolist(),
        "two_candidate_closed_form_scores": closed_form.tolist(),
        "two_candidate_absolute_errors": closed_form_error.tolist(),
        "nested_mc_scores": mc_np.tolist(), "nested_mc_absolute_errors": mc_error.tolist(),
        "nested_mc_nominal_standard_errors": se_np.tolist(), "nested_mc_metadata": mc_metadata,
        "exhaustive_tolerance": 1e-8, "two_candidate_tolerance": 1e-8,
        "mc_tolerance_rule": "absolute_error<=max(0.02,5*nominal_standard_error)",
        "passed": True, "outer_labels_read": False,
    }
