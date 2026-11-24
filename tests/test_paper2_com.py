import unittest

import torch

from proxy_function_gap.paper2_com import COMConfig, adversarial_negatives, conservative_loss


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 50))

    def forward(self, inputs):
        return (inputs * self.weight).sum(dim=(1, 2))


class Paper2COMTest(unittest.TestCase):
    def test_adversarial_relaxed_inputs_increase_prediction(self):
        torch.manual_seed(11)
        model = LinearModel()
        inputs = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 50)), 4).float().transpose(1, 2)
        initial = torch.softmax(torch.log(0.6 * inputs + 0.1), dim=1)
        before = model(initial).detach()
        negatives = adversarial_negatives(model, inputs, 0.01, 5, 0.6)
        after = model(negatives).detach()
        self.assertTrue(torch.all(after >= before))
        self.assertTrue(torch.allclose(negatives.sum(dim=1), torch.ones(3, 50)))

    def test_model_and_dual_gradients_match_separate_lagrangians(self):
        torch.manual_seed(37)
        model = LinearModel()
        inputs = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 50)), 4).float().transpose(1, 2)
        labels = torch.zeros(2)
        log_alpha = torch.nn.Parameter(torch.tensor(0.0))
        config = COMConfig(data_manifest="unused", output_root="unused", particle_steps=1)
        negatives = adversarial_negatives(model, inputs, config.particle_lr, config.particle_steps, config.soft_interpolation)
        positive = model(inputs)
        negative = model(negatives)
        alpha = torch.nn.functional.softplus(log_alpha)
        conservatism = negative - positive
        expected = torch.mean((positive - labels) ** 2) + alpha.detach() * conservatism.mean() + alpha * (config.overestimation_limit - conservatism.detach().mean())
        expected.backward()
        expected_model = model.weight.grad.detach().clone()
        expected_alpha = log_alpha.grad.detach().clone()
        model.weight.grad = None
        log_alpha.grad = None
        actual, _, _, _ = conservative_loss(model, inputs, labels, log_alpha, config)
        actual.backward()
        self.assertTrue(torch.allclose(model.weight.grad, expected_model))
        self.assertTrue(torch.allclose(log_alpha.grad, expected_alpha))

    def test_conservative_loss_has_dual_and_model_terms(self):
        torch.manual_seed(23)
        model = LinearModel()
        inputs = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 50)), 4).float().transpose(1, 2)
        labels = torch.zeros(2)
        log_alpha = torch.nn.Parameter(torch.tensor(0.0))
        config = COMConfig(data_manifest="unused", output_root="unused", particle_steps=1)
        loss, regression, conservatism, alpha = conservative_loss(model, inputs, labels, log_alpha, config)
        loss.backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(log_alpha.grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(alpha), 0)
        expected_alpha_gradient = torch.sigmoid(log_alpha.detach()) * (config.overestimation_limit - conservatism)
        self.assertTrue(torch.allclose(log_alpha.grad, expected_alpha_gradient, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
