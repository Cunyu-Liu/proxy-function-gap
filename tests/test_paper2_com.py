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


if __name__ == "__main__":
    unittest.main()
