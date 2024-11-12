import context
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats as stats
import torch

from nerf.util.util import resample


def calculate_normality_test(data, mu1, std1, alpha=0.05, visualize=False):
    """
    Tests if a set of 1D values comes from a normal distribution with mean mu1 and standard deviation std1.

    Parameters:
    - data: array-like, the sample data to test.
    - mu1: float, the mean of the normal distribution to test against.
    - std1: float, the standard deviation of the normal distribution to test against.
    - alpha: float, significance level for statistical tests (default is 0.05).

    Returns:
    - A dictionary with results of the tests and plots.
    """
    # Calculate sample statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)

    if visualize:
        # 1. Q-Q Plot
        plt.figure(figsize=(6, 6))
        stats.probplot(data, dist="norm", sparams=(mu1, std1), plot=plt)
        plt.title("Q-Q Plot")
        plt.show()

    # 2. Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    shapiro_result = shapiro_p > alpha

    # 3. Compare sample mean and standard deviation to mu1 and std1
    mean_close = np.isclose(sample_mean, mu1, atol=0.1 * std1)
    std_close = np.isclose(sample_std, std1, atol=0.1 * std1)

    # Print the results
    results = {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "mean_matches_mu1": mean_close,
        "std_matches_std1": std_close,
        "shapiro_statistic": shapiro_stat,
        "shapiro_p_value": shapiro_p,
        "shapiro_normal": shapiro_result,
        "conclusion": (
            "The data is likely from a normal distribution with the specified mean and std"
            if shapiro_result and mean_close and std_close
            else "The data is not likely from a normal distribution with the specified mean and std"
        ),
        "conclusion_flag": shapiro_result and mean_close and std_close,
    }

    return results


@pytest.mark.parametrize("mean,std", [(12, 2), (2, 0.1), (6, 1)])
def test_resample(mean, std):

    w_smp = 512
    t = torch.linspace(0, 20, w_smp).reshape(1, w_smp, 1)

    mean = 12.0
    std = 2.0

    w = torch.exp(-0.5 * ((t - mean) / std).pow(2)) / (
        std * torch.sqrt(2 * torch.tensor(np.pi))
    )

    t_smp = resample(w, t, 1000)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    results = calculate_normality_test(t_smp, mean, std)

    assert results["conclusion_flag"], "The resampled data is not normally distributed."
