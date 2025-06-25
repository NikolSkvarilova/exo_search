import numpy as np
import gpflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import scienceplots

plt.style.use(["science", "no-latex"])

"""Create figures for the Gaussian processes chapter.
"""

def plot_prior_posterior(x: np.ndarray, y: np.ndarray, n_samples: int = 3):

    # Create kernel
    kernel = gpflow.kernels.SquaredExponential(lengthscales=0.1, variance=0.5)

    # Plot prior samples
    plot_kernel(kernel, "Prior samples", fig_name="prior_samples")

    # Create and train model
    model = gpflow.models.GPR((x, y), kernel=kernel)
    Xplot = np.linspace(min(x[:, 0]) - 0.2, max(x[:, 0]) + 0.2, 250)[:, None]

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=1000)
    )

    # Posterior samples
    trained_m_samples = model.predict_f_samples(Xplot, n_samples)

    # Model
    m1, var1 = model.predict_f(Xplot, full_cov=False)
    m2, var2 = model.predict_y(Xplot)

    s1 = 2 * np.sqrt(var1)
    s2 = 2 * np.sqrt(var2)

    # Plot posterior samples and model
    _, ax = plt.subplots(dpi=300)
    plt.title("Posterior samples + Mean")
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(n_samples):
        ax.plot(
            Xplot[:, 0],
            trained_m_samples[i],
            color="blue",
            linewidth=0.8,
            linestyle="dashed",
        )

    plt.scatter(x, y, s=3, color="black")

    plt.fill_between(
        Xplot[:, 0],
        m1[:, 0] + s1[:, 0],
        m1[:, 0] - s1[:, 0],
        alpha=0.1,
        color="blue",
        edgecolors="none",
        label="95% Confidence interval",
    )
    plt.tight_layout()
    ax.plot(Xplot, m1, color="blue")

    plt.savefig("figures/kernels/posterior_samples.pdf")
    plt.show()
    plt.close()


def plot_kernel(kernel, kernel_name, n_samples=3, fig_name=None, file_format="pdf"):
    # Create model
    Xplot = np.linspace(-2, 2, 250)[:, None]
    x = np.array([[1.0]])
    y = np.array([[2.0]])
    model = gpflow.models.GPR((x, y), kernel=kernel)

    # Predict samples
    untrained_m_samples = model.predict_f_samples(Xplot, n_samples)

    # Plot samples
    _, ax = plt.subplots(dpi=300)
    lines = ["solid", "dotted", "dashed"]
    for i in range(n_samples):
        ax.plot(
            Xplot[:, 0],
            untrained_m_samples[i],
            linewidth=0.7,
            color="blue",
            linestyle=lines[i % 3],
        )

    ax.set_title(kernel_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(top=2, bottom=-2)

    plt.tight_layout()
    plt.savefig(f"figures/kernels/{fig_name}.{file_format}")

    plt.show()
    plt.close()


def plot_mean(
    title,
    mean_function=None,
    x=None,
    y=None,
    x_offset_left=2.0,
    x_offset_right=2.0,
    fig_name=None,
):
    # Create model
    Xplot = np.linspace(x[0][0] - x_offset_left, x[:, 0][-1] + x_offset_right, 100)[
        :, None
    ]
    model = gpflow.models.GPR(
        (x, y),
        kernel=gpflow.kernels.SquaredExponential(variance=15, lengthscales=0.5),
        mean_function=mean_function,
    )

    # Train model
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=1000)
    )

    # Predict
    m1, var1 = model.predict_f(Xplot, full_cov=False)
    s1 = 2 * np.sqrt(var1)

    # Plot model
    plt.figure(dpi=300)
    plt.scatter(x, y, c="black", label="Data", s=3)
    plt.plot(Xplot[:, 0], m1, color="blue", linewidth=0.5, label="Trained mean")
    plt.fill_between(
        Xplot[:, 0],
        m1[:, 0] + s1[:, 0],
        m1[:, 0] - s1[:, 0],
        alpha=0.1,
        color="blue",
        edgecolors="none",
        label="95% Confidence interval",
    )
    plt.ylim(top=2, bottom=-2.5)
    plt.title(title)

    # Plot the mean
    plt.hlines(
        0,
        x[0][0] - x_offset_left,
        x[:, 0][-1] + x_offset_right,
        linestyles="dashed",
        color="blue",
        linewidth=0.5,
        alpha=0.6,
        label="Mean function",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figures/{fig_name}.pdf")
    plt.show()


def plot_covariance_matrix(kernel, title, fig_name=None):
    # Plot covariance matrix
    x = np.arange(0, 5, 0.1)[:, None]
    covariance_matrix = np.array(kernel(x))
    plt.figure(dpi=300)
    sns.heatmap(covariance_matrix, cmap="coolwarm", square=True)
    plt.title(title)
    plt.gca().invert_yaxis()
    # plt.axis("square")
    plt.xticks(ticks=np.arange(0, 50, 10)[1:], labels=x[:, 0][::10][1:])
    plt.yticks(ticks=np.arange(0, 50, 10)[1:], labels=x[:, 0][::10][1:])
    plt.xlabel("$x$")
    plt.ylabel("$x'$")
    plt.tight_layout()
    plt.savefig(f"./figures/matrix/{fig_name}.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    x = np.array([[0.3], [0.6], [0.9], [2.3]])
    y = np.array([[0.22], [0.25], [0.1], [0.55]])

    figures_path = Path("figures/kernels/")
    figures_path.mkdir(parents=True, exist_ok=True)

    plot_kernel(
        gpflow.kernels.SquaredExponential(lengthscales=0.2, variance=0.4),
        "SE",
        fig_name="se_kernel",
    )
 