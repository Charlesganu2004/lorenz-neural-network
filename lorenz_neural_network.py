"""Learn the Lorenz system with a Keras network and roll it forward on its own.

The network takes the state (x, y, z) and the parameter rho and predicts the
state one time step later. It is trained on odeint trajectories at rho = 10,
28 and 40, then rolled forward from fresh starting points at rho = 17 and
rho = 35 by feeding each prediction back in as the next input. The rollouts
are compared with odeint from the same starting points.
"""
import math
import os
import time

import keras
import matplotlib
import numpy as np
from keras import layers
from scipy.integrate import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 0
SIGMA = 10
BETA = 8 / 3
DT = 0.01

TRAIN_RHOS = (10, 28, 40)
TEST_RHOS = (17, 35)
N_TRAIN_PER_RHO = 300
N_VAL_PER_RHO = 30
N_TRAIN_TIMESTEPS = 300

HIDDEN_UNITS = 64
HIDDEN_LAYERS = 3
ACTIVATION = "swish"
EPOCHS = 40
BATCH_SIZE = 256
INITIAL_LEARNING_RATE = 0.002
FINAL_LEARNING_RATE = 0.00002

N_ROLLOUT_STEPS = 1000
N_ROLLOUT_STARTS = 20
ERROR_THRESHOLD = 1.0

# Training and validation starts use seeds 0 to 989. The network settings were
# chosen by comparing rollouts from seeds 1000 to 1019, so the reported rollouts
# start from seeds 5000 to 5019, which played no part in choosing the settings.
ROLLOUT_SEED_BASE = 5000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures")
MODEL_PATH = os.path.join(HERE, "lorenz_network.keras")


def lorenz(x, t, sigma=SIGMA, beta=BETA, rho=28):
    x_dot = sigma * (x[1] - x[0])
    y_dot = x[0] * (rho - x[2]) - x[1]
    z_dot = x[0] * x[1] - beta * x[2]
    return [x_dot, y_dot, z_dot]


def random_start(seed):
    return np.random.default_rng(seed).uniform(-15, 15, 3)


def integrate(x0, rho, n_timesteps, dt=DT):
    t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
    return odeint(lorenz, x0, t, args=(SIGMA, BETA, rho))


def generate_data(rho, seeds, n_timesteps, dt=DT):
    """One odeint trajectory per seed at the given rho, shape (n, n_timesteps, 3)."""
    return np.stack([integrate(random_start(s), rho, n_timesteps, dt) for s in seeds])


def make_pairs(data, rho):
    """One-step slicing: inputs (x, y, z, rho) at steps 0 to n-2, targets (x, y, z) at steps 1 to n-1."""
    n_samples, n_timesteps, _ = data.shape
    n_pairs = n_timesteps - 1
    X = np.zeros((n_samples * n_pairs, 4))
    Y = np.zeros((n_samples * n_pairs, 3))
    for i in range(n_samples):
        X[i * n_pairs:(i + 1) * n_pairs, :3] = data[i, :-1]
        Y[i * n_pairs:(i + 1) * n_pairs] = data[i, 1:]
    X[:, 3] = rho
    return X, Y


def build_dataset():
    """Training and validation pairs. Validation trajectories are held out whole."""
    train_X, train_Y, val_X, val_Y, val_rho = [], [], [], [], []
    seed = 0
    for rho in TRAIN_RHOS:
        train_seeds = range(seed, seed + N_TRAIN_PER_RHO)
        val_seeds = range(seed + N_TRAIN_PER_RHO, seed + N_TRAIN_PER_RHO + N_VAL_PER_RHO)
        seed += N_TRAIN_PER_RHO + N_VAL_PER_RHO
        X, Y = make_pairs(generate_data(rho, train_seeds, N_TRAIN_TIMESTEPS), rho)
        train_X.append(X)
        train_Y.append(Y)
        X, Y = make_pairs(generate_data(rho, val_seeds, N_TRAIN_TIMESTEPS), rho)
        val_X.append(X)
        val_Y.append(Y)
        val_rho.append(np.full(len(X), rho))
    return (np.concatenate(train_X), np.concatenate(train_Y),
            np.concatenate(val_X), np.concatenate(val_Y), np.concatenate(val_rho))


def build_model(train_X):
    """Maps (x, y, z, rho) to the next (x, y, z).

    The inputs are standardised with the training mean and standard deviation.
    The hidden layers produce a change in the state, which is added to the
    current state to give the prediction.
    """
    norm = layers.Normalization()
    norm.adapt(train_X)
    inputs = keras.Input(shape=(4,))
    state = inputs[:, :3]
    h = norm(inputs)
    for _ in range(HIDDEN_LAYERS):
        h = layers.Dense(HIDDEN_UNITS, activation=ACTIVATION)(h)
    step = layers.Dense(3)(h)
    outputs = layers.Add()([state, step])
    model = keras.Model(inputs, outputs)

    total_steps = EPOCHS * math.ceil(len(train_X) / BATCH_SIZE)
    schedule = keras.optimizers.schedules.CosineDecay(
        INITIAL_LEARNING_RATE, total_steps,
        alpha=FINAL_LEARNING_RATE / INITIAL_LEARNING_RATE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=schedule),
                  loss="mean_squared_error")
    return model


def rollout(model, x0, rho, n_steps):
    """Feed the network's own prediction back in as the next input.

    x0 has shape (n_starts, 3). Returns an array of shape (n_starts, n_steps, 3).
    """
    n_starts = len(x0)
    states = np.zeros((n_starts, n_steps, 3))
    states[:, 0] = x0
    rho_col = np.full((n_starts, 1), rho, dtype="float32")
    for i in range(1, n_steps):
        inputs = np.concatenate([states[:, i - 1].astype("float32"), rho_col], axis=1)
        states[:, i] = model(inputs, training=False).numpy()
    return states


def first_exceedance(error, threshold=ERROR_THRESHOLD):
    """Index of the first step where error exceeds threshold, or None if it never does."""
    above = np.nonzero(error > threshold)[0]
    return int(above[0]) if len(above) else None


def plot_rollout(truth, pred, rho, error, step_exceeded, path):
    t = np.arange(len(truth)) * DT
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])

    axes = []
    for k, name in enumerate("xyz"):
        ax = fig.add_subplot(grid[k, 0], sharex=axes[0] if axes else None)
        ax.plot(t, truth[:, k], color="tab:blue", label="odeint")
        ax.plot(t, pred[:, k], color="tab:orange", linestyle="--", label="network rollout")
        ax.set_ylabel(name)
        if step_exceeded is not None:
            ax.axvline(t[step_exceeded], color="gray", linestyle=":")
        axes.append(ax)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time")

    ax3d = fig.add_subplot(grid[:2, 1], projection="3d")
    ax3d.plot(*truth.T, color="tab:blue", linewidth=0.8, label="odeint")
    ax3d.plot(*pred.T, color="tab:orange", linewidth=0.8, label="network rollout")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(loc="upper left")

    ax_err = fig.add_subplot(grid[2, 1])
    ax_err.semilogy(t[1:], error[1:], color="tab:red")
    ax_err.axhline(ERROR_THRESHOLD, color="gray", linestyle=":")
    ax_err.set_xlabel("time")
    ax_err.set_ylabel("Euclidean error")

    fig.suptitle(f"Lorenz system at rho = {rho}: network rollout versus odeint")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    start = time.time()
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_X, train_Y, val_X, val_Y, val_rho = build_dataset()
    print(f"training pairs: {len(train_X)}, validation pairs: {len(val_X)}")

    model = build_model(train_X)
    history = model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(val_X, val_Y), shuffle=True, verbose=2)
    print(f"final training MSE: {history.history['loss'][-1]:.3e}")
    print(f"final validation one-step MSE: {history.history['val_loss'][-1]:.3e}")

    val_pred = model.predict(val_X, batch_size=1024, verbose=2)
    for rho in TRAIN_RHOS:
        mask = val_rho == rho
        mse = np.mean((val_pred[mask] - val_Y[mask]) ** 2)
        print(f"validation one-step MSE at rho = {rho}: {mse:.3e}")

    model.save(MODEL_PATH)

    horizon = (N_ROLLOUT_STEPS - 1) * DT
    seeds = [ROLLOUT_SEED_BASE + k for k in range(N_ROLLOUT_STARTS)]
    x0 = np.stack([random_start(s) for s in seeds])
    for rho in TEST_RHOS:
        truth = np.stack([integrate(x, rho, N_ROLLOUT_STEPS) for x in x0])
        pred = rollout(model, x0, rho, N_ROLLOUT_STEPS)
        error = np.linalg.norm(pred - truth, axis=2)

        steps = [first_exceedance(e) for e in error]
        times = [None if s is None else s * DT for s in steps]
        if times[0] is None:
            outcome = f"error stays within {ERROR_THRESHOLD} for all {horizon:.2f} time units"
        else:
            outcome = (f"error first exceeds {ERROR_THRESHOLD} at time {times[0]:.2f} "
                       f"of {horizon:.2f}")
        print(f"rho = {rho}, plotted start (seed {seeds[0]}): {outcome}; "
              f"maximum error {error[0].max():.3f}, final error {error[0, -1]:.3f}")

        exceeded = [tm for tm in times if tm is not None]
        stayed = N_ROLLOUT_STARTS - len(exceeded)
        print(f"rho = {rho}, {N_ROLLOUT_STARTS} fresh starts: {stayed} stay within "
              f"{ERROR_THRESHOLD} for all {horizon:.2f} time units; "
              f"maximum error over all starts {error.max():.3f}")
        if exceeded:
            print(f"rho = {rho}, {len(exceeded)} starts first exceed {ERROR_THRESHOLD} at "
                  f"median time {np.median(exceeded):.2f} "
                  f"(minimum {min(exceeded):.2f}, maximum {max(exceeded):.2f})")

        plot_rollout(truth[0], pred[0], rho, error[0], steps[0],
                     os.path.join(OUT_DIR, f"rollout_rho{rho}.png"))

    print(f"total run time: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
