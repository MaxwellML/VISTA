#generate_othonna.py
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 42

N_PLANTS = 30

# Experimental AOI in Unreal-world metres.
X_MIN_M = -480.0
X_MAX_M = 480.0
Y_MIN_M = -480.0
Y_MAX_M = 480.0

# Controls the characteristic size of the clusters.
PATCH_SIZE_M = 200.0

# Higher = stronger preference for high-noise areas.
CLUSTER_STRENGTH = 2.5

# Stops two shrubs spawning almost on top of each other.
MIN_SEPARATION_M = 6.0

N_CANDIDATES = 50_000

OUTPUT_CSV = Path("othonna_locations.csv")
OUTPUT_PLOT = Path("othonna_perlin_preview.png")


GRADIENTS = np.array([
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
]) #possible gradient directions.


def fade(t):
    """Improved Perlin fade function."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + t * (b - a)


def create_permutation(rng):
    permutation = np.arange(256)
    rng.shuffle(permutation)

    return np.concatenate([
        permutation,
        permutation,
    ])


def gradient(hash_value, x, y):
    g = GRADIENTS[hash_value & 7]

    return (
        g[..., 0] * x
        + g[..., 1] * y
    )


def perlin_2d(x, y, permutation):
    x_floor = np.floor(x)
    y_floor = np.floor(y)

    xi = x_floor.astype(int) & 255
    yi = y_floor.astype(int) & 255

    xf = x - x_floor
    yf = y - y_floor

    u = fade(xf)
    v = fade(yf)

    aa = permutation[permutation[xi] + yi]
    ab = permutation[permutation[xi] + yi + 1]

    ba = permutation[permutation[xi + 1] + yi]
    bb = permutation[permutation[xi + 1] + yi + 1]

    bottom = lerp(
        gradient(aa, xf, yf),
        gradient(ba, xf - 1, yf),
        u,
    )

    top = lerp(
        gradient(ab, xf, yf - 1),
        gradient(bb, xf - 1, yf - 1),
        u,
    )

    return lerp(bottom, top, v)


def generate_locations():
    rng = np.random.Generator(
        np.random.PCG64(SEED)
    )

    permutation = create_permutation(rng)

    # First generate many possible plant positions uniformly.
    candidate_x = rng.uniform(
        X_MIN_M,
        X_MAX_M,
        N_CANDIDATES,
    )

    candidate_y = rng.uniform(
        Y_MIN_M,
        Y_MAX_M,
        N_CANDIDATES,
    )

    # Evaluate the underlying smooth spatial field.
    noise = perlin_2d(
        candidate_x / PATCH_SIZE_M,
        candidate_y / PATCH_SIZE_M,
        permutation,
    )

    # Convert noise into sampling weights.
    #
    # High Perlin values become substantially more attractive,
    # producing clusters without making low regions impossible.
    weights = np.exp(
        CLUSTER_STRENGTH * noise
    )

    chosen = []

    for _ in range(N_PLANTS):
        probabilities = weights / weights.sum()

        index = rng.choice(
            N_CANDIDATES,
            p=probabilities,
        )

        chosen.append(index)

        # Remove all candidates too close to this plant.
        dx = candidate_x - candidate_x[index]
        dy = candidate_y - candidate_y[index]

        too_close = (
            dx * dx + dy * dy
            < MIN_SEPARATION_M ** 2
        )

        weights[too_close] = 0.0

    chosen = np.asarray(chosen)

    x = candidate_x[chosen]
    y = candidate_y[chosen]
    scores = noise[chosen]

    # Random orientation for each shrub.
    yaw = rng.uniform(
        0.0,
        360.0,
        N_PLANTS,
    )

    return x, y, scores, yaw, permutation


def save_csv(x, y, scores, yaw):
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "id",
            "x_m",
            "y_m",
            "ue_x_cm",
            "ue_y_cm",
            "yaw_deg",
            "scale",
            "perlin_score",
        ])

        for i in range(N_PLANTS):
            writer.writerow([
                f"Othonna_{i + 1:03d}",

                round(x[i], 3),
                round(y[i], 3),

                round(x[i] * 100, 1),
                round(y[i] * 100, 1),

                round(yaw[i], 2),

                1.0,

                round(scores[i], 6),
            ])


def save_preview(x, y, permutation):
    axis_x = np.linspace(
        X_MIN_M,
        X_MAX_M,
        350,
    )

    axis_y = np.linspace(
        Y_MIN_M,
        Y_MAX_M,
        350,
    )

    xx, yy = np.meshgrid(
        axis_x,
        axis_y,
    )

    field = perlin_2d(
        xx / PATCH_SIZE_M,
        yy / PATCH_SIZE_M,
        permutation,
    )

    plt.figure(figsize=(8, 7))

    plt.imshow(
        field,
        origin="lower",
        extent=[
            X_MIN_M,
            X_MAX_M,
            Y_MIN_M,
            Y_MAX_M,
        ],
    )

    plt.scatter(
        x,
        y,
    )

    plt.xlabel("Unreal X (m)")
    plt.ylabel("Unreal Y (m)")

    plt.title(
        f"Othonna cerarioides placement — seed {SEED}"
    )

    plt.colorbar(
        label="Perlin score"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=180,
    )

    plt.show()


def main():
    x, y, scores, yaw, permutation = (
        generate_locations()
    )

    save_csv(
        x,
        y,
        scores,
        yaw,
    )

    save_preview(
        x,
        y,
        permutation,
    )

    print(
        f"Generated {N_PLANTS} plants."
    )

    print(
        f"Saved coordinates to {OUTPUT_CSV}"
    )

    print(
        f"Saved preview to {OUTPUT_PLOT}"
    )


if __name__ == "__main__":
    main()