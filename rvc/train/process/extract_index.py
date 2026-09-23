"""Stage 4: build the voice's retrieval index from its training features.

At conversion time, each input frame's content features can be blended with their nearest
neighbours from the training set, which pulls the output's articulation toward the target voice.

Writes <experiment dir>/<name>.index. Run with: python -m rvc.train.process.extract_index --experiment-dir DIR
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

FEATURE_DIM = 768
MAX_POINTS = 200_000  # Above this many frames, the features are reduced to KMEANS_CLUSTERS centroids.
KMEANS_CLUSTERS = 10_000
ADD_BATCH_SIZE = 8192

type IndexAlgorithm = Literal["auto", "kmeans"]


def build_index(experiment_dir: Path, algorithm: IndexAlgorithm = "auto") -> Path:
    feature_dir = experiment_dir / "extracted"
    index_path = experiment_dir / f"{experiment_dir.name}.index"
    if index_path.exists():
        print(f"{index_path} already exists.")
        return index_path
    feature_files = sorted(feature_dir.glob("*.npy")) if feature_dir.is_dir() else []
    if not feature_files:
        raise FileNotFoundError(f"No features found in {feature_dir}. Run the extract stage first.")

    print(f"Building the index for {experiment_dir.name}...")
    features = np.concatenate([np.load(path) for path in feature_files], axis=0)
    np.random.shuffle(features)

    if features.shape[0] > MAX_POINTS or algorithm == "kmeans":
        kmeans = MiniBatchKMeans(
            n_clusters=KMEANS_CLUSTERS,
            verbose=True,
            batch_size=256 * (os.cpu_count() or 1),
            compute_labels=False,
            init="random",
        )
        features = kmeans.fit(features).cluster_centers_

    n_ivf = min(int(16 * np.sqrt(features.shape[0])), features.shape[0] // 39)
    index = faiss.index_factory(FEATURE_DIM, f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(features)
    for start in range(0, features.shape[0], ADD_BATCH_SIZE):
        index.add(features[start : start + ADD_BATCH_SIZE])

    faiss.write_index(index, str(index_path))
    print(f"Saved {index_path}")
    return index_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=["auto", "kmeans"],
        default="auto",
        help="kmeans always clusters the features first. auto does only for large datasets.",
    )
    args = parser.parse_args(argv)
    build_index(args.experiment_dir, args.algorithm)


if __name__ == "__main__":
    main()
