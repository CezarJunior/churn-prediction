"""
Entry point for:
    python -m src.pipeline.train

Trains the XGBoost churn model, logs it to MLflow, and saves it to
models/best_model.joblib so the Gradio app can load it.

Options
-------
--trials N      Number of Optuna hyperparameter search trials (default: 20).
--data PATH     Path to a local CSV to use instead of downloading from the web.
--no-save       Skip saving the model to disk (useful for quick CI checks).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the churn prediction XGBoost model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        metavar="N",
        help="Number of Optuna hyperparameter search trials.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a local CSV file. If omitted, the dataset is downloaded automatically.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not persist the model to disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import joblib
    import pandas as pd

    from src.config import BEST_MODEL_PATH
    from src.pipeline.features import clean_data, load_data
    from src.pipeline.train import train_model

    # ------------------------------------------------------------------ data
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            logger.error("Data file not found: %s", data_path)
            return 1
        logger.info("Loading data from %s", data_path)
        raw = pd.read_csv(data_path)
        df = clean_data(raw)
    else:
        logger.info("No --data flag supplied — downloading dataset …")
        raw = load_data()
        df = clean_data(raw)

    # ----------------------------------------------------------------- train
    logger.info("Starting training with %d Optuna trials …", args.trials)
    pipeline, metrics = train_model(df=df, n_trials=args.trials)

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 55)
    print("  Training complete")
    print("=" * 55)
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  CV AUC    : {metrics.get('cv_auc', 'n/a')}")
    print("=" * 55)
    print(f"\n  Best hyperparameters:")
    for k, v in metrics.get("best_params", {}).items():
        print(f"    {k:<25}: {v}")
    print()

    # ------------------------------------------------------------------ save
    if not args.no_save:
        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, BEST_MODEL_PATH)
        logger.info("Model saved to %s", BEST_MODEL_PATH)
        print(f"  Model saved: {BEST_MODEL_PATH}")
    else:
        logger.info("--no-save flag set — model NOT written to disk")

    print("\nRun `python app.py` to start the Gradio demo.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
