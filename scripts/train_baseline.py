"""Train a single-symbol daily baseline from read-only PostgreSQL."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from stock_ml_dynamic.db import engine, price_history
from stock_ml_dynamic.training import TrainingConfig, train_classifier, save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval", default="1d", choices=["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"])
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-root", default="artifacts/training_runs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9.^_-]{1,20}", args.symbol):
        parser.error("Invalid ticker")
    if args.bars < 250:
        parser.error("--bars must be >= 250")
    with engine().connect() as conn:
        prices = price_history(conn, args.symbol.upper(), args.interval, args.bars)
    config = TrainingConfig(horizon=args.horizon, test_fraction=args.test_fraction)
    models, metadata = train_classifier(prices, config)
    metadata.update({"symbol": args.symbol.upper(), "interval": args.interval})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = save_run(models, metadata, Path(args.output_root) / f"{args.symbol.upper()}_{args.interval}_{timestamp}")
    print(f"Saved run: {folder}")
    for name, scores in metadata["metrics"].items():
        print(f"{name}: " + ", ".join(f"{key}={value:.4f}" for key, value in scores.items()))
    print(f"Train rows: {metadata['train_rows']} | Test rows: {metadata['test_rows']} | Purged bars: {args.horizon}")


if __name__ == "__main__":
    main()
