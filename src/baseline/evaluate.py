"""Evaluation script for Stage 1 predictions.

This script is provided to participants for transparency and local validation.
It evaluates submissions against a solution file that contains true ratings
and public/private split information. Note: the solution file itself is not
provided to participants, only this evaluation logic.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def validate_submission_format(df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    """Validate submission file format and content.

    Raises:
        ValueError: On any format or content error.
    """
    if df.empty:
        raise ValueError("Submission file is empty")

    required_cols = {"user_id", "book_id", "rating_predict"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Expected: {required_cols}")

    try:
        df["rating_predict"] = pd.to_numeric(df["rating_predict"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Column 'rating_predict' contains non-numeric values: {e}") from e

    if df["rating_predict"].isna().any():
        null_count = df["rating_predict"].isna().sum()
        raise ValueError(f"Column 'rating_predict' contains {null_count} null values")

    duplicates = df.duplicated(subset=["user_id", "book_id"])
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = df[duplicates][["user_id", "book_id"]].to_dict("records")[:5]
        raise ValueError(f"Found {dup_count} duplicate (user_id, book_id) pairs. " f"Examples: {dup_examples}")

    if df.shape[0] != solution_df.shape[0]:
        raise ValueError(f"Row count mismatch: {df.shape[0]} in submission, " f"{solution_df.shape[0]} expected")

    solution_pairs = set(zip(solution_df["user_id"], solution_df["book_id"], strict=False))
    submission_pairs = set(zip(df["user_id"], df["book_id"], strict=False))

    missing_pairs = solution_pairs - submission_pairs
    if missing_pairs:
        examples = list(missing_pairs)[:5]
        raise ValueError(f"Missing {len(missing_pairs)} required pairs from solution. Examples: {examples}")

    extra_pairs = submission_pairs - solution_pairs
    if extra_pairs:
        examples = list(extra_pairs)[:5]
        raise ValueError(f"Found {len(extra_pairs)} extra pairs not in solution. Examples: {examples}")


def _validate_solution_columns(df: pd.DataFrame) -> None:
    """Validate solution file has required columns."""
    required_cols = {"user_id", "book_id", "rating", "stage"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Expected: {required_cols}")


def _validate_solution_rating(df: pd.DataFrame) -> None:
    """Validate solution rating column."""
    duplicates = df.duplicated(subset=["user_id", "book_id"])
    if duplicates.any():
        raise ValueError(f"Found {duplicates.sum()} duplicate pairs in solution")

    try:
        df["rating"] = pd.to_numeric(df["rating"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Column 'rating' contains non-numeric values: {e}") from e

    if df["rating"].isna().any():
        raise ValueError(f"Column 'rating' contains {df['rating'].isna().sum()} null values")

    invalid_ratings = df[(df["rating"] < 0) | (df["rating"] > 10)]
    if not invalid_ratings.empty:
        examples = invalid_ratings[["user_id", "book_id", "rating"]].to_dict("records")[:5]
        raise ValueError(f"Rating values out of range [0, 10]. Examples: {examples}")


def _validate_solution_stage(df: pd.DataFrame) -> None:
    """Validate solution stage column."""
    if df["stage"].isna().any():
        null_count = df["stage"].isna().sum()
        raise ValueError(f"Column 'stage' contains {null_count} null values. " f"Must be 'public' or 'private'")

    valid_stages = {"public", "private"}
    invalid = set(df["stage"].unique()) - valid_stages
    if invalid:
        raise ValueError(f"Invalid stage values: {invalid}. Allowed: {valid_stages}")

    if "public" not in df["stage"].to_numpy():
        raise ValueError("No records with stage='public' found in solution")
    if "private" not in df["stage"].to_numpy():
        raise ValueError("No records with stage='private' found in solution")


def validate_solution_format(df: pd.DataFrame) -> None:
    """Validate solution file format and content.

    Raises:
        ValueError: On any format or content error.
    """
    if df.empty:
        raise ValueError("Solution file is empty")

    _validate_solution_columns(df)
    _validate_solution_rating(df)
    _validate_solution_stage(df)


def calculate_stage1_metrics(merged_df: pd.DataFrame) -> dict[str, float]:
    """Calculate RMSE, MAE, and Score metrics.

    Predictions are clipped to [0, 10] range.
    """
    if merged_df.empty:
        return {"Score": 0.0, "RMSE": 0.0, "MAE": 0.0}

    y_true = merged_df["rating"]
    y_pred = merged_df["rating_predict"].clip(0, 10)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    range_width = 10.0
    mae_norm = mae / range_width
    rmse_norm = rmse / range_width
    score = 1 - (0.5 * rmse_norm + 0.5 * mae_norm)

    return {"Score": score, "RMSE": rmse, "MAE": mae}


def main() -> dict[str, float]:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 predictions")
    parser.add_argument(
        "--submission",
        type=str,
        default="submission.csv",
        help="Path to submission file (default: submission.csv)",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="solution.csv",
        help="Path to solution file (default: solution.csv)",
    )
    args = parser.parse_args()

    try:
        submission = pd.read_csv(args.submission)
        solution = pd.read_csv(args.solution)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_solution_format(solution)
    except ValueError as e:
        print(f"Solution validation error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_submission_format(submission, solution)
    except ValueError as e:
        print(f"Submission validation error: {e}", file=sys.stderr)
        sys.exit(1)

    solution_public = solution[solution["stage"] == "public"].copy()
    solution_private = solution[solution["stage"] == "private"].copy()

    public_merged = submission.merge(solution_public, on=["user_id", "book_id"], how="inner")
    private_merged = submission.merge(solution_private, on=["user_id", "book_id"], how="inner")

    # Additional safety check after merge
    try:
        if public_merged.shape[0] != solution_public.shape[0]:
            missing = solution_public.shape[0] - public_merged.shape[0]
            missing_pairs = solution_public[
                ~solution_public.set_index(["user_id", "book_id"]).index.isin(
                    public_merged.set_index(["user_id", "book_id"]).index
                )
            ][["user_id", "book_id"]].to_dict("records")
            raise ValueError(f"Missing {missing} required pairs in public part. " f"Examples: {missing_pairs[:5]}")

        if private_merged.shape[0] != solution_private.shape[0]:
            missing = solution_private.shape[0] - private_merged.shape[0]
            missing_pairs = solution_private[
                ~solution_private.set_index(["user_id", "book_id"]).index.isin(
                    private_merged.set_index(["user_id", "book_id"]).index
                )
            ][["user_id", "book_id"]].to_dict("records")
            raise ValueError(f"Missing {missing} required pairs in private part. " f"Examples: {missing_pairs[:5]}")
    except ValueError as e:
        print(f"Data merge error: {e}", file=sys.stderr)
        sys.exit(1)

    public_metrics = calculate_stage1_metrics(public_merged)
    private_metrics = calculate_stage1_metrics(private_merged)

    print("--- Public ---")
    for metric, value in public_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\n--- Private ---")
    for metric, value in private_metrics.items():
        print(f"{metric}: {value:.6f}")

    return {
        "public_score": public_metrics["Score"],
        "private_score": private_metrics["Score"],
    }


if __name__ == "__main__":
    main()
