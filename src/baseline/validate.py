"""
Script to validate the format of the submission file.
"""

import pandas as pd

from . import config, constants


def validate() -> None:
    """Validates the structure and format of the submission file.

    Performs a series of checks to ensure the submission file is valid before
    uploading, such as verifying the number of rows, checking for missing
    values, and ensuring the user/book pairs match the test set.

    Raises:
        FileNotFoundError: If the test data or submission file does not exist.
        AssertionError: If any of the validation checks fail.
    """
    print("Validating submission file...")

    try:
        # Load test data and submission file
        # CSV files use comma as separator (default pandas behavior)
        test_df = pd.read_csv(config.RAW_DATA_DIR / constants.TEST_FILENAME)
        sub_df = pd.read_csv(config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME)

        # 1. Check length
        assert len(sub_df) == len(test_df), f"Submission length mismatch. Expected {len(test_df)}, got {len(sub_df)}."
        print("✅ Length check passed.")

        # 2. Check for missing values in prediction
        assert (
            not sub_df[constants.COL_PREDICTION].isna().any()
        ), f"Missing values found in '{constants.COL_PREDICTION}'."
        print("✅ No missing values check passed.")

        # 3. Check that the set of (user_id, book_id) pairs match
        test_keys = (
            test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]]
            .copy()
            .set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        )
        sub_keys = (
            sub_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]]
            .copy()
            .set_index([constants.COL_USER_ID, constants.COL_BOOK_ID])
        )

        assert test_keys.index.equals(
            sub_keys.index
        ), "The set of (user_id, book_id) pairs does not match the test set."
        print("✅ (user_id, book_id) pair matching check passed.")

        # 4. Check prediction range
        assert (
            sub_df[constants.COL_PREDICTION]
            .between(constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)
            .all()
        ), f"Predictions are not within the [{constants.PREDICTION_MIN_VALUE}, {constants.PREDICTION_MAX_VALUE}] range."
        print("✅ Prediction range [0, 10] check passed.")

        print("\nValidation successful! The submission file appears to be in the correct format.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the required files exist.")
    except AssertionError as e:
        print(f"Validation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    validate()
