from __future__ import annotations

import polars as pl
from shared.utils import get_logger


logger = get_logger(__name__)


class MDLFileValidator:
    """Validator for MDL file types."""

    # Required columns for different file types
    MDL_INPUT_TESTING_COLUMNS = [
        'Document No',
        'Title',
        'Plan - First Date',
        'Plan - Final Date',
    ]

    MDL_HISTORICAL_COLUMNS = [
        'Document No',
        'Title',
        'Plan - First Date',
        'Plan - Final Date',
        'Actual - First Date',
        'Actual - Final Date',
    ]

    DATE_COLUMNS_INPUT = [
        'Plan - First Date',
        'Plan - Final Date',
    ]

    DATE_COLUMNS_HISTORICAL = [
        'Plan - First Date',
        'Plan - Final Date',
        'Actual - First Date',
        'Actual - Final Date',
    ]

    def validate_columns(self, df: pl.DataFrame, file_type: str) -> None:
        """Validate required columns are present."""
        if file_type == 'mdl_input_testing':
            required_columns = self.MDL_INPUT_TESTING_COLUMNS
        elif file_type == 'mdl_historical':
            required_columns = self.MDL_HISTORICAL_COLUMNS
        else:
            # L3 files - skip validation for now
            # TODO: validation for L3
            return

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f'Missing required columns for {file_type}: {missing_columns}',
            )

        logger.info(f'All required columns present for {file_type}')

    def validate_date_columns(
        self,
        df: pl.DataFrame,
        file_type: str,
    ) -> pl.DataFrame:
        """Validate and convert date columns."""
        if file_type == 'l3':
            return df  # Skip date validation for L3 files

        date_columns = (
            self.DATE_COLUMNS_INPUT if file_type == 'mdl_input_testing'
            else self.DATE_COLUMNS_HISTORICAL
        )

        validated_df = df.clone()

        for date_col in date_columns:
            if date_col not in df.columns:
                continue

            # Check for null values
            null_count = validated_df[date_col].null_count()
            if null_count > 0:
                raise ValueError(
                    f'Found {null_count} null values in date column '
                    f"'{date_col}'. All date columns must have valid values.",
                )

            # Get current dtype
            current_dtype = validated_df.schema[date_col]

            # If already Datetime, skip
            if current_dtype == pl.Datetime:
                logger.info(f"Column '{date_col}' is already Datetime type")
                continue

            # If Date, cast to Datetime
            if current_dtype == pl.Date:
                validated_df = validated_df.with_columns([
                    pl.col(date_col).cast(pl.Datetime).alias(date_col),
                ])
                logger.info(
                    f"Column '{date_col}' converted from Date to Datetime",
                )
                continue

            # If String, try parsing
            if current_dtype == pl.Utf8:
                try:
                    validated_df = validated_df.with_columns([
                        pl.col(date_col).str.strptime(
                            pl.Datetime,
                            fmt='%Y-%m-%d', strict=False,
                        ).alias(date_col),
                    ])
                    logger.info(
                        f"Successfully converted '{date_col}' to Datetime",
                    )
                except Exception:
                    # Identify problematic rows
                    problematic_rows = (
                        validated_df
                        .with_columns([
                            pl.col(date_col).str.strptime(
                                pl.Datetime, fmt='%Y-%m-%d', strict=False,
                            ).alias('parsed'),
                        ])
                        .filter(pl.col('parsed').is_null())
                        .select([date_col])
                    )
                    logger.error(
                        f"Invalid date values in '{date_col}': "
                        f'{problematic_rows}',
                    )
                    raise ValueError(
                        f"Cannot convert '{date_col}' to Datetime due "
                        f'to invalid format values',
                    )

            else:
                raise ValueError(
                    f"Column '{date_col}' has unsupported"
                    f' type: {current_dtype}',
                )

        return validated_df
