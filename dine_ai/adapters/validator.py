import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

# Ensure the root workspace directory is in the Python path when running as a direct script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from dine_ai.adapters.schema_mapper import CANONICAL_SCHEMA

MANDATORY_COLUMNS = {"recipe_name", "ingredients"}
RECOMMENDED_COLUMNS = {
    "price", "meal_type", "diet_labels",
    "protein_g_per_serving", "calories_per_serving"
}
OPTIONAL_COLUMNS = {
    "rating", "description", "image_url", "cuisine_type",
    "preparation_time_minutes", "availability", "popularity", "item_type"
}


class ValidationStatus(Enum):
    """
    Standardized validation status codes to prevent magic strings.
    """
    PASS = "PASS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    ERROR = "ERROR"


@dataclass
class ValidationMessage:
    """
    Structured feedback message generated during dataset validation.
    """
    validator_name: str
    severity: ValidationStatus
    column_name: Optional[str]
    row_index: Any  # Can be an int, List[int], or None
    invalid_value: Any
    message: str
    code: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """
    Data container that aggregates diagnostic output from the validation pipeline.
    """
    is_valid: bool = True
    errors: List[ValidationMessage] = field(default_factory=list)
    warnings: List[ValidationMessage] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    validator_results: Dict[str, ValidationStatus] = field(default_factory=dict)


class BaseValidator(ABC):
    """
    Abstract Base Class for all pluggable DataFrame validators.
    """

    @abstractmethod
    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """
        Executes validation logic against a standardized pandas DataFrame.
        Appends any violations to the result errors or warnings list, and
        records the test outcome in result.validator_results.

        Parameters
        ----------
        df : pandas.DataFrame
            The input standardized DataFrame to validate.
        result : ValidationResult
            The mutable collector for tracking overall validation outputs.
        """
        pass


class RequiredColumnsValidator(BaseValidator):
    """
    Checks if all mandatory canonical schema fields are present in the DataFrame columns.
    """

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        missing_fields = []
        for field in MANDATORY_COLUMNS:
            if field not in df.columns:
                missing_fields.append(field)

        if missing_fields:
            result.is_valid = False
            for field in missing_fields:
                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.FAILED,
                    column_name=field,
                    row_index=None,
                    invalid_value=None,
                    message=f"Missing required canonical column: '{field}'",
                    code="MISSING_REQUIRED_COLUMN"
                )
                result.errors.append(msg)
            result.validator_results[name] = ValidationStatus.FAILED
        else:
            result.validator_results[name] = ValidationStatus.PASS


class RecommendedColumnsValidator(BaseValidator):
    """
    Checks if recommended fields are present, issuing warnings (not errors) if they are missing.
    """

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        missing_fields = []
        for field in RECOMMENDED_COLUMNS:
            if field not in df.columns:
                missing_fields.append(field)

        if missing_fields:
            for field in missing_fields:
                # Custom descriptive warning matching capabilities
                cap_disabled_str = ""
                if field == "price":
                    cap_disabled_str = "Price information is unavailable — Budget Filtering disabled."
                elif field == "meal_type":
                    cap_disabled_str = "Meal type information is unavailable — Meal Type Filtering disabled."
                elif field == "diet_labels":
                    cap_disabled_str = "Dietary labels are unavailable — Keto/Vegan/Dietary Recommendations disabled."
                elif field == "protein_g_per_serving":
                    cap_disabled_str = "Protein information is unavailable — High Protein Recommendations disabled."
                elif field == "calories_per_serving":
                    cap_disabled_str = "Calorie information is unavailable — Nutrition Search / Low Calorie Recommendations disabled."
                else:
                    cap_disabled_str = f"Recommended column '{field}' is missing."

                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.WARNING,
                    column_name=field,
                    row_index=None,
                    invalid_value=None,
                    message=cap_disabled_str,
                    code="MISSING_RECOMMENDED_COLUMN"
                )
                result.warnings.append(msg)
            result.validator_results[name] = ValidationStatus.WARNING
        else:
            result.validator_results[name] = ValidationStatus.PASS


class DuplicateColumnsValidator(BaseValidator):
    """
    Checks if there are duplicate column names present in the DataFrame.
    """

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        duplicated_flags = df.columns.duplicated()
        if any(duplicated_flags):
            duplicates = set(df.columns[duplicated_flags])
            result.is_valid = False
            for col in duplicates:
                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.FAILED,
                    column_name=col,
                    row_index=None,
                    invalid_value=None,
                    message=f"Duplicate column detected: '{col}'",
                    code="DUPLICATE_COLUMN"
                )
                result.errors.append(msg)
            result.validator_results[name] = ValidationStatus.FAILED
        else:
            result.validator_results[name] = ValidationStatus.PASS


class EmptyDatasetValidator(BaseValidator):
    """
    Checks if the input DataFrame is empty (has 0 rows).
    """

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        if df.empty or len(df) == 0:
            result.is_valid = False
            msg = ValidationMessage(
                validator_name=name,
                severity=ValidationStatus.FAILED,
                column_name=None,
                row_index=None,
                invalid_value=None,
                message="Dataset has zero rows (empty dataset).",
                code="EMPTY_DATASET"
            )
            result.errors.append(msg)
            result.validator_results[name] = ValidationStatus.FAILED
        else:
            result.validator_results[name] = ValidationStatus.PASS


class MissingValuesValidator(BaseValidator):
    """
    Detects missing values (null/NaN/None) per column in the DataFrame.
    Calculates missing percentages, populates results in result.statistics,
    and issues warnings if the percentage of missing values exceeds a given threshold.
    """

    def __init__(self, warning_threshold: float = 50.0):
        self.warning_threshold = warning_threshold

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        total_rows = len(df)
        if total_rows == 0:
            result.validator_results[name] = ValidationStatus.PASS
            return

        missing_percentages = {}
        has_warnings = False

        # Use index-based loop to safely handle duplicate column names
        for i in range(len(df.columns)):
            col = df.columns[i]
            series = df.iloc[:, i]
            missing_count = series.isna().sum()
            pct = (missing_count / total_rows) * 100
            
            # Deduplicate dictionary keys if duplicate columns exist
            key = col
            suffix = 1
            while key in missing_percentages:
                key = f"{col}_{suffix}"
                suffix += 1
                
            missing_percentages[key] = round(pct, 2)

            if pct > self.warning_threshold:
                has_warnings = True
                nan_indices = df.index[series.isna()].tolist()
                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.WARNING,
                    column_name=col,
                    row_index=nan_indices,
                    invalid_value=None,
                    message=(
                        f"Column '{col}' (position {i}) has {pct:.2f}% missing values, "
                        f"which exceeds the threshold of {self.warning_threshold}%."
                    ),
                    code="HIGH_MISSING_VALUES_RATE"
                )
                result.warnings.append(msg)

        result.statistics["missing_values_percentage"] = missing_percentages

        if has_warnings:
            result.validator_results[name] = ValidationStatus.WARNING
        else:
            result.validator_results[name] = ValidationStatus.PASS


class DataTypeValidator(BaseValidator):
    """
    Compares the pandas DataFrame column data types against the expected dtypes
    defined in the CANONICAL_SCHEMA.
    """

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        has_errors = False

        # Use index-based loop to safely handle duplicate column names
        for i in range(len(df.columns)):
            col = df.columns[i]
            if col not in CANONICAL_SCHEMA:
                continue

            expected_type = CANONICAL_SCHEMA[col].get("dtype")
            series = df.iloc[:, i]

            # Skip checking type if all values are null/empty
            if series.isna().all():
                continue

            is_valid_type = True
            if expected_type == str:
                # String columns in pandas are usually of type 'object' or 'string'
                if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                    is_valid_type = False
            elif expected_type == float:
                if not pd.api.types.is_numeric_dtype(series):
                    is_valid_type = False
            elif expected_type == object:
                if not pd.api.types.is_object_dtype(series):
                    is_valid_type = False

            if not is_valid_type:
                has_errors = True
                result.is_valid = False
                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.FAILED,
                    column_name=col,
                    row_index=None,
                    invalid_value=series.dtype.name,
                    message=(
                        f"Data type mismatch for column '{col}' (position {i}): expected subclass/subtype "
                        f"of {expected_type.__name__}, got {series.dtype}."
                    ),
                    code="DATATYPE_MISMATCH"
                )
                result.errors.append(msg)

        if has_errors:
            result.validator_results[name] = ValidationStatus.FAILED
        else:
            result.validator_results[name] = ValidationStatus.PASS


# ==========================================================
# Business Rules Engine Interfaces & Concrete Rules
# ==========================================================

class BusinessRule(ABC):
    """
    Abstract Base Class for defining reusable, pluggable business rules.
    """

    @abstractmethod
    def validate_series(self, series: pd.Series) -> pd.Series:
        """
        Validates a pandas Series.
        Returns a boolean Series of the same length, where True indicates a rule violation.
        """
        pass

    @abstractmethod
    def get_message(self, column_name: str, violations_count: int) -> str:
        """
        Returns a descriptive error message for rule failures.
        """
        pass

    @property
    @abstractmethod
    def code(self) -> str:
        """
        Returns the unique business rule warning/error code.
        """
        pass


class PositiveValueRule(BusinessRule):
    """Rule verifying that numeric values are greater than or equal to zero."""

    def validate_series(self, series: pd.Series) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Return True for any value < 0
        return numeric_series < 0

    def get_message(self, column_name: str, violations_count: int) -> str:
        return f"Business rule violation: '{column_name}' cannot be negative. Found {violations_count} violations."

    @property
    def code(self) -> str:
        return "NEGATIVE_VALUE"


class GreaterThanZeroRule(BusinessRule):
    """Rule verifying that numeric values are strictly greater than zero."""

    def validate_series(self, series: pd.Series) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Return True for any value <= 0
        return numeric_series <= 0

    def get_message(self, column_name: str, violations_count: int) -> str:
        return f"Business rule violation: '{column_name}' must be greater than zero. Found {violations_count} violations."

    @property
    def code(self) -> str:
        return "ZERO_OR_NEGATIVE_VALUE"


class RangeRule(BusinessRule):
    """Rule verifying that numeric values fall within a specified range [min_val, max_val]."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def validate_series(self, series: pd.Series) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Return True if outside the specified bounds
        return (numeric_series < self.min_val) | (numeric_series > self.max_val)

    def get_message(self, column_name: str, violations_count: int) -> str:
        return (
            f"Business rule violation: '{column_name}' must be within range "
            f"[{self.min_val}, {self.max_val}]. Found {violations_count} violations."
        )

    @property
    def code(self) -> str:
        return "OUT_OF_RANGE"


class NonEmptyStringRule(BusinessRule):
    """Rule verifying that string values are not empty or purely whitespace."""

    def validate_series(self, series: pd.Series) -> pd.Series:
        # Cast to string and check if empty
        string_series = series.astype(str).str.strip()
        return series.isna() | (string_series == "") | (string_series == "nan")

    def get_message(self, column_name: str, violations_count: int) -> str:
        return f"Business rule violation: '{column_name}' cannot be empty. Found {violations_count} violations."

    @property
    def code(self) -> str:
        return "EMPTY_STRING"


# ==========================================================
# Business Rules Registry
# ==========================================================

BUSINESS_RULES: Dict[str, List[BusinessRule]] = {
    "price": [
        PositiveValueRule()
    ],
    "servings": [
        GreaterThanZeroRule()
    ],
    "calories_per_serving": [
        PositiveValueRule()
    ],
    "protein_g_per_serving": [
        PositiveValueRule()
    ]
}


class BusinessRulesValidator(BaseValidator):
    """
    Enforces business logical integrity constraints defined in BUSINESS_RULES registry.
    """

    def __init__(self, rules_registry: Optional[Dict[str, List[BusinessRule]]] = None):
        self.rules_registry = rules_registry if rules_registry is not None else BUSINESS_RULES

    def validate(self, df: pd.DataFrame, result: ValidationResult) -> None:
        name = self.__class__.__name__
        has_errors = False

        # Loop through columns by index to support duplicate column name scenarios safely
        for i in range(len(df.columns)):
            col = df.columns[i]
            if col not in self.rules_registry:
                continue

            series = df.iloc[:, i]
            rules = self.rules_registry[col]

            for rule in rules:
                violations_mask = rule.validate_series(series)
                # Find the actual indices where the rule failed
                violating_indices = df.index[violations_mask].tolist()

                if violating_indices:
                    has_errors = True
                    result.is_valid = False

                    # Extract sample invalid value for reporting
                    sample_invalid_val = series.loc[violating_indices[0]]
                    msg_text = rule.get_message(col, len(violating_indices))

                    message = ValidationMessage(
                        validator_name=name,
                        severity=ValidationStatus.FAILED,
                        column_name=col,
                        row_index=violating_indices,
                        invalid_value=sample_invalid_val,
                        message=msg_text,
                        code=rule.code
                    )
                    result.errors.append(message)

        if has_errors:
            result.validator_results[name] = ValidationStatus.FAILED
        else:
            result.validator_results[name] = ValidationStatus.PASS


class DataFrameValidator:
    """
    Main coordinator for the DineAI validation pipeline.
    Statelessly executes a series of pluggable validators against a pandas DataFrame.
    """

    def __init__(self, validators: Optional[List[BaseValidator]] = None):
        """
        Initializes the DataFrameValidator.

        Parameters
        ----------
        validators : Optional[List[BaseValidator]]
            A custom list of validator rules. If None, default pipeline containing
            all standard validators will be configured.
        """
        if validators is not None:
            self.validators = validators
        else:
            self.validators = [
                RequiredColumnsValidator(),
                RecommendedColumnsValidator(),
                DuplicateColumnsValidator(),
                EmptyDatasetValidator(),
                MissingValuesValidator(),
                DataTypeValidator(),
                BusinessRulesValidator()
            ]

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Statelessly executes all configured validators on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to validate.

        Returns
        -------
        ValidationResult
            The accumulated validation results.
        """
        result = ValidationResult()
        
        # Capture basic statistics before running validators
        result.statistics["total_rows"] = len(df)
        result.statistics["total_columns"] = len(df.columns)

        for validator in self.validators:
            try:
                validator.validate(df, result)
            except Exception as e:
                # Capture unexpected exceptions during a validator execution
                name = validator.__name__ if hasattr(validator, "__name__") else validator.__class__.__name__
                result.is_valid = False
                msg = ValidationMessage(
                    validator_name=name,
                    severity=ValidationStatus.ERROR,
                    column_name=None,
                    row_index=None,
                    invalid_value=None,
                    message=f"Unexpected error in validator '{name}': {str(e)}",
                    code="UNEXPECTED_ERROR"
                )
                result.errors.append(msg)
                result.validator_results[name] = ValidationStatus.ERROR

        # If any validation errors occurred, the dataset is invalid
        if result.errors:
            result.is_valid = False

        return result

    @staticmethod
    def generate_report(result: ValidationResult) -> None:
        """
        Prints a detailed, enterprise ETL-style validation report to console.
        """
        print(_build_validation_report(result))


def _build_validation_report(result: ValidationResult) -> str:
    """
    Constructs a detailed terminal-friendly validation report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'DINEAI DATA VALIDATION REPORT':^80}")
    lines.append("=" * 80)
    lines.append("VALIDATION PIPELINE METRICS:")
    lines.append("-" * 80)
    
    for val_name, status in result.validator_results.items():
        lines.append(f"  {val_name:<35} | {status.value:<10}")
        
    lines.append("-" * 80)
    overall_status = "PASS" if result.is_valid else "FAILED"
    lines.append(f"  OVERALL VALIDATION STATUS          : {overall_status:<10}")
    lines.append("=" * 80)

    # Errors section
    if result.errors:
        lines.append("ERRORS:")
        lines.append("-" * 80)
        for error in result.errors:
            lines.append(f"  [ERROR] {error.message}")
        lines.append("=" * 80)

    # Warnings section
    if result.warnings:
        lines.append("WARNINGS:")
        lines.append("-" * 80)
        for warning in result.warnings:
            lines.append(f"  [WARN] {warning.message}")
        lines.append("=" * 80)

    # Statistics section
    if result.statistics:
        lines.append("STATISTICS:")
        lines.append("-" * 80)
        lines.append(f"  - Total Rows    : {result.statistics.get('total_rows', 0)}")
        lines.append(f"  - Total Columns : {result.statistics.get('total_columns', 0)}")
        
        missing_stats = result.statistics.get("missing_values_percentage", {})
        if missing_stats:
            lines.append("  - Missing values percentage per column:")
            for col, pct in missing_stats.items():
                lines.append(f"      * {col:<30}: {pct:.2f}%")
        lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Standard testing demonstration for DataFrameValidator pipeline
    print("Executing self-test for DataFrameValidator...\n")
    
    # Create a test DataFrame that violates several checks:
    # 1. Missing required column 'recipe_name' (will fail RequiredColumnsValidator)
    # 2. Duplicate column 'price' (will fail DuplicateColumnsValidator)
    # 3. Missing values rate high on 'description' (will warn MissingValuesValidator)
    # 4. Data type mismatch for 'servings' (float expected, got string - DataTypeValidator)
    # 5. Negative values for price and calories (BusinessRulesValidator)
    
    df_invalid = pd.DataFrame([
        # Row 0: mostly valid except datatype or duplicate column later
        {
            "calories_per_serving": 250.0,
            "protein_g_per_serving": 15.0,
            "servings": "Five",  # Mismatch (DataTypeValidator)
            "price": 10.0,
            "description": None,
        },
        # Row 1: violates business rules and missing values
        {
            "calories_per_serving": -50.0,   # Negative (BusinessRulesValidator)
            "protein_g_per_serving": -2.0,   # Negative (BusinessRulesValidator)
            "servings": "Zero",              # Mismatch (DataTypeValidator)
            "price": -15.0,                  # Negative (BusinessRulesValidator)
            "description": None,
        }
    ])
    
    # Force duplicate columns programmatically
    df_invalid.columns = ["calories_per_serving", "protein_g_per_serving", "servings", "price", "price"]
    
    # 1. Execute Validation on Invalid DataFrame
    validator = DataFrameValidator()
    print("--- 1. Testing Invalid Dataset ---")
    invalid_result = validator.validate(df_invalid)
    DataFrameValidator.generate_report(invalid_result)
    
    # 2. Execute Validation on Valid DataFrame
    print("\n--- 2. Testing Valid Dataset ---")
    df_valid = pd.DataFrame([
        {
            "recipe_name": "Truffle Pasta",
            "calories_per_serving": 450.0,
            "protein_g_per_serving": 12.0,
            "servings": 2.0,
            "price": 22.0,
            "description": "Delicious truffle pasta.",
            "ingredients": ["pasta", "truffles"],
        },
        {
            "recipe_name": "Avocado Salad",
            "calories_per_serving": 200.0,
            "protein_g_per_serving": 4.0,
            "servings": 1.0,
            "price": 12.50,
            "description": None,  # 50% missing (warns if threshold exceeded, here it equals 50%)
            "ingredients": ["avocado", "lettuce"],
        }
    ])
    
    valid_result = validator.validate(df_valid)
    DataFrameValidator.generate_report(valid_result)
    
    # 3. Execute Validation on Empty DataFrame
    print("\n--- 3. Testing Empty Dataset ---")
    df_empty = pd.DataFrame()
    empty_result = validator.validate(df_empty)
    DataFrameValidator.generate_report(empty_result)
