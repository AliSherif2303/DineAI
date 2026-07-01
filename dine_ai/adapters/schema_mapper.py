import re
from typing import Dict, List, Optional

class ColumnNormalizer:
    """
    Normalize raw column names into a consistent format.

    Example:
    --------
    Recipe Name -> recipe_name
    Protein (g) -> protein_g
    Meal-Type -> meal_type
    """

    @staticmethod
    def normalize(column_name: str) -> str:

        column_name = column_name.strip().lower()

        # Replace spaces and hyphens
        column_name = re.sub(r"[\s\-]+", "_", column_name)

        # Remove brackets
        column_name = re.sub(r"[()]", "", column_name)

        # Remove duplicate underscores
        column_name = re.sub(r"_+", "_", column_name)

        return column_name

""" 
print(ColumnNormalizer.normalize("Recipe Name"))
print(ColumnNormalizer.normalize("Protein (g)"))
print(ColumnNormalizer.normalize("Meal-Type")) """


# ==========================================================
# DineAI Internal Canonical Schema
# ==========================================================

CANONICAL_SCHEMA = {

    # ---------- Basic Information ----------

    "recipe_name": {
        "required": True,
        "dtype": str,
        "description": "Recipe or menu item name"
    },

    "item_type": {
        "required": False,
        "dtype": str,
        "description": "Menu item category"
    },

    "description": {
        "required": False,
        "dtype": str,
        "description": "Recipe description"
    },

    "meal_type": {
        "required": False,
        "dtype": str,
        "description": "Breakfast, Lunch, Dinner..."
    },

    "dish_type": {
        "required": False,
        "dtype": str,
        "description": "Main course, Soup, Salad..."
    },

    "cuisine_type": {
        "required": False,
        "dtype": str,
        "description": "Italian, American..."
    },

    "ingredients": {
        "required": False,
        "dtype": object,
        "description": "List of ingredients"
    },

    "servings": {
        "required": False,
        "dtype": float,
        "description": "Number of servings"
    },

    # ---------- Nutrition Per Serving ----------

    "calories_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "kcal"
    },

    "protein_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "carbs_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "net_carbs_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "fat_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "saturated_fat_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "trans_fat_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "monounsaturated_fat_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "polyunsaturated_fat_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "fiber_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "sugar_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "added_sugar_g_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "g"
    },

    "sodium_mg_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "mg"
    },

    "cholesterol_mg_per_serving": {
        "required": False,
        "dtype": float,
        "unit": "mg"
    },

    # ---------- Dietary Information ----------

    "diet_labels": {
        "required": False,
        "dtype": object,
        "description": "Balanced, Low-Carb, High-Protein..."
    },

    "health_labels": {
        "required": False,
        "dtype": object,
        "description": "Gluten-Free, Vegan..."
    },

    "cautions": {
        "required": False,
        "dtype": object,
        "description": "Allergens or warnings"
    },

    # ---------- Restaurant Information ----------

    "price": {
        "required": False,
        "dtype": float,
        "unit": "currency"
    },

    "currency": {
        "required": False,
        "dtype": str
    },

    "rating": {
        "required": False,
        "dtype": float
    },

    "popularity": {
        "required": False,
        "dtype": float
    },

    "preparation_time_minutes": {
        "required": False,
        "dtype": float
    },

    # ---------- Media ----------

    "image_url": {
        "required": False,
        "dtype": str
    },

    "url": {
        "required": False,
        "dtype": str
    },

    "source": {
        "required": False,
        "dtype": str
    }

}

class AliasRegistry:
    """
    Stores all aliases for the internal DineAI schema.

    Example
    -------
    recipe_name:
        - recipe
        - dish_name
        - title
        - name

    calories:
        - calories
        - kcal
        - energy
    """

    def __init__(self):

        self.aliases = {

    # ---------- Basic Information ----------

    "recipe_name": [
        "recipe_name",
        "recipe",
        "dish_name",
        "dish",
        "title",
        "name",
        "item_name",
        "menu_item_name",
        "menu_item",
        "item",
        "product_name",
        "food_name",
        "menu_item_id"
    ],

    "item_type": [
        "item_type",
        "dish_category",
        "category_type",
        "type",
        "item_category"
    ],

    "description": [
        "description",
        "summary",
        "details"
    ],

    "meal_type": [
        "meal_type",
        "meal",
        "meal_category",
        "category"
    ],

    "dish_type": [
        "dish_type",
        "course",
        "food_type",
        "dish_category"
    ],

    "cuisine_type": [
        "cuisine",
        "cuisine_type",
        "origin"
    ],

    "ingredients": [
        "ingredients",
        "ingredient_lines",
        "ingredient_list"
    ],

    "servings": [
        "servings",
        "serves",
        "yield",
        "yield_count"
    ],

    # ---------- Nutrition ----------

    "calories_per_serving": [
        "calories",
        "kcal",
        "energy",
        "energy_kcal",
        "calories_per_serving"
    ],

    "protein_g_per_serving": [
        "protein",
        "protein_g",
        "protein_grams",
        "proteingrams",
        "protein_per_serving"
    ],

    "carbs_g_per_serving": [
        "carbs",
        "carbohydrates",
        "carbohydrate",
        "carbs_g",
        "carbs_per_serving"
    ],

    "net_carbs_g_per_serving": [
        "net_carbs",
        "net_carbs_g",
        "net_carbs_per_serving"
    ],

    "fat_g_per_serving": [
        "fat",
        "fat_g",
        "total_fat",
        "fat_per_serving"
    ],

    "saturated_fat_g_per_serving": [
        "saturated_fat",
        "sat_fat",
        "saturated_fat_g"
    ],

    "trans_fat_g_per_serving": [
        "trans_fat",
        "trans_fat_g"
    ],

    "monounsaturated_fat_g_per_serving": [
        "monounsaturated_fat",
        "monounsaturated_fat_g"
    ],

    "polyunsaturated_fat_g_per_serving": [
        "polyunsaturated_fat",
        "polyunsaturated_fat_g"
    ],

    "fiber_g_per_serving": [
        "fiber",
        "fiber_g"
    ],

    "sugar_g_per_serving": [
        "sugar",
        "sugar_g"
    ],

    "added_sugar_g_per_serving": [
        "added_sugar",
        "added_sugar_g"
    ],

    "sodium_mg_per_serving": [
        "sodium",
        "sodium_mg"
    ],

    "cholesterol_mg_per_serving": [
        "cholesterol",
        "cholesterol_mg"
    ],

    # ---------- Labels ----------

    "diet_labels": [
        "diet_labels",
        "diet",
        "diet_types"
    ],

    "health_labels": [
        "health_labels",
        "health",
        "health_tags"
    ],

    "cautions": [
        "cautions",
        "allergens",
        "warnings"
    ],

    # ---------- Restaurant ----------

    "price": [
        "price",
        "cost",
        "menu_price"
    ],

    "currency": [
        "currency"
    ],

    "rating": [
        "rating",
        "stars",
        "score"
    ],

    "popularity": [
        "popularity",
        "orders",
        "likes"
    ],

    "preparation_time_minutes": [
        "prep_time",
        "preparation_time",
        "preparation_time_minutes",
        "cook_time"
    ],

    # ---------- Media ----------

    "image_url": [
        "image",
        "image_url",
        "photo"
    ],

    "url": [
        "url",
        "link"
    ],

    "source": [
        "source",
        "provider"
    ]
}

        self._validate_aliases()

    # --------------------------------------------------

    def _validate_canonical_field(self, canonical_name: str) -> None:
        """
        Validate that a canonical field name exists within the CANONICAL_SCHEMA.

        Parameters
        ----------
        canonical_name : str
            The field name to check.

        Raises
        ------
        ValueError
            If the canonical name does not exist in CANONICAL_SCHEMA.
        """
        if canonical_name not in CANONICAL_SCHEMA:
            raise ValueError(
                f"Invalid alias registry entry: The key '{canonical_name}' "
                f"does not exist in the canonical schema (CANONICAL_SCHEMA)."
            )

    def _validate_aliases(self) -> None:
        """
        Perform batch validation on all current keys in self.aliases.

        Raises
        ------
        ValueError
            If any key in self.aliases is invalid.
        """
        for canonical_name in self.aliases:
            self._validate_canonical_field(canonical_name)

    # --------------------------------------------------

    def get_aliases(self) -> Dict[str, List[str]]:
        """
        Return the whole alias dictionary.
        """
        return self.aliases

    # --------------------------------------------------

    def get(self, canonical_name: str) -> List[str]:
        """
        Return aliases for one field.
        """

        return self.aliases.get(canonical_name, [])

    # --------------------------------------------------

    def add_alias(
        self,
        canonical_name: str,
        alias: str
    ):
        self._validate_canonical_field(canonical_name)

        if canonical_name not in self.aliases:
            self.aliases[canonical_name] = []

        if alias not in self.aliases[canonical_name]:
            self.aliases[canonical_name].append(alias)

    # --------------------------------------------------

    def remove_alias(
        self,
        canonical_name: str,
        alias: str
    ):

        if canonical_name in self.aliases:

            if alias in self.aliases[canonical_name]:

                self.aliases[canonical_name].remove(alias)

    # --------------------------------------------------

    def find_canonical_name(
        self,
        column_name: str
    ) -> Optional[str]:
        """
        Given a normalized column name,
        return its canonical schema field.
        """

        for canonical, aliases in self.aliases.items():

            if column_name in aliases:
                return canonical

        return None

class AliasMatcher:
    """
    Matches normalized dataset columns to the DineAI canonical schema.

    This class is stateless.
    Every call returns a new mapping result.
    """

    def __init__(self, registry: AliasRegistry):
        self.registry = registry

    # --------------------------------------------------

    def match_columns(self, columns: List[str]) -> Dict:
        """
        Matches normalized dataset columns to the DineAI canonical schema.

        This method is stateless. It tracks already-mapped canonical fields.
        If multiple columns attempt to map to the same canonical field,
        only the first encountered column succeeds. Subsequent attempts
        are rejected, placed in unmatched columns, and warning messages are recorded.

        Parameters
        ----------
        columns : List[str]
            List of normalized dataset columns to map.

        Returns
        -------
        Dict
            A dictionary containing:
            - "mapping": Dict[str, str] mapping source columns to canonical names.
            - "matched": List[str] source columns successfully matched.
            - "unmatched": List[str] source columns rejected or not recognized.
            - "warnings": List[str] warning messages explaining why mappings were rejected.
            - "diagnostics": Dict[str, str] diagnostic status message for each unmapped/rejected column.
        """
        mapping = {}
        matched = []
        unmatched = []
        warnings = []
        diagnostics = {}
        used_canonicals = set()

        for column in columns:
            canonical = self.registry.find_canonical_name(column)

            if canonical is not None:
                if canonical in used_canonicals:
                    # Find which column originally mapped to this canonical name
                    first_mapped = next(
                        col for col, canon in mapping.items() if canon == canonical
                    )
                    warnings.append(
                        f"Duplicate mapping rejected: Column '{column}' was rejected because "
                        f"canonical field '{canonical}' is already mapped to by '{first_mapped}'."
                    )
                    unmatched.append(column)
                    diagnostics[column] = f"REJECTED - Duplicate of '{first_mapped}'"
                else:
                    used_canonicals.add(canonical)
                    mapping[column] = canonical
                    matched.append(column)
            else:
                unmatched.append(column)
                diagnostics[column] = "UNKNOWN - No matching canonical alias"

        return {
            "mapping": mapping,
            "matched": matched,
            "unmatched": unmatched,
            "warnings": warnings,
            "diagnostics": diagnostics
        }

    def generate_report(self, result: Dict):
        """
        Prints a detailed mapping report by delegating to the standalone mapping_report.
        """
        mapping_report(result)


def build_text_report(result: Dict) -> str:
    """
    Constructs a detailed schema mapping report as a formatted string.
    """
    total_cols = len(result.get("matched", [])) + len(result.get("unmatched", []))
    mapped_cols = len(result.get("matched", []))
    unmapped_cols = len(result.get("unmatched", []))
    coverage = (mapped_cols / total_cols * 100) if total_cols > 0 else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append(f"{'DINEAI SCHEMA MAPPING REPORT':^80}")
    lines.append("=" * 80)
    lines.append("MAPPING STATISTICS:")
    lines.append("-" * 80)
    lines.append(f"  Total Input Columns   : {total_cols}")
    lines.append(f"  Mapped Columns        : {mapped_cols}")
    lines.append(f"  Unmapped/Rejected     : {unmapped_cols}")
    lines.append(f"  Mapping Coverage      : {coverage:.2f}%")
    lines.append("=" * 80)

    # Mapped Columns Section
    lines.append("MAPPED COLUMNS:")
    lines.append("-" * 80)
    lines.append(f"  {'Source Column':<35} | {'Target Canonical Field':<28} | {'Status':<8}")
    lines.append("-" * 80)
    for original, canonical in result.get("mapping", {}).items():
        lines.append(f"  {original:<35} | {canonical:<28} | {'MAPPED':<8}")

    # Unmapped / Rejected Section if any
    if result.get("unmatched"):
        diagnostics = result.get("diagnostics", {})
        lines.append("=" * 80)
        lines.append("UNMAPPED / REJECTED COLUMNS:")
        lines.append("-" * 80)
        lines.append(f"  {'Source Column':<35} | {'Reason / Diagnosis':<38}")
        lines.append("-" * 80)
        for column in result["unmatched"]:
            reason = diagnostics.get(column, "No canonical alias matched")
            lines.append(f"  {column:<35} | {reason:<38}")

    # Warnings list if there are additional details
    warnings = result.get("warnings", [])
    if warnings:
        lines.append("=" * 80)
        lines.append("MAPPING WARNINGS / REJECTIONS LOG:")
        lines.append("-" * 80)
        for warning in warnings:
            lines.append(f"  [WARN] {warning}")

    lines.append("=" * 80)
    return "\n".join(lines)


def mapping_report(result: Dict) -> None:
    """
    Prints a detailed schema mapping report including metrics, mapped columns,
    and unmatched columns in an enterprise ETL-style format.
    """
    print(build_text_report(result))


import pandas as pd


class SchemaMapper:
    """
    Main DineAI schema mapper. Completely stateless.

    Pipeline:
        1. Normalize column names
        2. Match to canonical schema
        3. Rename dataframe columns
        4. Return standardized dataframe and the mapping result
    """

    def __init__(self):
        self.registry = AliasRegistry()
        self.matcher = AliasMatcher(self.registry)

    # --------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """
        Normalize and map dataframe columns statelessly.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to normalize and map.

        Returns
        -------
        tuple[pandas.DataFrame, Dict]
            A tuple containing:
            - The mapped/standardized pandas.DataFrame.
            - The mapping result Dict containing 'mapping', 'matched', 'unmatched', etc.
        """
        df = df.copy()

        # ------------------------------------------
        # Normalize column names
        # ------------------------------------------
        normalized_columns = [
            ColumnNormalizer.normalize(col)
            for col in df.columns
        ]
        df.columns = normalized_columns

        # ------------------------------------------
        # Match aliases
        # ------------------------------------------
        result = self.matcher.match_columns(df.columns)

        # ------------------------------------------
        # Rename dataframe
        # ------------------------------------------
        df = df.rename(columns=result["mapping"])

        return df, result

    # --------------------------------------------------
    # Backward Compatibility / Helper Methods (Stateless)
    # --------------------------------------------------

    def get_mapping(self, result: Dict) -> Dict[str, str]:
        """
        Extract mapping dictionary from result.
        """
        return result.get("mapping", {})

    def get_unmatched_columns(self, result: Dict) -> List[str]:
        """
        Extract unmatched columns from result.
        """
        return result.get("unmatched", [])

    def get_warnings(self, result: Dict) -> List[str]:
        """
        Extract mapping warnings from result.
        """
        return result.get("warnings", [])

    # --------------------------------------------------
    # Exporters
    # --------------------------------------------------

    def export_report(
        self,
        result: Dict,
        filepath: str,
        file_format: Optional[str] = None
    ) -> str:
        """
        Exports the mapping report to a specified file format (JSON, CSV, or TXT).

        Parameters
        ----------
        result : Dict
            The mapping result dictionary.
        filepath : str
            The destination file path.
        file_format : Optional[str]
            The export format ('json', 'csv', 'txt'). If None, it will be inferred
            from the file extension of the filepath.

        Returns
        -------
        str
            The absolute path of the exported file.
        """
        import os
        import json
        import csv

        # Infer format from filepath extension if not explicitly set
        if not file_format:
            _, ext = os.path.splitext(filepath)
            file_format = ext.lstrip(".").lower()

        if file_format not in ("json", "csv", "txt"):
            raise ValueError(
                f"Unsupported export format '{file_format}'. Supported formats: json, csv, txt."
            )

        total_cols = len(result.get("matched", [])) + len(result.get("unmatched", []))
        mapped_cols = len(result.get("matched", []))
        unmapped_cols = len(result.get("unmatched", []))
        coverage = (mapped_cols / total_cols * 100) if total_cols > 0 else 0.0

        if file_format == "json":
            unmatched_data = []
            diagnostics = result.get("diagnostics", {})
            for col in result.get("unmatched", []):
                unmatched_data.append({
                    "column": col,
                    "reason": diagnostics.get(col, "No canonical alias matched")
                })

            export_data = {
                "statistics": {
                    "total_columns": total_cols,
                    "mapped_columns": mapped_cols,
                    "unmapped_columns": unmapped_cols,
                    "mapping_coverage_pct": round(coverage, 2)
                },
                "mapping": result.get("mapping", {}),
                "unmatched_columns": unmatched_data,
                "warnings": result.get("warnings", [])
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)

        elif file_format == "csv":
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Section", "Source_Column", "Target_Value", "Reason_Status"])
                
                # Write statistics
                writer.writerow(["statistic", "total_columns", str(total_cols), "N/A"])
                writer.writerow(["statistic", "mapped_columns", str(mapped_cols), "N/A"])
                writer.writerow(["statistic", "unmapped_columns", str(unmapped_cols), "N/A"])
                writer.writerow(["statistic", "mapping_coverage_pct", f"{coverage:.2f}", "N/A"])
                
                # Write mapping
                for original, canonical in result.get("mapping", {}).items():
                    writer.writerow(["mapping", original, canonical, "MAPPED"])
                
                # Write unmatched
                diagnostics = result.get("diagnostics", {})
                for col in result.get("unmatched", []):
                    reason = diagnostics.get(col, "No canonical alias matched")
                    writer.writerow(["unmatched", col, "N/A", reason])
                
                # Write warnings
                for warning in result.get("warnings", []):
                    writer.writerow(["warning", "N/A", warning, "N/A"])

        elif file_format == "txt":
            report_text = build_text_report(result)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)

        return os.path.abspath(filepath)


if __name__ == "__main__":

    mapper = SchemaMapper()

    df = pd.DataFrame({
        "dish_name": ["Pasta", "Salad"],
        "title": ["Italian Pasta", "Garden Salad"], # Duplicate mapping: 'title' and 'dish_name' both map to 'recipe_name'
        "calories": [500, 300],
        "protein": [20, 10],
        "carbs": [60, 25],
        "fat": [20, 15],
        "price": [15.0, 12.0],
        "unknown_column": [1, 2]
    })

    standardized_df, mapping_result = mapper.fit_transform(df)

    print("Standardized DataFrame:")
    print(standardized_df)

    print("\nMapping Report:")
    mapping_report(mapping_result)

    # Exporters Test
    print("\nExporting Reports...")
    json_path = mapper.export_report(mapping_result, "mapping_report.json")
    csv_path = mapper.export_report(mapping_result, "mapping_report.csv")
    txt_path = mapper.export_report(mapping_result, "mapping_report.txt")
    print(f"Exported JSON to: {json_path}")
    print(f"Exported CSV to:  {csv_path}")
    print(f"Exported TXT to:  {txt_path}")