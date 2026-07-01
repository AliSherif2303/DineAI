import sys
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

# Ensure the root workspace directory is in the Python path when running as a direct script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd


@dataclass
class FeatureEngineerConfig:
    """
    Centralized configuration holding thresholds, bins, and score weights
    for all Feature Engineering modules.
    """
    # Nutrition Thresholds
    high_protein_ratio: float = 0.20
    low_carb_ratio: float = 0.30
    low_fat_ratio: float = 0.30
    low_sodium_mg: float = 140.0
    high_fiber_g: float = 5.0

    # Nutrition Score Weights
    nutrition_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "fiber": 2.0,
        "protein": 1.0,
        "sodium": -0.05,
        "sugar": -1.0,
        "saturated_fat": -2.0
    })

    # Dietary Thresholds
    low_calorie_threshold: float = 150.0
    high_protein_threshold_g: float = 15.0
    keto_carbs_g_threshold: float = 20.0

    # Pricing Buckets
    price_bins: List[float] = field(default_factory=lambda: [-float('inf'), 10.0, 25.0, float('inf')])
    price_labels: List[str] = field(default_factory=lambda: ["budget", "mid-range", "premium"])
    price_levels: List[str] = field(default_factory=lambda: ["$", "$$", "$$$"])
    price_buckets: List[str] = field(default_factory=lambda: ["<=$10", "$10-$25", ">$25"])



from enum import Enum


class FeatureCategory(Enum):
    """
    Strongly typed categories of engineered features.
    """
    NUTRITION = "nutrition"
    DIETARY = "dietary"
    MEAL = "meal"
    METADATA = "metadata"
    PRICE = "price"


@dataclass
class FeatureDefinition:
    """
    Metadata describing an engineered feature column for automated documentation.
    """
    feature_name: str
    description: str
    dependencies: List[str]
    dtype: Any
    created_by: str
    category: FeatureCategory


class FeatureRegistry:
    """
    Registry collecting, searching, and exporting engineered feature definitions.
    """

    def __init__(self, engineers: List["BaseFeatureEngineer"]):
        self.definitions: List[FeatureDefinition] = []
        for eng in engineers:
            self.definitions.extend(eng.get_definitions())

    def list_features(self) -> List[FeatureDefinition]:
        """
        Lists all registered feature definitions.
        """
        return self.definitions

    def search(self, query: str) -> List[FeatureDefinition]:
        """
        Searches feature definitions by name or description (case-insensitive).
        """
        query_lower = query.lower()
        return [
            d for d in self.definitions
            if query_lower in d.feature_name.lower() or query_lower in d.description.lower()
        ]

    def export_json(self, filepath: str) -> None:
        """
        Exports the registry documentation to a JSON file.
        """
        import json
        data = [
            {
                "feature_name": d.feature_name,
                "description": d.description,
                "dependencies": d.dependencies,
                "dtype": getattr(d.dtype, "__name__", str(d.dtype)),
                "created_by": d.created_by,
                "category": d.category.value
            }
            for d in self.definitions
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def export_markdown(self, filepath: str) -> None:
        """
        Exports the registry documentation to a Markdown table file.
        """
        lines = [
            "# DineAI Feature Registry Documentation",
            "",
            "| Feature Name | Category | Type | Description | Dependencies | Created By |",
            "|---|---|---|---|---|---|",
        ]
        for d in self.definitions:
            dtype_str = getattr(d.dtype, "__name__", str(d.dtype))
            deps_str = ", ".join(d.dependencies) if d.dependencies else "None"
            lines.append(
                f"| `{d.feature_name}` | {d.category.value} | `{dtype_str}` | {d.description} | {deps_str} | {d.created_by} |"
            )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


@dataclass
class EngineerMetadata:
    """
    Performance and diagnostic metrics for an individual feature engineer execution.
    """
    name: str
    description: str
    version: str
    execution_time_seconds: float = 0.0
    features_created: List[str] = field(default_factory=list)


from datetime import datetime


@dataclass
class FeatureWarning:
    """
    Structured warning detail generated during feature engineering.
    """
    engineer: str
    feature: Optional[str]
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureEngineeringResult:
    """
    Diagnostic result container that aggregates telemetry across all feature engineers.
    """
    is_success: bool = True
    created_features: List[str] = field(default_factory=list)
    warnings: List[FeatureWarning] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, EngineerMetadata] = field(default_factory=dict)


class BaseFeatureEngineer(ABC):
    """
    Abstract Base Class contract for pluggable, stateless Feature Engineering components.
    """

    required_columns: List[str] = []
    optional_columns: List[str] = []

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        """
        Initializes the base feature engineer with optional configuration.
        """
        self.config = config if config is not None else FeatureEngineerConfig()

    @abstractmethod
    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        """
        Executes vectorized feature engineering operations on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to enrich.
        result : FeatureEngineeringResult
            The shared collector for recording feature telemetry and status.

        Returns
        -------
        pandas.DataFrame
            The enriched DataFrame containing the new engineered columns.
        """
        pass

    @abstractmethod
    def get_definitions(self) -> List[FeatureDefinition]:
        """
        Returns a list of feature definitions generated by this component.

        Returns
        -------
        List[FeatureDefinition]
            Metadata definitions of the created columns.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the class name of the feature engineer.
        """
        return self.__class__.__name__


class NutritionFeatureEngineer(BaseFeatureEngineer):
    """
    Engineers nutritional density features, scores, and categorizations
    using macronutrient and calorie fields.
    """

    required_columns = ["calories_per_serving"]
    optional_columns = [
        "protein_g_per_serving", "carbs_g_per_serving", "fat_g_per_serving",
        "fiber_g_per_serving", "sodium_mg_per_serving", "sugar_g_per_serving",
        "saturated_fat_g_per_serving"
    ]

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name
        
        # Helper to do division safely
        def safe_div(num_col, den_col):
            if num_col not in df.columns or den_col not in df.columns:
                return pd.Series(0.0, index=df.index)
            den_clean = df[den_col].astype(float).replace(0.0, float('nan'))
            return (df[num_col].astype(float) / den_clean).fillna(0.0)

        # 1. Densities (macro grams / calories)
        if "protein_g_per_serving" in df.columns:
            df["protein_density"] = safe_div("protein_g_per_serving", "calories_per_serving")
            created.append("protein_density")
            
            # protein_calorie_ratio (g protein per 100 kcal)
            df["protein_calorie_ratio"] = df["protein_density"] * 100
            created.append("protein_calorie_ratio")
            
            # is_high_protein (>= high_protein_ratio of calories from protein. 1g protein = 4 kcal)
            df["is_high_protein"] = (df["protein_g_per_serving"].astype(float) * 4) / df["calories_per_serving"].astype(float).replace(0.0, float('nan')) >= self.config.high_protein_ratio
            df["is_high_protein"] = df["is_high_protein"].fillna(False)
            created.append("is_high_protein")

        if "carbs_g_per_serving" in df.columns:
            df["carb_density"] = safe_div("carbs_g_per_serving", "calories_per_serving")
            created.append("carb_density")
            
            # is_low_carb (<= low_carb_ratio calories from carbs)
            df["is_low_carb"] = (df["carbs_g_per_serving"].astype(float) * 4) / df["calories_per_serving"].astype(float).replace(0.0, float('nan')) <= self.config.low_carb_ratio
            df["is_low_carb"] = df["is_low_carb"].fillna(False)
            created.append("is_low_carb")

        if "fat_g_per_serving" in df.columns:
            df["fat_density"] = safe_div("fat_g_per_serving", "calories_per_serving")
            created.append("fat_density")
            
            # is_low_fat (<= low_fat_ratio calories from fat)
            df["is_low_fat"] = (df["fat_g_per_serving"].astype(float) * 9) / df["calories_per_serving"].astype(float).replace(0.0, float('nan')) <= self.config.low_fat_ratio
            df["is_low_fat"] = df["is_low_fat"].fillna(False)
            created.append("is_low_fat")

        if "fiber_g_per_serving" in df.columns:
            df["fiber_density"] = safe_div("fiber_g_per_serving", "calories_per_serving")
            created.append("fiber_density")

        # 2. Nutrition Score (comprehensive metric)
        score = pd.Series(0.0, index=df.index)
        w = self.config.nutrition_score_weights
        if "fiber_g_per_serving" in df.columns:
            score += df["fiber_g_per_serving"].fillna(0.0) * w.get("fiber", 2.0)
        if "protein_g_per_serving" in df.columns:
            score += df["protein_g_per_serving"].fillna(0.0) * w.get("protein", 1.0)
        if "sodium_mg_per_serving" in df.columns:
            score += df["sodium_mg_per_serving"].fillna(0.0) * w.get("sodium", -0.05)
        if "sugar_g_per_serving" in df.columns:
            score += df["sugar_g_per_serving"].fillna(0.0) * w.get("sugar", -1.0)
        if "saturated_fat_g_per_serving" in df.columns:
            score += df["saturated_fat_g_per_serving"].fillna(0.0) * w.get("saturated_fat", -2.0)
            
        df["nutrition_score"] = score
        created.append("nutrition_score")

        # 3. Simple threshold flags
        if "sodium_mg_per_serving" in df.columns:
            df["is_low_sodium"] = df["sodium_mg_per_serving"] < self.config.low_sodium_mg
            df["is_low_sodium"] = df["is_low_sodium"].fillna(False)
            created.append("is_low_sodium")
            
        if "fiber_g_per_serving" in df.columns:
            df["is_high_fiber"] = df["fiber_g_per_serving"] >= self.config.high_fiber_g
            df["is_high_fiber"] = df["is_high_fiber"].fillna(False)
            created.append("is_high_fiber")

        # Log metadata
        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Computes nutritional densities, composite health scores, and binary diet flags.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)
        
        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.NUTRITION
        return [
            FeatureDefinition("protein_density", "Ratio of protein grams to calories", ["protein_g_per_serving", "calories_per_serving"], float, name, cat),
            FeatureDefinition("protein_calorie_ratio", "Grams of protein per 100 calories", ["protein_g_per_serving", "calories_per_serving"], float, name, cat),
            FeatureDefinition("is_high_protein", "True if >= 20% of calories are from protein", ["protein_g_per_serving", "calories_per_serving"], bool, name, cat),
            FeatureDefinition("carb_density", "Ratio of carbohydrate grams to calories", ["carbs_g_per_serving", "calories_per_serving"], float, name, cat),
            FeatureDefinition("is_low_carb", "True if <= 30% of calories are from carbohydrates", ["carbs_g_per_serving", "calories_per_serving"], bool, name, cat),
            FeatureDefinition("fat_density", "Ratio of fat grams to calories", ["fat_g_per_serving", "calories_per_serving"], float, name, cat),
            FeatureDefinition("is_low_fat", "True if <= 30% of calories are from fat", ["fat_g_per_serving", "calories_per_serving"], bool, name, cat),
            FeatureDefinition("fiber_density", "Ratio of fiber grams to calories", ["fiber_g_per_serving", "calories_per_serving"], float, name, cat),
            FeatureDefinition("nutrition_score", "Composite metric scoring overall healthiness of recipe", [], float, name, cat),
            FeatureDefinition("is_low_sodium", "True if sodium is less than 140mg", ["sodium_mg_per_serving"], bool, name, cat),
            FeatureDefinition("is_high_fiber", "True if fiber is greater than or equal to 5g", ["fiber_g_per_serving"], bool, name, cat),
        ]


class DietaryFeatureEngineer(BaseFeatureEngineer):
    """
    Infers dietary attributes (vegan, vegetarian, gluten-free, dairy-free, keto, halal,
    low-calorie, high-protein) from labels, nutritional metrics, and ingredients lists.
    """

    required_columns = []
    optional_columns = [
        "health_labels", "ingredients", "diet_labels", 
        "calories_per_serving", "protein_g_per_serving", "carbs_g_per_serving"
    ]

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name

        def list_contains_any(col_name, keywords):
            import re
            if col_name not in df.columns:
                return pd.Series(False, index=df.index)
            def check(val):
                if isinstance(val, list):
                    items = val
                elif isinstance(val, str):
                    items = [i.strip() for i in re.split(r'[,;]', val)]
                else:
                    return False
                return any(any(kw in str(item).lower() for kw in keywords) for item in items)
            return df[col_name].apply(check)

        # 1. Vegan & Vegetarian
        has_vegan_label = list_contains_any("health_labels", ["vegan"])
        has_animal_ing = list_contains_any(
            "ingredients", 
            ["meat", "chicken", "pork", "beef", "fish", "egg", "milk", "cheese", 
             "butter", "cream", "yogurt", "honey", "lamb", "duck", "turkey", 
             "seafood", "shrimp", "crab", "lobster", "gelatin", "lard"]
        )
        df["is_vegan"] = has_vegan_label
        if "ingredients" in df.columns:
            df["is_vegan"] = df["is_vegan"] | (~has_animal_ing & df["ingredients"].notna())
        df["is_vegan"] = df["is_vegan"].fillna(False).astype(bool)
        created.append("is_vegan")

        has_veg_label = list_contains_any("health_labels", ["vegetarian", "vegan"])
        has_meat_ing = list_contains_any(
            "ingredients", 
            ["meat", "chicken", "pork", "beef", "fish", "lamb", "duck", "turkey", 
             "seafood", "shrimp", "crab", "lobster", "lard"]
        )
        df["is_vegetarian"] = has_veg_label | df["is_vegan"]
        if "ingredients" in df.columns:
            df["is_vegetarian"] = df["is_vegetarian"] | (~has_meat_ing & df["ingredients"].notna())
        df["is_vegetarian"] = df["is_vegetarian"].fillna(False).astype(bool)
        created.append("is_vegetarian")

        # 2. Gluten-Free
        has_gf_label = list_contains_any("health_labels", ["gluten-free", "gluten free"])
        has_gluten_ing = list_contains_any(
            "ingredients", 
            ["wheat", "barley", "rye", "flour", "semolina", "spelt", "pasta", "bread"]
        )
        df["is_gluten_free"] = has_gf_label
        if "ingredients" in df.columns:
            df["is_gluten_free"] = df["is_gluten_free"] | (~has_gluten_ing & df["ingredients"].notna())
        df["is_gluten_free"] = df["is_gluten_free"].fillna(False).astype(bool)
        created.append("is_gluten_free")

        # 3. Dairy-Free
        has_df_label = list_contains_any("health_labels", ["dairy-free", "dairy free"])
        has_dairy_ing = list_contains_any(
            "ingredients", 
            ["milk", "cheese", "butter", "cream", "yogurt", "whey", "casein", "ghee"]
        )
        df["is_dairy_free"] = has_df_label
        if "ingredients" in df.columns:
            df["is_dairy_free"] = df["is_dairy_free"] | (~has_dairy_ing & df["ingredients"].notna())
        df["is_dairy_free"] = df["is_dairy_free"].fillna(False).astype(bool)
        created.append("is_dairy_free")

        # 4. Keto
        has_keto_label = list_contains_any("diet_labels", ["keto"]) | list_contains_any("health_labels", ["keto"])
        is_keto_macros = pd.Series(False, index=df.index)
        if "carbs_g_per_serving" in df.columns:
            is_keto_macros = df["carbs_g_per_serving"].fillna(100.0) < self.config.keto_carbs_g_threshold
        df["is_keto"] = has_keto_label | is_keto_macros
        df["is_keto"] = df["is_keto"].fillna(False).astype(bool)
        created.append("is_keto")

        # 5. Halal
        has_halal_label = list_contains_any("health_labels", ["halal"])
        has_haram_ing = list_contains_any("ingredients", ["pork", "lard", "alcohol", "wine", "beer", "gelatin", "bacon"])
        df["is_halal"] = has_halal_label
        if "ingredients" in df.columns:
            df["is_halal"] = df["is_halal"] | (~has_haram_ing & df["ingredients"].notna())
        df["is_halal"] = df["is_halal"].fillna(False).astype(bool)
        created.append("is_halal")

        # 6. Low-Calorie
        if "calories_per_serving" in df.columns:
            df["is_low_calorie"] = df["calories_per_serving"] < self.config.low_calorie_threshold
            df["is_low_calorie"] = df["is_low_calorie"].fillna(False).astype(bool)
            created.append("is_low_calorie")

        # 7. High-Protein (if not already set, fallback to check >= high_protein_threshold_g)
        if "is_high_protein" not in df.columns:
            if "protein_g_per_serving" in df.columns:
                df["is_high_protein"] = df["protein_g_per_serving"] >= self.config.high_protein_threshold_g
                df["is_high_protein"] = df["is_high_protein"].fillna(False).astype(bool)
                created.append("is_high_protein")

        # Log metadata
        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Infers dietary markers (vegan, vegetarian, gluten-free, dairy-free, keto, halal) from metadata and ingredients.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)

        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.DIETARY
        return [
            FeatureDefinition("is_vegan", "True if the dish is vegan (no animal products)", ["health_labels", "ingredients"], bool, name, cat),
            FeatureDefinition("is_vegetarian", "True if the dish is vegetarian (no meat)", ["health_labels", "ingredients"], bool, name, cat),
            FeatureDefinition("is_gluten_free", "True if the dish is gluten-free", ["health_labels", "ingredients"], bool, name, cat),
            FeatureDefinition("is_dairy_free", "True if the dish contains no dairy products", ["health_labels", "ingredients"], bool, name, cat),
            FeatureDefinition("is_keto", "True if the dish conforms to ketogenic macro profile", ["diet_labels", "carbs_g_per_serving"], bool, name, cat),
            FeatureDefinition("is_halal", "True if the dish is halal", ["health_labels", "ingredients"], bool, name, cat),
            FeatureDefinition("is_low_calorie", "True if the dish is under 150 calories per serving", ["calories_per_serving"], bool, name, cat),
            FeatureDefinition("is_high_protein", "True if the dish contains >= 15g protein per serving", ["protein_g_per_serving"], bool, name, cat),
        ]


class MealFeatureEngineer(BaseFeatureEngineer):
    """
    Infers meal categories (breakfast, lunch, dinner, snack, dessert) using
    the meal_type and dish_type fields.
    """

    required_columns = []
    optional_columns = ["meal_type", "dish_type"]

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name

        def contains_keyword(col_name, keywords):
            if col_name not in df.columns:
                return pd.Series(False, index=df.index)
            
            def check_val(x):
                if isinstance(x, list):
                    return any(any(kw in str(item).lower() for kw in keywords) for item in x)
                elif pd.notna(x):
                    return any(kw in str(x).lower() for kw in keywords)
                return False
                
            return df[col_name].apply(check_val)

        # 1. breakfast
        df["is_breakfast"] = contains_keyword("meal_type", ["breakfast", "morning", "brunch"])
        df["is_breakfast"] = df["is_breakfast"].fillna(False).astype(bool)
        created.append("is_breakfast")

        # 2. lunch
        df["is_lunch"] = contains_keyword("meal_type", ["lunch", "brunch", "midday"])
        df["is_lunch"] = df["is_lunch"].fillna(False).astype(bool)
        created.append("is_lunch")

        # 3. dinner
        df["is_dinner"] = contains_keyword("meal_type", ["dinner", "supper", "evening", "night"])
        df["is_dinner"] = df["is_dinner"].fillna(False).astype(bool)
        created.append("is_dinner")

        # 4. snack
        df["is_snack"] = contains_keyword("meal_type", ["snack", "tea time"]) | contains_keyword("dish_type", ["snack", "starter", "appetizer", "hors d'oeuvre", "side dish"])
        df["is_snack"] = df["is_snack"].fillna(False).astype(bool)
        created.append("is_snack")

        # 5. dessert
        df["is_dessert"] = contains_keyword("dish_type", ["dessert", "sweet", "cake", "pastry", "cookie", "pie", "pudding"]) | contains_keyword("meal_type", ["dessert"])
        df["is_dessert"] = df["is_dessert"].fillna(False).astype(bool)
        created.append("is_dessert")

        # Log metadata
        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Infers meal times and courses (breakfast, lunch, dinner, snack, dessert) from input types.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)

        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.MEAL
        return [
            FeatureDefinition("is_breakfast", "True if the dish is intended for breakfast/brunch", ["meal_type"], bool, name, cat),
            FeatureDefinition("is_lunch", "True if the dish is intended for lunch", ["meal_type"], bool, name, cat),
            FeatureDefinition("is_dinner", "True if the dish is intended for dinner", ["meal_type"], bool, name, cat),
            FeatureDefinition("is_snack", "True if the dish is classified as a snack/appetizer/side dish", ["meal_type", "dish_type"], bool, name, cat),
            FeatureDefinition("is_dessert", "True if the dish is a dessert/sweet dish", ["meal_type", "dish_type"], bool, name, cat),
        ]


class PriceFeatureEngineer(BaseFeatureEngineer):
    """
    Categorizes items based on price bounds. Skips execution if no 'price' column exists.
    """

    required_columns = ["price"]
    optional_columns = []

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name

        prices = df["price"].astype(float)
        
        # budget_category
        df["budget_category"] = pd.cut(
            prices,
            bins=self.config.price_bins,
            labels=self.config.price_labels
        ).astype(str)
        created.append("budget_category")

        # price_level
        df["price_level"] = pd.cut(
            prices,
            bins=self.config.price_bins,
            labels=self.config.price_levels
        ).astype(str)
        created.append("price_level")

        # price_bucket
        df["price_bucket"] = pd.cut(
            prices,
            bins=self.config.price_bins,
            labels=self.config.price_buckets
        ).astype(str)
        created.append("price_bucket")

        # Log metadata
        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Categorizes pricing into budget category levels and buckets.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)

        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.PRICE
        return [
            FeatureDefinition("budget_category", "Categorized budget bracket ('budget', 'mid-range', 'premium')", ["price"], str, name, cat),
            FeatureDefinition("price_level", "Price rating level symbol ('$', '$$', '$$$')", ["price"], str, name, cat),
            FeatureDefinition("price_bucket", "Explicit pricing range values", ["price"], str, name, cat),
        ]


class MetadataFeatureEngineer(BaseFeatureEngineer):
    """
    Computes count characteristics (ingredients, health/diet labels) and string character lengths.
    """

    required_columns = []
    optional_columns = ["ingredients", "health_labels", "diet_labels", "recipe_name", "description"]

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name

        def safe_len(col_name):
            if col_name not in df.columns:
                return pd.Series(0, index=df.index)
            return df[col_name].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # 1. ingredient_count
        df["ingredient_count"] = safe_len("ingredients")
        df["ingredient_count"] = df["ingredient_count"].fillna(0).astype(int)
        created.append("ingredient_count")

        # 2. health_label_count
        df["health_label_count"] = safe_len("health_labels")
        df["health_label_count"] = df["health_label_count"].fillna(0).astype(int)
        created.append("health_label_count")

        # 3. diet_label_count
        df["diet_label_count"] = safe_len("diet_labels")
        df["diet_label_count"] = df["diet_label_count"].fillna(0).astype(int)
        created.append("diet_label_count")

        # 4. recipe_name_length
        if "recipe_name" in df.columns:
            df["recipe_name_length"] = df["recipe_name"].fillna("").astype(str).str.len()
            df["recipe_name_length"] = df["recipe_name_length"].fillna(0).astype(int)
            created.append("recipe_name_length")

        # 5. description_length
        if "description" in df.columns:
            df["description_length"] = df["description"].fillna("").astype(str).str.len()
            df["description_length"] = df["description_length"].fillna(0).astype(int)
            created.append("description_length")

        # Log metadata
        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Extracts metadata measurements (lengths, element list counts) from input fields.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)

        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.METADATA
        return [
            FeatureDefinition("ingredient_count", "Total number of ingredients listed", ["ingredients"], int, name, cat),
            FeatureDefinition("health_label_count", "Total number of health labels assigned", ["health_labels"], int, name, cat),
            FeatureDefinition("diet_label_count", "Total number of diet labels assigned", ["diet_labels"], int, name, cat),
            FeatureDefinition("recipe_name_length", "Character length of the recipe title", ["recipe_name"], int, name, cat),
            FeatureDefinition("description_length", "Character length of the recipe description", ["description"], int, name, cat),
        ]


class ItemTypeEngineer(BaseFeatureEngineer):
    """
    Engineers the 'item_type' column by auto-classifying items based on their recipe_name
    if 'item_type' is not already provided in the input DataFrame.
    """

    required_columns = ["recipe_name"]
    optional_columns = []

    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        super().__init__(config)

    def transform(self, df: pd.DataFrame, result: FeatureEngineeringResult) -> pd.DataFrame:
        import logging
        logger = logging.getLogger(__name__)
        start_time = time.time()
        df = df.copy()
        created = []
        name = self.name

        # Only apply classification if 'item_type' is not already a column
        if "item_type" not in df.columns:
            logger.info("ItemTypeEngineer: 'item_type' column not found in input. Auto-classifying items.")
            
            # Classification rules based on lowercase name keywords (in order of priority)
            def classify(title: str) -> str:
                if not isinstance(title, str):
                    return "main_course"
                t_lower = title.lower()
                
                # coffee
                if any(kw in t_lower for kw in ["coffee", "latte", "cappuccino", "espresso", "americano", "macchiato"]):
                    return "coffee"
                # smoothie
                if "smoothie" in t_lower:
                    return "smoothie"
                # juice
                if any(kw in t_lower for kw in ["juice", "lemonade"]):
                    return "juice"
                # drink
                if any(kw in t_lower for kw in ["coke", "pepsi", "water", "soda", "tea", "iced tea", "energy drink", "beer", "wine"]):
                    return "drink"
                # dessert
                if any(kw in t_lower for kw in ["cake", "chocolate", "ice cream", "gelato", "cookie", "brownie", "pudding", "cheesecake", "tart", "muffin"]):
                    return "dessert"
                # soup
                if any(kw in t_lower for kw in ["soup", "broth", "bisque", "chowder"]):
                    return "soup"
                # salad
                if "salad" in t_lower:
                    return "salad"
                # appetizer
                if any(kw in t_lower for kw in ["spring roll", "bruschetta", "nachos", "mozzarella sticks", "wings", "sampler", "starter"]):
                    return "appetizer"
                # sauce
                if any(kw in t_lower for kw in ["sauce", "dip", "dressing", "gravy", "salsa"]):
                    return "sauce"
                # side
                if any(kw in t_lower for kw in ["side", "fries", "rice", "coleslaw", "bread"]):
                    return "side"
                # combo
                if any(kw in t_lower for kw in ["combo", "deal", "set", "bundle", "platter"]):
                    return "combo"
                
                return "main_course"

            df["item_type"] = df["recipe_name"].apply(classify)
            created.append("item_type")
        else:
            logger.info("ItemTypeEngineer: 'item_type' column already present. Skipping classification.")

        duration = time.time() - start_time
        meta = EngineerMetadata(
            name=name,
            description="Auto-classifies menu item categories using title keyword mapping.",
            version="1.0.0",
            execution_time_seconds=duration,
            features_created=created
        )
        result.execution_metadata[name] = meta
        result.created_features.extend(created)

        return df

    def get_definitions(self) -> List[FeatureDefinition]:
        name = self.name
        cat = FeatureCategory.METADATA
        return [
            FeatureDefinition("item_type", "Standardized menu item category", ["recipe_name"], str, name, cat),
        ]


class FeatureEngineer:
    """
    Coordinator pipeline for DineAI feature engineering.
    Runs a list of pluggable engineers sequentially and statelessly.
    """

    def __init__(self, engineers: Optional[List[BaseFeatureEngineer]] = None, config: Optional[FeatureEngineerConfig] = None):
        """
        Initializes the FeatureEngineer coordinator.

        Parameters
        ----------
        engineers : Optional[List[BaseFeatureEngineer]]
            Pluggable list of feature engineers. If None, default set of 5 is registered.
        config : Optional[FeatureEngineerConfig]
            The configuration holding parameters for feature transformation.
        """
        self.config = config if config is not None else FeatureEngineerConfig()
        if engineers is not None:
            self.engineers = engineers
        else:
            self.engineers = [
                ItemTypeEngineer(self.config),
                NutritionFeatureEngineer(self.config),
                DietaryFeatureEngineer(self.config),
                MealFeatureEngineer(self.config),
                PriceFeatureEngineer(self.config),
                MetadataFeatureEngineer(self.config)
            ]
        self.registry = FeatureRegistry(self.engineers)
        self.last_result: Optional[FeatureEngineeringResult] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all registered feature engineers sequentially on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to enrich.

        Returns
        -------
        pandas.DataFrame
            The enriched DataFrame.
        """
        result = FeatureEngineeringResult()
        self.last_result = result
        df_enriched = df.copy()
        input_columns = set(df.columns)

        for eng in self.engineers:
            missing_reqs = [col for col in eng.required_columns if col not in df_enriched.columns]
            if missing_reqs:
                result.warnings.append(
                    FeatureWarning(
                        engineer=eng.name,
                        feature=None,
                        severity="WARNING",
                        message=f"Skipped. Required columns {missing_reqs} not found in input DataFrame."
                    )
                )
                result.execution_metadata[eng.name] = EngineerMetadata(
                    name=eng.name,
                    description=f"Skipped: Missing required columns {missing_reqs}",
                    version="1.0.0",
                    execution_time_seconds=0.0,
                    features_created=[]
                )
                continue

            # Record features count before transform
            features_before = list(result.created_features)

            try:
                df_enriched = eng.transform(df_enriched, result)
            except Exception as e:
                result.is_success = False
                result.warnings.append(
                    FeatureWarning(
                        engineer=eng.name,
                        feature=None,
                        severity="ERROR",
                        message=f"Unexpected error executing {eng.name}: {str(e)}"
                    )
                )
                # Log a failed metadata segment
                result.execution_metadata[eng.name] = EngineerMetadata(
                    name=eng.name,
                    description=f"Failed execution: {str(e)}",
                    version="1.0.0",
                    execution_time_seconds=0.0,
                    features_created=[]
                )
                continue

            # Determine features actually created in this step
            features_after = list(result.created_features)
            newly_created = [f for f in features_after if f not in features_before]

            # 1. Collision check: Overwriting input columns
            overwritten = [feat for feat in newly_created if feat in input_columns]
            if overwritten:
                raise ValueError(
                    f"Feature collision detected! The engineer '{eng.name}' attempted to "
                    f"overwrite existing input columns {overwritten}."
                )

            # 2. Collision check: Duplicate creation across engineers
            duplicates = [feat for feat in newly_created if feat in features_before]
            if duplicates:
                raise ValueError(
                    f"Feature collision detected! The engineer '{eng.name}' generated "
                    f"features {duplicates} which were already created by a previous engineer in the pipeline."
                )

        # Calculate statistics
        skipped_count = sum(1 for m in result.execution_metadata.values() if "Skipped" in m.description)
        failed_count = sum(1 for m in result.execution_metadata.values() if "Failed" in m.description)
        active_metadata = [
            m for m in result.execution_metadata.values()
            if "Skipped" not in m.description and "Failed" not in m.description
        ]
        avg_time = sum(m.execution_time_seconds for m in active_metadata) / len(active_metadata) if active_metadata else 0.0
        total_time = sum(m.execution_time_seconds for m in result.execution_metadata.values())

        result.statistics["total_features"] = len(result.created_features)
        result.statistics["skipped_engineers"] = skipped_count
        result.statistics["failed_engineers"] = failed_count
        result.statistics["total_execution_time_seconds"] = total_time
        result.statistics["average_engineer_time_seconds"] = avg_time
        result.statistics["new_columns_added"] = len(result.created_features)

        return df_enriched

    def generate_report(self) -> None:
        """
        Prints a detailed, enterprise-style Feature Engineering Report for the last execution.
        """
        if self.last_result is None:
            print("No feature engineering run has been executed yet.")
            return

        result = self.last_result
        print("=" * 60)
        print(f"{'Feature Engineering Report':^60}")
        print("=" * 60)

        for eng_name, meta in result.execution_metadata.items():
            status = "PASS" if meta.features_created or "Skipped" not in meta.description else "SKIPPED"
            if "Failed" in meta.description:
                status = "FAILED"
                
            print(f"{eng_name}")
            print(f"{status}")
            if meta.features_created:
                print("Created:")
                for feat in meta.features_created:
                    print(f"  - {feat}")
            elif status == "SKIPPED":
                print("Reason:")
                matching_warn = next((w.message for w in result.warnings if w.engineer == eng_name), "Skipped by rule.")
                print(f"  [WARN] {matching_warn}")
            print("-" * 60)

        print("=" * 60)
        print(f"Total Features Created   : {result.statistics.get('total_features', 0)}")
        print(f"New Columns Added        : {result.statistics.get('new_columns_added', 0)}")
        print(f"Skipped Feature Engines  : {result.statistics.get('skipped_engineers', 0)}")
        print(f"Failed Feature Engines   : {result.statistics.get('failed_engineers', 0)}")
        print(f"Average Engine Time      : {result.statistics.get('average_engineer_time_seconds', 0.0):.4f} seconds")
        print(f"Total Pipeline Runtime   : {result.statistics.get('total_execution_time_seconds', 0.0):.4f} seconds")
        
        # Display warnings if any
        if result.warnings:
            print("-" * 60)
            print("Runtime Warnings Log:")
            for warn in result.warnings:
                print(f"  [{warn.severity}] {warn.engineer}: {warn.message}")
                
        print("=" * 60)


if __name__ == "__main__":
    print("Executing self-test for FeatureEngineer pipeline...\n")

    # Sample DataFrame with valid columns for full transformation
    df_sample = pd.DataFrame([
        {
            "recipe_name": "Cheesy Truffle Risotto",
            "description": "Creamy risotto with cheese and black truffle oil.",
            "calories_per_serving": 400.0,
            "protein_g_per_serving": 12.0,
            "carbs_g_per_serving": 50.0,
            "fat_g_per_serving": 15.0,
            "fiber_g_per_serving": 3.0,
            "sodium_mg_per_serving": 120.0,
            "sugar_g_per_serving": 2.0,
            "saturated_fat_g_per_serving": 4.0,
            "health_labels": ["Vegetarian", "Gluten-Free"],
            "diet_labels": ["Low-Sodium"],
            "ingredients": ["Arborio Rice", "Parmesan Cheese", "Truffle Oil", "Vegetable Broth"],
            "meal_type": ["lunch", "dinner"],
            "dish_type": ["main course"],
            "price": 24.50,
        },
        {
            "recipe_name": "Keto Grilled Salmon",
            "description": "Fresh salmon fillet grilled with lemon and herbs.",
            "calories_per_serving": 350.0,
            "protein_g_per_serving": 30.0,
            "carbs_g_per_serving": 2.0,
            "fat_g_per_serving": 25.0,
            "fiber_g_per_serving": 0.0,
            "sodium_mg_per_serving": 90.0,
            "sugar_g_per_serving": 0.0,
            "saturated_fat_g_per_serving": 2.0,
            "health_labels": ["Gluten-Free", "Halal", "Keto"],
            "diet_labels": ["High-Protein", "Low-Carb"],
            "ingredients": ["Salmon Fillet", "Lemon Juice", "Olive Oil", "Dill"],
            "meal_type": ["dinner"],
            "dish_type": ["main course"],
            "price": 28.00,
        }
    ])

    # 1. Transform with Price column present
    engineer = FeatureEngineer()
    print("--- 1. Testing Transformation with Price Column ---")
    df_engineered = engineer.transform(df_sample)
    print(df_engineered[["recipe_name", "protein_density", "is_vegan", "is_keto", "is_dinner", "budget_category", "ingredient_count"]])
    engineer.generate_report()

    # 2. Transform with Price column missing (to test skip logic)
    print("\n--- 2. Testing PriceFeatureEngineer Skip Logic ---")
    df_no_price = df_sample.drop(columns=["price"])
    df_engineered_no_price = engineer.transform(df_no_price)
    # Check that price-dependent fields are not added
    price_fields = ["budget_category", "price_level", "price_bucket"]
    present_price_fields = [f for f in price_fields if f in df_engineered_no_price.columns]
    print(f"Price fields in output: {present_price_fields}")
    engineer.generate_report()
