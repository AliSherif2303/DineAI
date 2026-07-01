import sys
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Ensure the root workspace directory is in the Python path when running as a direct script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd


class L10nManager:
    """
    Manages translation catalogs loading from translations/ directory.
    """
    _translations: Dict[str, Dict[str, str]] = {}
    _default_lang: str = "en"

    @classmethod
    def load_translations(cls, translations_dir: str = "translations"):
        # If running as standard pipeline, lookup relative directory translations
        if not os.path.exists(translations_dir):
            # Try workspace root fallback relative to this file
            translations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "translations"))
            
        if os.path.exists(translations_dir):
            for filename in os.listdir(translations_dir):
                if filename.endswith(".json"):
                    lang = filename[:-5]
                    filepath = os.path.join(translations_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            cls._translations[lang] = json.load(f)
                    except Exception:
                        pass

    @classmethod
    def get(cls, key: str, lang: str = "en") -> str:
        if not cls._translations:
            cls.load_translations()
            
        lang_dict = cls._translations.get(lang, cls._translations.get(cls._default_lang, {}))
        return lang_dict.get(key, key)


def is_present(val) -> bool:
    """
    Checks if a cell value is present, handling lists, dictionaries, strings, and scalar NaNs.
    """
    if val is None:
        return False
    if isinstance(val, (list, set, tuple, dict)):
        return len(val) > 0
    if isinstance(val, str):
        return len(val.strip()) > 0
    try:
        return bool(pd.notna(val))
    except ValueError:
        # Handles any edge-cases where notna returns an array
        return True


@dataclass
class SectionContent:
    """
    Structured model holding content fields and metadata for a single section block.
    """
    title: Optional[str]
    fields: Dict[str, Any]
    bullet_keys: List[str] = field(default_factory=list)


class BaseRenderer(ABC):
    """
    Abstract Base Class defining the contract for layout presentation formats.
    """

    @abstractmethod
    def render_section(self, content: SectionContent) -> str:
        """
        Renders a single SectionContent block.
        """
        pass

    @abstractmethod
    def render_document(self, rendered_sections: List[str]) -> str:
        """
        Combines a list of rendered section strings into a single unified output structure.
        """
        pass


class PlainTextRenderer(BaseRenderer):
    """
    Renders structured section content as plain key-value text.
    """

    def render_section(self, content: SectionContent) -> str:
        lines = []
        if content.title:
            lines.append(f"{content.title}:")
        for key, value in content.fields.items():
            is_bullet = key in content.bullet_keys
            bullet_prefix = "- " if is_bullet else ""
            if isinstance(value, list):
                if key in ["items", "List", ""]:
                    for item in value:
                        lines.append(f"- {item}")
                else:
                    lines.append(f"{bullet_prefix}{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
            else:
                lines.append(f"{bullet_prefix}{key}: {value}")
        return "\n".join(lines)

    def render_document(self, rendered_sections: List[str]) -> str:
        return "\n\n".join([s for s in rendered_sections if s.strip()])


class MarkdownRenderer(BaseRenderer):
    """
    Renders structured section content as Markdown layout blocks.
    """

    def render_section(self, content: SectionContent) -> str:
        lines = []
        if content.title:
            lines.append(f"### {content.title}")
        for key, value in content.fields.items():
            is_bullet = key in content.bullet_keys
            if isinstance(value, list):
                if key in ["items", "List", ""]:
                    for item in value:
                        lines.append(f"* {item}")
                else:
                    prefix = "* " if is_bullet else ""
                    lines.append(f"{prefix}**{key}**:")
                    for item in value:
                        lines.append(f"  * {item}")
            else:
                if is_bullet:
                    lines.append(f"* **{key}**: {value}")
                else:
                    lines.append(f"**{key}**: {value}  ")
        return "\n".join(lines)

    def render_document(self, rendered_sections: List[str]) -> str:
        return "\n\n".join([s for s in rendered_sections if s.strip()])


class HTMLRenderer(BaseRenderer):
    """
    Renders structured section content as semantic HTML fragments.
    """

    def render_section(self, content: SectionContent) -> str:
        lines = []
        if content.title:
            lines.append(f"<h3>{content.title}</h3>")
        
        in_list = False
        for key, value in content.fields.items():
            is_bullet = key in content.bullet_keys
            if is_bullet and not in_list:
                lines.append("<ul>")
                in_list = True
            elif not is_bullet and in_list:
                lines.append("</ul>")
                in_list = False

            if isinstance(value, list):
                if key in ["items", "List", ""]:
                    list_nested = []
                    if not in_list:
                        list_nested.append("<ul>")
                    for item in value:
                        list_nested.append(f"  <li>{item}</li>")
                    if not in_list:
                        list_nested.append("</ul>")
                    lines.extend(list_nested)
                else:
                    if is_bullet:
                        lines.append(f"  <li><strong>{key}</strong>:")
                        lines.append("    <ul>")
                        for item in value:
                            lines.append(f"      <li>{item}</li>")
                        lines.append("    </ul>")
                        lines.append("  </li>")
                    else:
                        lines.append(f"<p><strong>{key}</strong>:</p>")
                        lines.append("<ul>")
                        for item in value:
                            lines.append(f"  <li>{item}</li>")
                        lines.append("</ul>")
            else:
                if is_bullet:
                    lines.append(f"  <li><strong>{key}</strong>: {value}</li>")
                else:
                    lines.append(f"<p><strong>{key}</strong>: {value}</p>")
        
        if in_list:
            lines.append("</ul>")
            
        return "\n".join(lines)

    def render_document(self, rendered_sections: List[str]) -> str:
        return "\n\n".join([s for s in rendered_sections if s.strip()])


import json


class JSONRenderer(BaseRenderer):
    """
    Serializes structured content as clean JSON documents.
    """

    def render_section(self, content: SectionContent) -> str:
        data = {
            "title": content.title,
            "fields": content.fields
        }
        return json.dumps(data)

    def render_document(self, rendered_sections: List[str]) -> str:
        docs = []
        for s in rendered_sections:
            if s.strip():
                try:
                    docs.append(json.loads(s))
                except Exception:
                    docs.append({"raw": s})
        return json.dumps(docs, indent=2)


@dataclass
class TextDefinition:
    """
    Metadata registry entry describing a generated text column for automatic documentation.
    """
    name: str
    description: str
    template_name: str
    sections_used: List[str]
    purpose: str


@dataclass
class BuilderMetadata:
    """
    Performance and diagnostic metrics for an individual section builder execution.
    """
    name: str
    description: str
    execution_time_seconds: float = 0.0
    sections_generated: List[str] = field(default_factory=list)


@dataclass
class TextWarning:
    """
    Structured warning details generated during text building execution.
    """
    builder_name: str
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TextBuildingResult:
    """
    Accumulator aggregating execution stats and warning telemetry across the TextBuilder pipeline.
    """
    is_success: bool = True
    generated_columns: List[str] = field(default_factory=list)
    warnings: List[TextWarning] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, BuilderMetadata] = field(default_factory=dict)


class BaseSectionBuilder(ABC):
    """
    Abstract Base Class contract for pluggable, stateless Section Builders.

    Class Attributes
    ----------------
    required_columns : List[str]
        Columns that MUST be present in the DataFrame for this builder to run.
        If any required column is absent, the builder is skipped entirely.
    optional_columns : List[str]
        Columns used by this builder but whose absence is tolerated (partial output).
    """

    required_columns: List[str] = []
    optional_columns: List[str] = []

    def __init__(self, language: str = "en"):
        self.language = language

    def can_build(self, df: pd.DataFrame) -> bool:
        """
        Returns True if all required_columns are present in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to validate columns against.

        Returns
        -------
        bool
            True when the builder's requirements are satisfied, False otherwise.
        """
        missing = [col for col in self.required_columns if col not in df.columns]
        return len(missing) == 0

    @abstractmethod
    def build_section(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates structured SectionContent objects from the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing standardized recipes/columns.

        Returns
        -------
        pandas.Series
            A pandas Series of SectionContent type containing the generated section content for each row.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the class name of the section builder.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Returns a description of what this section builder generates.
        """
        pass


class TextTemplate:
    """
    Defines the ordering and layout of sections to generate a unified textual column.
    """

    def __init__(
        self,
        name: str,
        purpose: str,
        target_column: str,
        sections: List[BaseSectionBuilder],
        description: str = "",
        renderer: Optional[BaseRenderer] = None
    ):
        """
        Initializes the TextTemplate.

        Parameters
        ----------
        name : str
            The unique name of the template.
        purpose : str
            What this text column is used for (e.g. semantic embeddings, LLM feeding).
        target_column : str
            The output column name inside the pandas DataFrame.
        sections : List[BaseSectionBuilder]
            Ordered list of section builders.
        description : str
            Description of the template purpose.
        renderer : Optional[BaseRenderer]
            The renderer implementation to convert structured sections into text.
        """
        self.name = name
        self.purpose = purpose
        self.target_column = target_column
        self.sections = sections
        self.description = description
        self.renderer = renderer if renderer is not None else PlainTextRenderer()

    def render(self, df: pd.DataFrame, section_cache: Dict[str, pd.Series]) -> pd.Series:
        """
        Stitches the cached sections for each row using the template's renderer.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame (used to resolve index matching).
        section_cache : Dict[str, pd.Series]
            Cache mapping section builder names to their generated SectionContent Series.

        Returns
        -------
        pandas.Series
            The joined textual representations Series.
        """
        def render_row(row_idx):
            rendered_sects = []
            for sec in self.sections:
                sec_series = section_cache.get(sec.name)
                if sec_series is not None:
                    # Retrieve the row value using loc
                    content = sec_series.loc[row_idx]
                    if content is not None:
                        rendered_sects.append(self.renderer.render_section(content))
            return self.renderer.render_document(rendered_sects)

        return pd.Series([render_row(idx) for idx in df.index], index=df.index)


class TextRegistry:
    """
    Registry for describing and documenting generated text output schemas.
    """

    def __init__(self, templates: List[TextTemplate]):
        self.definitions: List[TextDefinition] = []
        for temp in templates:
            self.definitions.append(
                TextDefinition(
                    name=temp.target_column,
                    description=temp.description,
                    template_name=temp.name,
                    sections_used=[sec.name for sec in temp.sections],
                    purpose=temp.purpose
                )
            )

    def list_outputs(self) -> List[TextDefinition]:
        """
        Lists all registered text output definitions.
        """
        return self.definitions

    def search(self, query: str) -> List[TextDefinition]:
        """
        Searches output definitions by query term.
        """
        query_lower = query.lower()
        return [
            d for d in self.definitions
            if query_lower in d.name.lower() or query_lower in d.description.lower() or query_lower in d.purpose.lower()
        ]


class TemplateValidationError(Exception):
    """
    Raised when one or more template configuration errors are detected during static validation.
    """
    def __init__(self, errors: List[str]):
        self.errors = errors
        bullet_list = "\n  - ".join(errors)
        super().__init__(f"Template validation failed with {len(errors)} error(s):\n  - {bullet_list}")


class TemplateValidator:
    """
    Static validator that checks TextTemplate configurations for common configuration errors.

    Checks Performed
    ----------------
    1. Empty template (no section builders defined)
    2. Duplicate section builders within a single template
    3. Duplicate target_column names across different templates
    4. Template name collisions across different templates
    """

    @staticmethod
    def validate(templates: List["TextTemplate"]) -> None:
        """
        Validates a list of templates. Raises TemplateValidationError if any issues are found.

        Parameters
        ----------
        templates : List[TextTemplate]
            The ordered list of templates to validate.

        Raises
        ------
        TemplateValidationError
            If one or more validation errors are found.
        """
        errors: List[str] = []

        seen_columns: Dict[str, str] = {}   # target_column -> template_name
        seen_names: Dict[str, int] = {}     # template_name -> count

        for temp in templates:
            # Check 1: Empty template
            if not temp.sections:
                errors.append(f"Template '{temp.name}' has no section builders defined.")

            # Check 2: Duplicate builders within a single template
            builder_names_seen = {}
            for sec in temp.sections:
                if sec.name in builder_names_seen:
                    errors.append(
                        f"Template '{temp.name}': duplicate section builder '{sec.name}' detected."
                    )
                builder_names_seen[sec.name] = True

            # Check 3: Duplicate target_column across templates
            if temp.target_column in seen_columns:
                errors.append(
                    f"Duplicate target_column '{temp.target_column}' used by both "
                    f"'{seen_columns[temp.target_column]}' and '{temp.name}'."
                )
            else:
                seen_columns[temp.target_column] = temp.name

            # Check 4: Duplicate template names
            seen_names[temp.name] = seen_names.get(temp.name, 0) + 1

        for name, count in seen_names.items():
            if count > 1:
                errors.append(f"Duplicate template name '{name}' found {count} times.")

        if errors:
            raise TemplateValidationError(errors)


class HeaderSectionBuilder(BaseSectionBuilder):
    """
    Constructs the menu item title, item type, cuisine type, course types, and meal types header text block.
    """

    required_columns: List[str] = ["recipe_name"]
    optional_columns: List[str] = ["item_type", "cuisine_type", "meal_type", "dish_type"]

    @property
    def description(self) -> str:
        return "Constructs the menu item title, item type, cuisine style, course types, and meal types header."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            if "recipe_name" in r and is_present(r["recipe_name"]):
                fields[L10nManager.get("recipe_name", self.language)] = r["recipe_name"]
            if "item_type" in r and is_present(r["item_type"]):
                fields[L10nManager.get("item_type", self.language)] = str(r["item_type"]).replace("_", " ").title()
            if "cuisine_type" in r and is_present(r["cuisine_type"]):
                cuisines = ", ".join(r["cuisine_type"]) if isinstance(r["cuisine_type"], list) else str(r["cuisine_type"])
                fields[L10nManager.get("cuisine_type", self.language)] = cuisines
            if "meal_type" in r and is_present(r["meal_type"]):
                meals = ", ".join(r["meal_type"]) if isinstance(r["meal_type"], list) else str(r["meal_type"])
                fields[L10nManager.get("meal_type", self.language)] = meals
            if "dish_type" in r and is_present(r["dish_type"]):
                dishes = ", ".join(r["dish_type"]) if isinstance(r["dish_type"], list) else str(r["dish_type"])
                fields[L10nManager.get("dish_type", self.language)] = dishes
            
            if fields:
                return SectionContent(title=None, fields=fields)
            return None

        return df.apply(format_row, axis=1)


class DescriptionSectionBuilder(BaseSectionBuilder):
    """
    Stitches recipe description text, prep notes, and serving volume.
    """

    required_columns: List[str] = []
    optional_columns: List[str] = ["description", "prep_notes", "servings"]

    @property
    def description(self) -> str:
        return "Stitches description text, prep notes, and serving volume."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            if "description" in r and is_present(r["description"]):
                fields[L10nManager.get("description", self.language)] = r["description"]
            if "prep_notes" in r and is_present(r["prep_notes"]):
                fields[L10nManager.get("prep_notes", self.language)] = r["prep_notes"]
            if "servings" in r and is_present(r["servings"]):
                fields[L10nManager.get("servings", self.language)] = f"{r['servings']} {L10nManager.get('servings_unit', self.language)}"
            
            if fields:
                return SectionContent(title=None, fields=fields)
            return None

        return df.apply(format_row, axis=1)


class NutritionSectionBuilder(BaseSectionBuilder):
    """
    Constructs calories, macronutrients, and nutrition ratios text block.
    """

    required_columns: List[str] = ["calories_per_serving"]
    optional_columns: List[str] = [
        "protein_g_per_serving", "carbs_g_per_serving", "fat_g_per_serving",
        "fiber_g_per_serving", "sugar_g_per_serving", "sodium_mg_per_serving",
        "nutrition_score", "protein_calorie_ratio", "protein_density"
    ]

    @property
    def description(self) -> str:
        return "Constructs calories, macronutrients, and nutrition ratios text block."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            bullet_keys = []
            
            lbl_calories = L10nManager.get("calories", self.language)
            lbl_protein = L10nManager.get("protein", self.language)
            lbl_carbs = L10nManager.get("carbs", self.language)
            lbl_fat = L10nManager.get("fat", self.language)
            lbl_fiber = L10nManager.get("fiber", self.language)
            lbl_sugar = L10nManager.get("sugar", self.language)
            lbl_sodium = L10nManager.get("sodium", self.language)
            lbl_score = L10nManager.get("nutrition_score", self.language)
            lbl_ratio = L10nManager.get("protein_calorie_ratio", self.language)
            lbl_density = L10nManager.get("protein_density", self.language)

            if "calories_per_serving" in r and is_present(r["calories_per_serving"]):
                fields[lbl_calories] = f"{r['calories_per_serving']} kcal"
                bullet_keys.append(lbl_calories)
            if "protein_g_per_serving" in r and is_present(r["protein_g_per_serving"]):
                fields[lbl_protein] = f"{r['protein_g_per_serving']}g"
                bullet_keys.append(lbl_protein)
            if "carbs_g_per_serving" in r and is_present(r["carbs_g_per_serving"]):
                net_carbs_str = ""
                if "fiber_g_per_serving" in r and is_present(r["fiber_g_per_serving"]):
                    net_carbs = max(0.0, r["carbs_g_per_serving"] - r["fiber_g_per_serving"])
                    net_carbs_str = f" (Net Carbs: {net_carbs:.1f}g)"
                fields[lbl_carbs] = f"{r['carbs_g_per_serving']}g{net_carbs_str}"
                bullet_keys.append(lbl_carbs)
            if "fat_g_per_serving" in r and is_present(r["fat_g_per_serving"]):
                fields[lbl_fat] = f"{r['fat_g_per_serving']}g"
                bullet_keys.append(lbl_fat)
            if "fiber_g_per_serving" in r and is_present(r["fiber_g_per_serving"]):
                fields[lbl_fiber] = f"{r['fiber_g_per_serving']}g"
                bullet_keys.append(lbl_fiber)
            if "sugar_g_per_serving" in r and is_present(r["sugar_g_per_serving"]):
                fields[lbl_sugar] = f"{r['sugar_g_per_serving']}g"
                bullet_keys.append(lbl_sugar)
            if "sodium_mg_per_serving" in r and is_present(r["sodium_mg_per_serving"]):
                fields[lbl_sodium] = f"{r['sodium_mg_per_serving']}mg"
                bullet_keys.append(lbl_sodium)
            if "nutrition_score" in r and is_present(r["nutrition_score"]):
                fields[lbl_score] = f"{r['nutrition_score']:.2f}"
                bullet_keys.append(lbl_score)
            if "protein_calorie_ratio" in r and is_present(r["protein_calorie_ratio"]):
                fields[lbl_ratio] = f"{r['protein_calorie_ratio']:.2f} {L10nManager.get('g_per_100_kcal', self.language)}"
                bullet_keys.append(lbl_ratio)
            if "protein_density" in r and is_present(r["protein_density"]):
                fields[lbl_density] = f"{r['protein_density']:.4f}"
                bullet_keys.append(lbl_density)

            if fields:
                return SectionContent(
                    title=L10nManager.get("nutrition_facts", self.language),
                    fields=fields,
                    bullet_keys=bullet_keys
                )
            return None

        return df.apply(format_row, axis=1)


class DietarySectionBuilder(BaseSectionBuilder):
    """
    Constructs dietary course classifications, health labels, and binary diet properties.
    """

    required_columns: List[str] = []
    optional_columns: List[str] = [
        "diet_labels", "health_labels", "is_vegan", "is_vegetarian",
        "is_gluten_free", "is_dairy_free", "is_keto", "is_halal",
        "is_low_calorie", "is_high_protein", "is_low_carb", "is_low_fat",
        "is_low_sodium", "is_high_fiber"
    ]

    @property
    def description(self) -> str:
        return "Constructs dietary course classifications, health labels, and binary diet properties."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            bullet_keys = []
            
            lbl_diet = L10nManager.get("diet_labels", self.language)
            lbl_health = L10nManager.get("health_labels", self.language)
            lbl_attrs = L10nManager.get("dietary_attributes", self.language)

            if "diet_labels" in r and is_present(r["diet_labels"]):
                labels = ", ".join(r["diet_labels"]) if isinstance(r["diet_labels"], list) else str(r["diet_labels"])
                fields[lbl_diet] = labels
                bullet_keys.append(lbl_diet)
            if "health_labels" in r and is_present(r["health_labels"]):
                labels = ", ".join(r["health_labels"]) if isinstance(r["health_labels"], list) else str(r["health_labels"])
                fields[lbl_health] = labels
                bullet_keys.append(lbl_health)

            flags = []
            flag_mappings = {
                "is_vegan": "vegan",
                "is_vegetarian": "vegetarian",
                "is_gluten_free": "gluten_free",
                "is_dairy_free": "dairy_free",
                "is_keto": "keto",
                "is_halal": "halal",
                "is_low_calorie": "low_calorie",
                "is_high_protein": "high_protein",
                "is_low_carb": "low_carb",
                "is_low_fat": "low_fat",
                "is_low_sodium": "low_sodium",
                "is_high_fiber": "high_fiber"
            }
            for col, key in flag_mappings.items():
                if col in r and r[col] is True:
                      flags.append(L10nManager.get(key, self.language))
                      
            if flags:
                fields[lbl_attrs] = ", ".join(flags)
                bullet_keys.append(lbl_attrs)

            if fields:
                return SectionContent(
                    title=L10nManager.get("dietary_profile", self.language),
                    fields=fields,
                    bullet_keys=bullet_keys
                )
            return None

        return df.apply(format_row, axis=1)


class IngredientsSectionBuilder(BaseSectionBuilder):
    """
    Constructs ingredient lists and total counts.
    """

    required_columns: List[str] = ["ingredients"]
    optional_columns: List[str] = ["ingredient_count"]

    @property
    def description(self) -> str:
        return "Constructs ingredient list and total item count."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            if "ingredients" in r and is_present(r["ingredients"]):
                ing_list = r["ingredients"]
                fields["List"] = ing_list if isinstance(ing_list, list) else [ing_list]
            if "ingredient_count" in r and is_present(r["ingredient_count"]):
                fields[L10nManager.get("total_ingredients", self.language)] = r["ingredient_count"]
            
            if fields:
                return SectionContent(
                    title=L10nManager.get("ingredients", self.language),
                    fields=fields
                )
            return None

        return df.apply(format_row, axis=1)


class MetadataSectionBuilder(BaseSectionBuilder):
    """
    Constructs pricing levels, ratings, source details, preparation times, and popularity metrics.
    """

    required_columns: List[str] = []
    optional_columns: List[str] = ["price", "rating", "source", "prep_time_minutes", "popularity"]

    @property
    def description(self) -> str:
        return "Constructs pricing levels, ratings, source details, prep durations, and popularity scoring."

    def build_section(self, df: pd.DataFrame) -> pd.Series:
        def format_row(r):
            fields = {}
            bullet_keys = []
            
            lbl_price = L10nManager.get("price", self.language)
            lbl_rating = L10nManager.get("rating", self.language)
            lbl_source = L10nManager.get("source", self.language)
            lbl_prep = L10nManager.get("prep_time", self.language)
            lbl_popularity = L10nManager.get("popularity", self.language)

            if "price" in r and is_present(r["price"]):
                price_val = r["price"]
                level_info = []
                if "price_level" in r and is_present(r["price_level"]):
                    level_info.append(f"Price level: {r['price_level']}")
                if "budget_category" in r and is_present(r["budget_category"]):
                    level_info.append(f"Category: {r['budget_category']}")
                
                lvl_str = f" ({', '.join(level_info)})" if level_info else ""
                fields[lbl_price] = f"${price_val:.2f}{lvl_str}"
                bullet_keys.append(lbl_price)
            
            if "rating" in r and is_present(r["rating"]):
                fields[lbl_rating] = f"{r['rating']} / 5.0"
                bullet_keys.append(lbl_rating)
            if "source" in r and is_present(r["source"]):
                fields[lbl_source] = r["source"]
                bullet_keys.append(lbl_source)
            if "prep_time_minutes" in r and is_present(r["prep_time_minutes"]):
                fields[lbl_prep] = f"{r['prep_time_minutes']} {L10nManager.get('minutes', self.language)}"
                bullet_keys.append(lbl_prep)
            if "popularity" in r and is_present(r["popularity"]):
                pop = r["popularity"]
                pop_str = f"{pop:.0%}" if pop <= 1.0 else f"{pop}%"
                fields[lbl_popularity] = pop_str
                bullet_keys.append(lbl_popularity)

            if fields:
                return SectionContent(
                    title=L10nManager.get("metadata", self.language),
                    fields=fields,
                    bullet_keys=bullet_keys
                )
            return None

        return df.apply(format_row, axis=1)


class TextBuilder:
    """
    Pipeline Coordinator for generating structured textual chunks from Standardized DataFrames.
    """

    def __init__(self, templates: Optional[List[TextTemplate]] = None, language: str = "en"):
        """
        Initializes the TextBuilder.

        Parameters
        ----------
        templates : Optional[List[TextTemplate]]
            Custom list of text templates. If None, default embedding, LLM, and summary templates are configured.
        language : str
            The translation catalog locale string to resolve labels (e.g. 'en', 'ar', 'fr').
        """
        self.language = language
        if templates is not None:
            self.templates = templates
        else:
            # Configure default section builders
            header = HeaderSectionBuilder(language)
            desc = DescriptionSectionBuilder(language)
            nutr = NutritionSectionBuilder(language)
            diet = DietarySectionBuilder(language)
            ingr = IngredientsSectionBuilder(language)
            meta = MetadataSectionBuilder(language)

            self.templates = [
                TextTemplate(
                    name="Embedding",
                    purpose="semantic search",
                    target_column="search_text",
                    sections=[header, nutr, diet, ingr],
                    description="Concise representation optimized for retrieval embedding models."
                ),
                TextTemplate(
                    name="LLM",
                    purpose="context window feeding",
                    target_column="text_chunk",
                    sections=[header, desc, nutr, ingr, meta],
                    description="Detailed natural language representation optimized for LLM comprehension."
                ),
                TextTemplate(
                    name="Summary",
                    purpose="UI previews",
                    target_column="summary_text",
                    sections=[header, desc],
                    description="Short preview summaries for UI display."
                )
            ]

        # Static validation — raises TemplateValidationError on misconfiguration
        TemplateValidator.validate(self.templates)
        self.registry = TextRegistry(self.templates)
        self.last_result: Optional[TextBuildingResult] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the text building pipeline, caching section builds, and appending generated columns to the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input standardized DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame enriched with textual columns.
        """
        start_time = time.time()
        result = TextBuildingResult()
        self.last_result = result
        
        # Avoid mutating the original DataFrame in place
        df_enriched = df.copy()

        # Identify all unique section builders needed across all templates
        unique_builders: Dict[str, BaseSectionBuilder] = {}
        for temp in self.templates:
            for sec in temp.sections:
                unique_builders[sec.name] = sec
                # Update language attribute dynamically
                sec.language = self.language

        section_cache: Dict[str, pd.Series] = {}

        # 1. Execute and Cache Section Builders
        for name, builder in unique_builders.items():
            sec_start = time.time()
            
            # Conditional section: skip builder if required columns are absent
            if not builder.can_build(df_enriched):
                missing = [col for col in builder.required_columns if col not in df_enriched.columns]
                result.warnings.append(
                    TextWarning(
                        builder_name=name,
                        severity="SKIP",
                        message=f"Section '{name}' skipped — missing required columns: {missing}"
                    )
                )
                result.execution_metadata[name] = BuilderMetadata(
                    name=name,
                    description=f"SKIPPED: missing {missing}",
                    execution_time_seconds=0.0,
                    sections_generated=[]
                )
                continue

            try:
                sec_series = builder.build_section(df_enriched)
                section_cache[name] = sec_series
                
                duration = time.time() - sec_start
                result.execution_metadata[name] = BuilderMetadata(
                    name=name,
                    description=builder.description,
                    execution_time_seconds=duration,
                    sections_generated=[name]
                )
            except Exception as e:
                duration = time.time() - sec_start
                result.is_success = False
                result.warnings.append(
                    TextWarning(
                        builder_name=name,
                        severity="ERROR",
                        message=f"Failed to generate section '{name}': {str(e)}"
                    )
                )
                result.execution_metadata[name] = BuilderMetadata(
                    name=name,
                    description=f"FAILED: {str(e)}",
                    execution_time_seconds=duration,
                    sections_generated=[]
                )

        # 2. Render and assign templates
        for temp in self.templates:
            try:
                rendered_series = temp.render(df_enriched, section_cache)
                df_enriched[temp.target_column] = rendered_series
                result.generated_columns.append(temp.target_column)
            except Exception as e:
                result.is_success = False
                result.warnings.append(
                    TextWarning(
                        builder_name=temp.name,
                        severity="ERROR",
                        message=f"Failed to render template '{temp.name}' in column '{temp.target_column}': {str(e)}"
                    )
                )

        # 3. Calculate statistics
        total_time = time.time() - start_time
        result.statistics["total_columns_generated"] = len(result.generated_columns)
        result.statistics["total_execution_time_seconds"] = total_time
        
        active_times = [m.execution_time_seconds for m in result.execution_metadata.values() if "FAILED" not in m.description]
        result.statistics["average_builder_time_seconds"] = sum(active_times) / len(active_times) if active_times else 0.0

        return df_enriched

    def generate_report(self) -> None:
        """
        Prints an enterprise-style Text Building Report summarizing execution diagnostics.
        """
        if self.last_result is None:
            print("No text building execution has occurred yet.")
            return

        result = self.last_result
        print("=" * 60)
        print(f"{'Text Builder Report':^60}")
        print("=" * 60)

        for name, meta in result.execution_metadata.items():
            status = "PASS" if "FAILED" not in meta.description else "FAILED"
            print(f"{name}")
            print(f"{status}")
            if status == "PASS":
                print(f"Generated: {', '.join(meta.sections_generated)}")
            else:
                print(f"Error: {meta.description}")
            print("-" * 60)

        print("=" * 60)
        print("Generated Columns:")
        for col in result.generated_columns:
            print(f"  - {col}")
        print("-" * 60)
        print(f"Execution Time           : {result.statistics.get('total_execution_time_seconds', 0.0):.4f} seconds")
        print(f"Average Section Builder  : {result.statistics.get('average_builder_time_seconds', 0.0):.4f} seconds")
        
        if result.warnings:
            print("-" * 60)
            print("Runtime Warnings Log:")
            for warn in result.warnings:
                print(f"  [{warn.severity}] {warn.builder_name}: {warn.message}")
        print("=" * 60)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Executing self-test for TextBuilder pipeline...\n")

    # Sample DataFrame with valid columns for full transformation
    df_sample = pd.DataFrame([
        {
            "recipe_name": "Cheesy Truffle Risotto",
            "cuisine_type": ["Italian"],
            "meal_type": ["lunch", "dinner"],
            "dish_type": ["main course"],
            "description": "Creamy risotto with cheese and black truffle oil.",
            "prep_notes": "Stir constantly while adding warm broth.",
            "servings": 4,
            "calories_per_serving": 400.0,
            "protein_g_per_serving": 12.0,
            "carbs_g_per_serving": 50.0,
            "fat_g_per_serving": 15.0,
            "fiber_g_per_serving": 3.0,
            "sodium_mg_per_serving": 120.0,
            "sugar_g_per_serving": 2.0,
            "nutrition_score": 12.5,
            "protein_calorie_ratio": 3.0,
            "protein_density": 0.03,
            "health_labels": ["Vegetarian", "Gluten-Free"],
            "diet_labels": ["Low-Sodium"],
            "ingredients": ["Arborio Rice", "Parmesan Cheese", "Truffle Oil", "Vegetable Broth"],
            "ingredient_count": 4,
            "is_vegan": False,
            "is_vegetarian": True,
            "is_gluten_free": True,
            "is_dairy_free": False,
            "is_keto": False,
            "is_halal": True,
            "is_low_calorie": False,
            "is_high_protein": False,
            "is_low_carb": False,
            "is_low_fat": False,
            "is_low_sodium": True,
            "is_high_fiber": False,
            "price": 24.50,
            "price_level": "$$",
            "budget_category": "mid-range",
            "rating": 4.8,
            "source": "Gourmet Kitchen",
            "prep_time_minutes": 45,
            "popularity": 0.92,
        }
    ])

    # 1. Test Default Text Building
    print("--- 1. Testing Default TextBuilder Run ---")
    builder = TextBuilder()
    df_out = builder.transform(df_sample)
    
    print("\n[Generated 'search_text']:")
    print(df_out["search_text"].iloc[0])
    print("\n" + "-"*40)
    
    print("\n[Generated 'text_chunk']:")
    print(df_out["text_chunk"].iloc[0])
    print("\n" + "-"*40)

    print("\n[Generated 'summary_text']:")
    print(df_out["summary_text"].iloc[0])
    print("\n" + "-"*40)

    builder.generate_report()

    # 2. Test Custom Section Builder and Custom Template (Extensibility Demonstration)
    print("\n--- 2. Testing Custom Extension (OCP Alignment) ---")
    
    class CustomFootnoteSectionBuilder(BaseSectionBuilder):
        @property
        def description(self) -> str:
            return "Appends a developer-specified custom footnote statement."

        def build_section(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series([
                SectionContent(title=None, fields={"Footnote": "Built automatically by DineAI RAG ETL pipeline."})
                for _ in df.index
            ], index=df.index)

    # Register custom template using the new custom section builder and existing ones
    footnote_builder = CustomFootnoteSectionBuilder()
    custom_temp = TextTemplate(
        name="CustomAudit",
        purpose="audit logging",
        target_column="audit_chunk",
        sections=[HeaderSectionBuilder(), footnote_builder],
        description="Audit representation displaying titles and pipeline footnotes."
    )

    custom_coordinator = TextBuilder(templates=[custom_temp])
    df_custom_out = custom_coordinator.transform(df_sample)

    print("\n[Generated 'audit_chunk']:")
    print(df_custom_out["audit_chunk"].iloc[0])
    print("\n" + "-"*40)
    
    custom_coordinator.generate_report()

    # 3. Test Arabic Localization
    print("\n--- 3. Testing Arabic Localization ---")
    ar_builder = TextBuilder(language="ar")
    df_ar_out = ar_builder.transform(df_sample)
    print("\n[Generated Arabic 'summary_text']:")
    print(df_ar_out["summary_text"].iloc[0])
    print("\n" + "-"*40)

    # 4. Test Conditional Section Skipping
    print("\n--- 4. Testing Conditional Section Skipping ---")
    df_sparse = pd.DataFrame([{
        "recipe_name": "Mystery Dish",
        "cuisine_type": ["Unknown"],
        # 'calories_per_serving' intentionally omitted → NutritionSectionBuilder should SKIP
        # 'ingredients' intentionally omitted → IngredientsSectionBuilder should SKIP
    }])
    sparse_builder = TextBuilder()
    df_sparse_out = sparse_builder.transform(df_sparse)
    print("\n[Generated 'search_text' from sparse DataFrame]:")
    print(df_sparse_out["search_text"].iloc[0])
    print("\nSkip Warnings Emitted:")
    for w in sparse_builder.last_result.warnings:
        if w.severity == "SKIP":
            print(f"  [{w.severity}] {w.builder_name}: {w.message}")
    print("\n" + "-"*40)

    # 5. Test Template Validation
    print("\n--- 5. Testing Template Validation ---")
    hdr = HeaderSectionBuilder()
    try:
        # Intentionally invalid: duplicate builder + duplicate target_column
        bad_builder = TextBuilder(templates=[
            TextTemplate("T1", "test", "col_a", [hdr, hdr]),        # duplicate builder
            TextTemplate("T2", "test", "col_a", [MetadataSectionBuilder()]),  # duplicate target_column
        ])
        print("  ERROR: Should have raised TemplateValidationError!")
    except TemplateValidationError as exc:
        print("  [PASS] TemplateValidationError raised as expected:")
        for err in exc.errors:
            print(f"    - {err}")
    print("\n" + "-"*40)

