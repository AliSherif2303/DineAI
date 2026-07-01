"""
DineAI — Built-in Capability Definitions
=========================================
All concrete BaseCapability subclasses that ship with DineAI out of the box.

Adding a new capability
-----------------------
1. Create a subclass of BaseCapability below.
2. Add an instance to the ALL_CAPABILITIES list at the bottom.
3. The framework auto-detects and activates it — no other code changes needed.

Future capabilities (examples already planned):
- AllergenFilteringCapability  (requires: cautions)
- StockAvailabilityCapability  (requires: availability)
- LiveInventoryCapability      (requires: in_stock)
- DeliveryTimeCapability       (requires: delivery_time_minutes)
- DiscountCapability           (requires: discount_price or promo_code)
- SeasonalMenuCapability       (requires: seasonal_label or start_date)
- LoyaltyProgramCapability     (requires: loyalty_points)
"""

from __future__ import annotations

from typing import Dict, List

from dine_ai.capabilities.base import BaseCapability


# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class NutritionSearchCapability(BaseCapability):
    """Full nutrition-based search (requires both calories and protein)."""
    @property
    def name(self) -> str: return "Nutrition Search"
    @property
    def required_columns(self) -> List[str]: return ["calories_per_serving", "protein_g_per_serving"]
    @property
    def optional_columns(self) -> List[str]: return ["fat_g_per_serving", "carbs_g_per_serving", "fiber_g_per_serving"]
    @property
    def description(self) -> str: return "Search by full nutrition profile (calories + protein)"
    @property
    def category(self) -> str: return "Nutrition"


class CalorieFilterCapability(BaseCapability):
    """Low-calorie filtering (requires calories column)."""
    @property
    def name(self) -> str: return "Calorie Filtering"
    @property
    def required_columns(self) -> List[str]: return ["calories_per_serving"]
    @property
    def description(self) -> str: return "Filter items by calorie count"
    @property
    def category(self) -> str: return "Nutrition"


class HighProteinCapability(BaseCapability):
    """High-protein recommendations."""
    @property
    def name(self) -> str: return "High Protein"
    @property
    def required_columns(self) -> List[str]: return ["protein_g_per_serving"]
    @property
    def description(self) -> str: return "Recommend high-protein items"
    @property
    def category(self) -> str: return "Nutrition"


class LowCalorieCapability(BaseCapability):
    """Low-calorie diet support."""
    @property
    def name(self) -> str: return "Low Calorie"
    @property
    def required_columns(self) -> List[str]: return ["calories_per_serving"]
    @property
    def description(self) -> str: return "Recommend low-calorie items"
    @property
    def category(self) -> str: return "Nutrition"


class LowCarbCapability(BaseCapability):
    """Low-carb diet support."""
    @property
    def name(self) -> str: return "Low Carb"
    @property
    def required_columns(self) -> List[str]: return ["carbs_g_per_serving"]
    @property
    def description(self) -> str: return "Recommend low-carbohydrate items"
    @property
    def category(self) -> str: return "Nutrition"


# ─────────────────────────────────────────────────────────────────────────────
# PRICING CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class BudgetRecommendationCapability(BaseCapability):
    """Price-based budget filtering."""
    @property
    def name(self) -> str: return "Budget Filtering"
    @property
    def required_columns(self) -> List[str]: return ["price"]
    @property
    def description(self) -> str: return "Filter and rank by price"
    @property
    def category(self) -> str: return "Pricing"


# ─────────────────────────────────────────────────────────────────────────────
# DIETARY CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class DietaryLabelCapability(BaseCapability):
    """General dietary label filtering."""
    @property
    def name(self) -> str: return "Dietary Labels"
    @property
    def required_columns(self) -> List[str]: return ["diet_labels"]
    @property
    def description(self) -> str: return "Filter by dietary labels (Low-Carb, High-Protein, etc.)"
    @property
    def category(self) -> str: return "Dietary"


class KetoCapability(BaseCapability):
    """Keto diet recommendation."""
    @property
    def name(self) -> str: return "Keto Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["diet_labels"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"diet_labels": ["keto"]}
    @property
    def description(self) -> str: return "Recommend keto-friendly items"
    @property
    def category(self) -> str: return "Dietary"


class VeganCapability(BaseCapability):
    """Vegan recommendation."""
    @property
    def name(self) -> str: return "Vegan Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["health_labels"]
    @property
    def optional_columns(self) -> List[str]: return ["diet_labels"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"health_labels": ["vegan"]}
    @property
    def description(self) -> str: return "Recommend vegan items"
    @property
    def category(self) -> str: return "Dietary"


class AllergenCapability(BaseCapability):
    """Allergen and caution filtering."""
    @property
    def name(self) -> str: return "Allergen Filtering"
    @property
    def required_columns(self) -> List[str]: return ["cautions"]
    @property
    def description(self) -> str: return "Filter items by allergen warnings"
    @property
    def category(self) -> str: return "Dietary"


# ─────────────────────────────────────────────────────────────────────────────
# RATING & POPULARITY CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class RatingCapability(BaseCapability):
    """Rating-based ranking."""
    @property
    def name(self) -> str: return "Rating Ranking"
    @property
    def required_columns(self) -> List[str]: return ["rating"]
    @property
    def description(self) -> str: return "Rank items by customer rating"
    @property
    def category(self) -> str: return "Rating & Popularity"


class PopularityCapability(BaseCapability):
    """Popularity-based ranking."""
    @property
    def name(self) -> str: return "Popularity Ranking"
    @property
    def required_columns(self) -> List[str]: return ["popularity"]
    @property
    def description(self) -> str: return "Rank items by popularity score"
    @property
    def category(self) -> str: return "Rating & Popularity"


# ─────────────────────────────────────────────────────────────────────────────
# MENU CONTENT CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class IngredientsCapability(BaseCapability):
    """Ingredient-based search."""
    @property
    def name(self) -> str: return "Ingredient Search"
    @property
    def required_columns(self) -> List[str]: return ["ingredients"]
    @property
    def description(self) -> str: return "Search and filter by specific ingredients"
    @property
    def category(self) -> str: return "Menu Content"


class MealTypeCapability(BaseCapability):
    """Meal type filtering (Breakfast, Lunch, Dinner)."""
    @property
    def name(self) -> str: return "Meal Type Filtering"
    @property
    def required_columns(self) -> List[str]: return ["meal_type"]
    @property
    def description(self) -> str: return "Filter by meal occasion (Breakfast, Lunch, Dinner)"
    @property
    def category(self) -> str: return "Menu Content"


class CuisineCapability(BaseCapability):
    """Cuisine type filtering."""
    @property
    def name(self) -> str: return "Cuisine Filtering"
    @property
    def required_columns(self) -> List[str]: return ["cuisine_type"]
    @property
    def description(self) -> str: return "Filter by cuisine type (Italian, American, etc.)"
    @property
    def category(self) -> str: return "Menu Content"


class PrepTimeCapability(BaseCapability):
    """Preparation time filtering."""
    @property
    def name(self) -> str: return "Prep Time Filtering"
    @property
    def required_columns(self) -> List[str]: return ["preparation_time_minutes"]
    @property
    def description(self) -> str: return "Filter by estimated preparation time"
    @property
    def category(self) -> str: return "Menu Content"


class AvailabilityCapability(BaseCapability):
    """Real-time stock/availability filtering."""
    @property
    def name(self) -> str: return "Availability Filtering"
    @property
    def required_columns(self) -> List[str]: return ["availability"]
    @property
    def description(self) -> str: return "Filter out unavailable items"
    @property
    def category(self) -> str: return "Menu Content"


# ─────────────────────────────────────────────────────────────────────────────
# ITEM TYPE CAPABILITIES (require item_type column with specific values)
# ─────────────────────────────────────────────────────────────────────────────

class DrinkCapability(BaseCapability):
    """Drink-specific recommendation (includes coffee, smoothie, juice)."""
    @property
    def name(self) -> str: return "Drink Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]:
        return {"item_type": ["drink", "coffee", "smoothie", "juice"]}
    @property
    def description(self) -> str: return "Recommend beverages (drinks, coffees, smoothies, juices)"
    @property
    def category(self) -> str: return "Item Type"


class DessertCapability(BaseCapability):
    """Dessert recommendation."""
    @property
    def name(self) -> str: return "Dessert Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"item_type": ["dessert"]}
    @property
    def description(self) -> str: return "Recommend dessert items"
    @property
    def category(self) -> str: return "Item Type"


class AppetizerCapability(BaseCapability):
    """Appetizer recommendation."""
    @property
    def name(self) -> str: return "Appetizer Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"item_type": ["appetizer"]}
    @property
    def description(self) -> str: return "Recommend starters and appetizers"
    @property
    def category(self) -> str: return "Item Type"


class SoupCapability(BaseCapability):
    """Soup recommendation."""
    @property
    def name(self) -> str: return "Soup Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"item_type": ["soup"]}
    @property
    def description(self) -> str: return "Recommend soups and broths"
    @property
    def category(self) -> str: return "Item Type"


class SaladCapability(BaseCapability):
    """Salad recommendation."""
    @property
    def name(self) -> str: return "Salad Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"item_type": ["salad"]}
    @property
    def description(self) -> str: return "Recommend fresh salads"
    @property
    def category(self) -> str: return "Item Type"


class MainCourseCapability(BaseCapability):
    """Main course recommendation."""
    @property
    def name(self) -> str: return "Main Course Recommendation"
    @property
    def required_columns(self) -> List[str]: return ["item_type"]
    @property
    def required_values(self) -> Dict[str, List[str]]: return {"item_type": ["main_course"]}
    @property
    def description(self) -> str: return "Recommend main course dishes"
    @property
    def category(self) -> str: return "Item Type"


# ─────────────────────────────────────────────────────────────────────────────
# MEDIA CAPABILITIES
# ─────────────────────────────────────────────────────────────────────────────

class ImageCapability(BaseCapability):
    """Item image display."""
    @property
    def name(self) -> str: return "Item Images"
    @property
    def required_columns(self) -> List[str]: return ["image_url"]
    @property
    def description(self) -> str: return "Display item images in recommendations"
    @property
    def category(self) -> str: return "Media"


# ─────────────────────────────────────────────────────────────────────────────
# ALL_CAPABILITIES — master list for DEFAULT_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Add new capabilities here; the framework activates them automatically.

ALL_CAPABILITIES = [
    # Nutrition
    NutritionSearchCapability(),
    CalorieFilterCapability(),
    HighProteinCapability(),
    LowCalorieCapability(),
    LowCarbCapability(),
    # Pricing
    BudgetRecommendationCapability(),
    # Dietary
    DietaryLabelCapability(),
    KetoCapability(),
    VeganCapability(),
    AllergenCapability(),
    # Rating & Popularity
    RatingCapability(),
    PopularityCapability(),
    # Menu Content
    IngredientsCapability(),
    MealTypeCapability(),
    CuisineCapability(),
    PrepTimeCapability(),
    AvailabilityCapability(),
    # Item Types
    DrinkCapability(),
    DessertCapability(),
    AppetizerCapability(),
    SoupCapability(),
    SaladCapability(),
    MainCourseCapability(),
    # Media
    ImageCapability(),
]
