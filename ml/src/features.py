"""
    Feature configuration for RealEstateIQ

    we wont use all 79 raw ames columns instead 
    we will curate ~18 freatures that are both 
    high-signal for price prediction and reasonable
    to ask user. for a web form

"""

RAW_TO_FRIENDLY = {
    "BedroomAbvGr": "bedrooms",
    "FullBath": "full_bathrooms",
    "HalfBath": "half_bathrooms",
    "GrLivArea": "living_area_sqft",
    "LotArea": "lot_size_sqft",
    "TotalBsmtSF": "basement_sqft",
    "GarageCars": "garage_cars",
    "YearBuilt": "year_built",
    "YrSold": "year_sold",
    "OverallQual": "overall_quality",
    "OverallCond": "overall_condition",
    "Fireplaces": "fireplaces",
    "TotRmsAbvGrd": "total_rooms",
    "Neighborhood": "location",
    "HouseStyle": "house_style",
    "KitchenQual": "kitchen_quality",
    "CentralAir": "central_air",
    "SaleCondition": "sale_condition",
}

NUMERIC_FEATURES = [
    "bedrooms", "full_bathrooms", "half_bathrooms", "living_area_sqft",
    "lot_size_sqft", "basement_sqft", "garage_cars", "house_age",
    "overall_quality", "overall_condition", "fireplaces", "total_rooms",
]

CATEGORICAL_FEATURES = ["location", "house_style", "kitchen_quality", "central_air"]

TARGET = "SalePrice"
ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def engineer_features(df):
    """
        house_age didn't exist in the raw data - it's derived
        from two columns that did. This is one piece of feature engineering
        beyond straight renaming
    """
    df = df.copy()
    df["house_age"] = df["year_sold"] - df["year_built"]
    df["house_age"] = df["house_age"].clip(lower=0)
    return df