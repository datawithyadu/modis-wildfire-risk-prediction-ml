import ee
from config import AREAS


# --------------------------------------------------
# Earth Engine initialization (SAFE for Cloud)
# --------------------------------------------------
def init_ee():
    """
    Initialize Google Earth Engine safely.

    - Local machine:
        ee.Initialize() works because you already authenticated.
    - Streamlit Community Cloud:
        ee.Initialize() fails → returns False
        (NO authentication attempt, NO crash)

    Returns:
        True  → Earth Engine ready
        False → Earth Engine not available (demo mode)
    """
    try:
        ee.Initialize()
        return True
    except Exception:
        # 🚫 NEVER call ee.Authenticate() on Streamlit Cloud
        return False


# --------------------------------------------------
# Vegetation features (NDVI, EVI)
# --------------------------------------------------
def get_vegetation_image(area, year):
    aoi = ee.Geometry.Rectangle(AREAS[area])

    img = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select(["NDVI", "EVI"])
        .median()
        .multiply(0.0001)
        .clip(aoi)
    )

    return img


# --------------------------------------------------
# Burned area label (MODIS)
# --------------------------------------------------
def get_burn_label(area, year):
    aoi = ee.Geometry.Rectangle(AREAS[area])

    burn = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select("BurnDate")
        .max()
        .unmask(0)
        .gt(0)
        .rename("BURN_LABEL")
        .clip(aoi)
    )

    return burn


# --------------------------------------------------
# Fire risk prediction (SAFE – handles no-fire cases)
# --------------------------------------------------
def get_fire_risk_prediction(area, year):
    """
    Train a Random Forest inside Earth Engine
    only if fire pixels exist.

    If no fire pixels exist → returns a zero-risk image.
    """

    aoi = ee.Geometry.Rectangle(AREAS[area])

    # Vegetation features
    veg = get_vegetation_image(area, year)

    # Burn label
    burn = get_burn_label(area, year)

    dataset = veg.addBands(burn)

    # Count fire pixels
    fire_count = burn.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=500,
        maxPixels=1e13
    ).get("BURN_LABEL")

    # No-fire case → safe output
    def no_fire_case():
        return ee.Image(0).rename("FIRE_RISK").clip(aoi)

    # Train RF only if fire pixels exist
    def train_case():
        fire_samples = dataset.updateMask(burn).sample(
            region=aoi,
            scale=500,
            numPixels=1500,
            seed=42
        )

        nofire_samples = dataset.updateMask(burn.Not()).sample(
            region=aoi,
            scale=500,
            numPixels=1500,
            seed=42
        )

        samples = fire_samples.merge(nofire_samples)

        rf = ee.Classifier.smileRandomForest(
            numberOfTrees=300
        ).train(
            features=samples,
            classProperty="BURN_LABEL",
            inputProperties=["NDVI", "EVI"]
        )

        return veg.classify(rf).rename("FIRE_RISK")

    prediction = ee.Image(
        ee.Algorithms.If(
            ee.Number(fire_count).gt(0),
            train_case(),
            no_fire_case()
        )
    )

    return prediction
