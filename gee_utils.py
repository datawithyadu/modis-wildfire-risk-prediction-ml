import ee
from config import AREAS

def init_ee():
    """
    Initialize Google Earth Engine
    """
    try:
        ee.Initialize()
        print("Earth Engine initialized")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine authenticated & initialized")


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

def get_fire_risk_prediction(area, year):
    """
    Safe fire risk prediction.
    Trains RF only if both classes exist.
    """

    aoi = ee.Geometry.Rectangle(AREAS[area])

    # Vegetation features
    veg = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select(["NDVI", "EVI"])
        .median()
        .multiply(0.0001)
        .clip(aoi)
    )

    # Burn label
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

    dataset = veg.addBands(burn)

    # Count fire pixels
    fire_count = burn.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=500,
        maxPixels=1e13
    ).get("BURN_LABEL")

    # If no fire pixels → return zero-risk map
    def no_fire_case():
        return ee.Image(0).rename("FIRE_RISK").clip(aoi)

    # Train only if fire pixels exist
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

        rf = ee.Classifier.smileRandomForest(300).train(
            features=samples,
            classProperty="BURN_LABEL",
            inputProperties=["NDVI", "EVI"]
        )

        return veg.classify(rf).rename("FIRE_RISK")

    # Conditional execution
    prediction = ee.Image(
        ee.Algorithms.If(
            ee.Number(fire_count).gt(0),
            train_case(),
            no_fire_case()
        )
    )

    return prediction
