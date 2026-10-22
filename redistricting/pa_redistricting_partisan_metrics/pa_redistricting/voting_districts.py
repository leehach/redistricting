"""Pennsylvania voting-district (VTD) shapes from the 2020 Census TIGER/Line files.

    from pa_redistricting.voting_districts import load_vtds
    vtds = load_vtds()     # GeoDataFrame, one row per voting district
"""

import io
import logging

import geopandas as gpd
import requests

from .config import VTD_URL

log = logging.getLogger(__name__)

VTD_COLUMNS = {"GEOID20": "GEOID", "NAME20": "VTD_NAME", "COUNTYFP20": "COUNTYFP",
               "ALAND20": "ALAND", "AWATER20": "AWATER"}


def load_vtds() -> gpd.GeoDataFrame:
    """Download the PA voting-district shapes with friendly column names."""
    log.info("Downloading 2020 Census voting districts for PA ...")
    r = requests.get(VTD_URL, timeout=600)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content)).rename(columns=VTD_COLUMNS)
    return gdf[[*VTD_COLUMNS.values(), "geometry"]]
