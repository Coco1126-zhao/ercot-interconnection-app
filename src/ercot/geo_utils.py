"""
ercot/geo_utils.py
==================
Map ERCOT projects to lat/lon for plotting.

Strategy (per project default):
  1. If a HIFLD substations CSV is present at data/geo/hifld_substations.csv,
     try exact substation-name match against POI Location.
  2. Otherwise fall back to Texas county centroid (US Census 2020 mean
     center of population). The centroid file is downloaded once on
     first call from a stable Census URL and cached.
  3. Add small deterministic jitter so multiple projects in the same
     county don't render on top of each other.
"""
from __future__ import annotations
import io
import hashlib
import requests
import pandas as pd
from pathlib import Path

GEO_DIR        = Path("data/geo")
COUNTY_CACHE   = GEO_DIR / "tx_county_centroids.csv"
HIFLD_SUBS     = GEO_DIR / "hifld_substations.csv"   # optional, user-provided

CENSUS_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/"
    "county/CenPop2020_Mean_CO48.txt"
)

# Approximate ERCOT CDR zone centroids — final fallback when no county info
ZONE_CENTROIDS = {
    "WEST":    (32.30, -101.40),
    "NORTH":   (32.95,  -97.30),
    "SOUTH":   (28.80,  -97.40),
    "HOUSTON": (29.75,  -95.36),
    "FWEST":   (31.50, -102.50),  # Far West
    "PANHANDLE":(35.20,-101.83),
    "ZNRZ":    (31.00, -100.00),  # generic
}


def _ensure_county_centroids() -> pd.DataFrame:
    """Download Texas county centroids once; return cached DataFrame."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    if COUNTY_CACHE.exists():
        return pd.read_csv(COUNTY_CACHE)

    print(f"[geo] downloading TX county centroids from Census ...")
    r = requests.get(CENSUS_URL, timeout=60)
    r.raise_for_status()
    # Census file ships with a UTF-8 BOM; encoding='utf-8-sig' strips it.
    df = pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    # File CO48 is already Texas-only; keep all rows.
    df["county_norm"] = df["COUNAME"].astype(str).str.strip().str.title()
    df = df[["county_norm", "LATITUDE", "LONGITUDE"]].rename(
        columns={"LATITUDE": "lat", "LONGITUDE": "lon"}
    )
    df.to_csv(COUNTY_CACHE, index=False)
    print(f"[geo] cached {len(df)} counties → {COUNTY_CACHE}")
    return df


def _load_hifld() -> pd.DataFrame | None:
    if not HIFLD_SUBS.exists():
        return None
    df = pd.read_csv(HIFLD_SUBS)
    # Expected cols: NAME, LATITUDE, LONGITUDE, STATE
    df = df[df["STATE"].str.upper() == "TX"].copy()
    df["name_norm"] = df["NAME"].str.strip().str.upper()
    return df


def _jitter(seed: str, scale: float = 0.05) -> tuple[float, float]:
    """Deterministic small offset based on a string seed."""
    h = hashlib.md5(seed.encode()).digest()
    dx = ((h[0] / 255.0) - 0.5) * scale
    dy = ((h[1] / 255.0) - 0.5) * scale
    return dx, dy


def attach_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lat / lon columns to an ERCOT projects DataFrame.

    Required input columns: 'County', 'CDR Reporting Zone', 'INR'.
    Optional: 'POI Location' (used if HIFLD CSV present).
    """
    out = df.copy()
    counties = _ensure_county_centroids()
    county_lookup = counties.set_index("county_norm")[["lat", "lon"]].to_dict("index")

    hifld = _load_hifld()
    sub_lookup = (
        hifld.set_index("name_norm")[["LATITUDE", "LONGITUDE"]].to_dict("index")
        if hifld is not None else {}
    )

    lats, lons, srcs = [], [], []
    for _, row in out.iterrows():
        lat = lon = None
        src = None

        # 1. HIFLD substation match
        poi = str(row.get("POI Location", "")).strip().upper()
        if poi and sub_lookup:
            for key in sub_lookup:
                if key and key in poi:
                    lat = sub_lookup[key]["LATITUDE"]
                    lon = sub_lookup[key]["LONGITUDE"]
                    src = "substation"
                    break

        # 2. County centroid
        if lat is None:
            county = str(row.get("County", "")).strip().title()
            if county in county_lookup:
                lat = county_lookup[county]["lat"]
                lon = county_lookup[county]["lon"]
                src = "county"

        # 3. Zone centroid
        if lat is None:
            zone = str(row.get("CDR Reporting Zone", "")).strip().upper()
            if zone in ZONE_CENTROIDS:
                lat, lon = ZONE_CENTROIDS[zone]
                src = "zone"

        # Jitter so co-located projects spread out
        if lat is not None and src in ("county", "zone"):
            dx, dy = _jitter(str(row.get("INR", "")))
            lat += dx
            lon += dy

        lats.append(lat)
        lons.append(lon)
        srcs.append(src)

    out["lat"] = lats
    out["lon"] = lons
    out["geo_source"] = srcs
    return out


if __name__ == "__main__":
    from src.ercot.loader import load_gis_large_gen
    df = load_gis_large_gen("data/ercot/latest.xlsx")
    df = attach_coordinates(df)
    print(df[["INR", "Project Name", "County", "CDR Reporting Zone",
              "lat", "lon", "geo_source"]].head(10).to_string())
    print("\nCoverage:", df["geo_source"].value_counts().to_dict())
