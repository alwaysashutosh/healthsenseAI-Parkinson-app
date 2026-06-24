"""Hospital / neurologist finder using OpenStreetMap (free, no API key).

- Geocode a city name to coordinates via Nominatim.
- Find nearby hospitals & clinics via the Overpass API.
Results are distance-sorted and include name, address, phone and a map link.
"""
import math

import requests
from flask import Blueprint, request, jsonify

from auth import token_required

hospitals_bp = Blueprint("hospitals", __name__, url_prefix="/api")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Nominatim's usage policy requires an identifying User-Agent.
HEADERS = {"User-Agent": "NeuroVoice-Parkinsons-Screening/1.0 (final-year project)"}
TIMEOUT = 25


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return round(r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _geocode(city):
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": city, "format": "json", "limit": 1},
        headers=HEADERS, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", city)


def _overpass_hospitals(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["healthcare"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"](around:{radius_m},{lat},{lon});
    );
    out center 40;
    """
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("elements", [])


def _format_element(el, origin_lat, origin_lon):
    tags = el.get("tags", {})
    lat = el.get("lat") or el.get("center", {}).get("lat")
    lon = el.get("lon") or el.get("center", {}).get("lon")
    if lat is None or lon is None:
        return None
    addr_parts = [
        tags.get("addr:housenumber"), tags.get("addr:street"),
        tags.get("addr:suburb"), tags.get("addr:city"),
    ]
    address = ", ".join(p for p in addr_parts if p) or tags.get("addr:full", "")
    return {
        "name": tags.get("name", "Unnamed facility"),
        "type": tags.get("amenity") or tags.get("healthcare") or "facility",
        "address": address,
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": tags.get("website") or tags.get("contact:website"),
        "lat": lat,
        "lon": lon,
        "distance_km": _haversine_km(origin_lat, origin_lon, lat, lon),
        "map_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}",
    }


@hospitals_bp.get("/hospitals")
@token_required
def hospitals():
    city = (request.args.get("city") or "").strip()
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    radius_km = request.args.get("radius", default=10, type=float)
    radius_m = int(max(1, min(radius_km, 50)) * 1000)

    try:
        if lat is None or lon is None:
            if not city:
                return jsonify({"error": "Provide a 'city' or 'lat'+'lon'"}), 400
            geo = _geocode(city)
            if geo is None:
                return jsonify({"error": f"Could not locate '{city}'"}), 404
            lat, lon, resolved = geo
        else:
            resolved = f"{lat:.4f}, {lon:.4f}"

        elements = _overpass_hospitals(lat, lon, radius_m)
        results = [_format_element(e, lat, lon) for e in elements]
        results = [r for r in results if r and r["name"] != "Unnamed facility"]
        # Prefer named facilities, nearest first.
        results.sort(key=lambda r: r["distance_km"])
        return jsonify({
            "location": {"lat": lat, "lon": lon, "resolved": resolved},
            "radius_km": radius_km,
            "count": len(results[:25]),
            "hospitals": results[:25],
        })
    except requests.Timeout:
        return jsonify({"error": "Map service timed out. Please try again."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Map service error: {e}"}), 502
