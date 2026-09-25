"""
청구동(서울 중구) 브이월드 데이터 수집 스크립트
  1) 건물 폴리곤 + 층수  → data/buildings.geojson   (V-World 데이터 API)
  2) 위성 영상 타일      → data/satellite.jpg        (V-World WMTS, 지면 텍스처용 스티칭)

사용법:
  1. https://www.vworld.kr 회원가입 → 오픈API 인증키 발급 (무료, 즉시)
  2. 아래 API_KEY에 붙여넣거나 환경변수 VWORLD_KEY 설정
  3. pip install requests pillow
  4. python fetch_cheonggu.py
"""
import os, json, math, io, sys
import requests

API_KEY = os.environ.get("VWORLD_KEY", "여기에_브이월드_인증키")

# 청구동 일대 bbox (WGS84). 청구역(5·6호선) 중심 약 1.2 x 1.0 km
BBOX = {"min_lon": 127.0065, "min_lat": 37.5555, "max_lon": 127.0205, "max_lat": 37.5650}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. 건물 통합정보 (폴리곤 + 지상층수 GRND_FLR) ──────────────────────
def fetch_buildings():
    """V-World 데이터 API 2.0, GIS건물통합정보(LT_C_SPBD).
    dataset id가 다르면 https://www.vworld.kr/dev/v4dv_2ddataguide2_s001.do 에서
    '건물' 검색 후 교체할 것."""
    url = "https://api.vworld.kr/req/data"
    geomFilter = "BOX({min_lon},{min_lat},{max_lon},{max_lat})".format(**BBOX)
    features, page = [], 1
    while True:
        params = {
            "service": "data", "request": "GetFeature", "data": "LT_C_SPBD",
            "key": API_KEY, "geomFilter": geomFilter, "format": "json",
            "size": "1000", "page": str(page), "crs": "EPSG:4326",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        status = j.get("response", {}).get("status")
        if status == "NOT_FOUND":
            break
        if status != "OK":
            sys.exit(f"[건물 API 오류] {json.dumps(j, ensure_ascii=False)[:500]}")
        fc = j["response"]["result"]["featureCollection"]
        features += fc["features"]
        total = int(j["response"]["record"]["total"])
        print(f"  건물 page {page}: 누적 {len(features)}/{total}")
        if len(features) >= total:
            break
        page += 1

    # 뷰어가 쓰는 필드만 남긴다: 폴리곤 + 층수(grnd_flr) + 건물명
    out = []
    for f in features:
        p = f.get("properties", {})
        floors = p.get("gro_flo_co") or 0   # LT_C_SPBD 지상층수 필드
        try:
            floors = max(1, int(float(floors)))
        except (TypeError, ValueError):
            floors = 1
        out.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {"floors": floors, "name": p.get("buld_nm") or p.get("BULD_NM") or ""},
        })
    path = os.path.join(DATA_DIR, "buildings.geojson")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"type": "FeatureCollection", "features": out}, fp, ensure_ascii=False)
    print(f"[OK] {path} — 건물 {len(out)}동")


# ── 2. 위성 타일 스티칭 (지면 텍스처) ──────────────────────────────────
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y

def fetch_satellite(z=17):
    from PIL import Image
    x0, y1 = lonlat_to_tile(BBOX["min_lon"], BBOX["min_lat"], z)
    x1, y0 = lonlat_to_tile(BBOX["max_lon"], BBOX["max_lat"], z)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    canvas = Image.new("RGB", (W, H))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            u = f"https://api.vworld.kr/req/wmts/1.0.0/{API_KEY}/Satellite/{z}/{ty}/{tx}.jpeg"
            r = requests.get(u, timeout=30)
            if r.ok:
                canvas.paste(Image.open(io.BytesIO(r.content)), ((tx - tx0) * 256, (ty - ty0) * 256))
            else:
                print(f"  타일 실패 {z}/{ty}/{tx}: HTTP {r.status_code}")
    # bbox에 맞춰 크롭
    l = int((x0 - tx0) * 256); r_ = int(W - (tx1 + 1 - x1) * 256)
    t = int((y0 - ty0) * 256); b = int(H - (ty1 + 1 - y1) * 256)
    canvas.crop((l, t, r_, b)).save(os.path.join(DATA_DIR, "satellite.jpg"), quality=90)
    print(f"[OK] data/satellite.jpg — z{z}, {r_-l}x{b-t}px")

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as fp:
        json.dump(BBOX, fp)
    print("[OK] data/meta.json")


# ── 3. 지형 DEM (Terrarium 타일, 키 불필요) ───────────────────────────
def fetch_terrain(z=14, gw=96, gh=80):
    """AWS 공개 elevation-tiles(terrarium) → bbox 표고 그리드 data/terrain.json.
    height(m) = R*256 + G + B/256 - 32768. 청구동은 남산 자락 경사지라 필수."""
    from PIL import Image
    tiles = {}
    def height_at(lon, lat):
        x, y = lonlat_to_tile(lon, lat, z)
        tx, ty = int(x), int(y)
        if (tx, ty) not in tiles:
            u = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png"
            r = requests.get(u, timeout=30); r.raise_for_status()
            tiles[(tx, ty)] = Image.open(io.BytesIO(r.content)).convert("RGB")
        px = min(255, int((x - tx) * 256)); py = min(255, int((y - ty) * 256))
        R, G, B = tiles[(tx, ty)].getpixel((px, py))
        return R * 256 + G + B / 256 - 32768

    grid = []
    for j in range(gh):
        lat = BBOX["max_lat"] + (BBOX["min_lat"] - BBOX["max_lat"]) * j / (gh - 1)  # 북→남 (영상과 동일)
        row = []
        for i in range(gw):
            lon = BBOX["min_lon"] + (BBOX["max_lon"] - BBOX["min_lon"]) * i / (gw - 1)
            row.append(round(height_at(lon, lat), 1))
        grid.append(row)
    flat = [v for r in grid for v in r]
    with open(os.path.join(DATA_DIR, "terrain.json"), "w") as fp:
        json.dump({"w": gw, "h": gh, "min": min(flat), "max": max(flat), "grid": grid}, fp)
    print(f"[OK] data/terrain.json — {gw}x{gh}, 표고 {min(flat):.0f}~{max(flat):.0f} m")


if __name__ == "__main__":
    if "여기에" in API_KEY:
        sys.exit("브이월드 인증키를 설정하세요 (환경변수 VWORLD_KEY 또는 파일 상단 API_KEY)")
    fetch_buildings()
    fetch_satellite()
    fetch_terrain()
    print("\n완료. 뷰어 실행:  python -m http.server 8000  →  http://localhost:8000/viewer.html")
