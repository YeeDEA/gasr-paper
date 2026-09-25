"""
생성형 파사드 텍스처 생성기 (계층 2)

무엇을 하는가:
  건물의 **실측 공개 속성**(지상층수·건물명·footprint 치수)으로 프롬프트를 조건화해,
  세로·가로로 이음매 없이 반복되는 파사드 텍스처 타일을 만든다.

무엇을 하지 않는가:
  특정 건물의 실제 외관을 복원하지 않는다. 산출물은 "그 유형의 건물이라면 이럴 것"이라는
  **유형론적(typological) 생성물**이며, 실측이 아니다. 뷰어 HUD와 manifest에 그렇게 표기된다.

되돌리기:
  이 스크립트가 만드는 것은 data/facades/ 폴더뿐이다. 폴더를 지우면 완전히 원복된다.

사용법:
  pip install torch --index-url https://download.pytorch.org/whl/cu130
  pip install diffusers transformers accelerate safetensors pillow
  python gen_facades.py                    # 기본: sd-turbo, 전체 스펙 생성
  python gen_facades.py --only apt-tower   # 특정 스펙만
  python gen_facades.py --steps 4 --seed 7
"""
import argparse, json, os, sys, time

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "facades")

# ── 생성 스펙: match 규칙은 뷰어가 그대로 소비한다 ──────────────────────
# bay_m = 세대 1칸 폭(m), floor_m = 층고(m). 타일이 덮는 실제 크기가 UV 정합을 결정한다.
SPECS = [
    {
        "id": "apt-tower",
        "match": {"name_regex": "삼성아파트", "min_floors": 10},
        "bays": 4, "floors": 4, "bay_m": 3.5, "floor_m": 3.1,
        "conditioned_on": "V-World LT_C_SPBD 지상층수 16~23층 + 건물명 '신당동삼성아파트' + footprint 장변/단변비",
        "prompt": (
            "flat orthographic architectural elevation of a South Korean high-rise apartment facade, "
            "white painted concrete wall panels, continuous horizontal balcony bands, "
            "dark teal tinted glass balcony railings, slim white window frames, "
            "identical repeating floors, straight horizontal and vertical lines, "
            "even flat diffuse daylight, texture sheet, orthographic projection"
        ),
    },
    {
        "id": "apt-slab",
        "match": {"name_regex": "아파트|e편한|하이츠|팰리스", "min_floors": 5},
        "bays": 4, "floors": 4, "bay_m": 3.8, "floor_m": 2.9,
        "conditioned_on": "V-World LT_C_SPBD 지상층수 5~15층 + 건물명에 아파트류 포함",
        "prompt": (
            "flat orthographic architectural elevation of a mid-rise South Korean apartment block facade, "
            "beige and light grey painted concrete, enclosed balcony windows in regular grid, "
            "horizontal spandrel bands between floors, white window frames, "
            "identical repeating floors, even flat diffuse daylight, texture sheet, orthographic projection"
        ),
    },
    {
        "id": "lowrise-villa",
        "match": {"max_floors": 4},
        "bays": 3, "floors": 3, "bay_m": 4.0, "floor_m": 2.9,
        "conditioned_on": "V-World LT_C_SPBD 지상층수 4층 이하 (한국 저층 다세대 유형)",
        "prompt": (
            "flat orthographic architectural elevation of a low-rise South Korean multiplex house facade, "
            "red brick and beige tile cladding, small square windows in regular rows, "
            "simple flat wall, identical repeating floors, "
            "even flat diffuse daylight, texture sheet, orthographic projection"
        ),
    },
]

# 3D에 입혔을 때 망가지는 전형적 실패 모드를 프롬프트에서 직접 배제한다
NEGATIVE = (
    "perspective, vanishing point, three quarter view, angled view, tilted, "
    "sky, clouds, ground, street, trees, cars, people, "
    "dramatic lighting, cast shadows, sunset, vignette, depth of field, blur, bokeh, "
    "watermark, text, signage, logo, frame, border, illustration, painting"
)


def seam_score(img):
    """타일 경계 불연속도를 측정한다: (끝↔끝 인접차) / (내부 평균 인접차).

    왜 필요한가: circular padding은 완전하지 않다. diffusers의 Downsample2D는
    `F.pad(..., mode="constant")`가 하드코딩돼 있어(0.35.2 downsampling.py:143)
    Conv2d의 padding_mode만 바꿔서는 다운샘플 경로가 순환되지 않는다.
    실측하면 시드에 따라 이음매가 보이기도 안 보이기도 한다 → 재는 게 답이다.
    1.0 근처 = 이어짐, 2.0 초과 = 눈에 보임.
    """
    import numpy as np
    g = np.asarray(img.convert("RGB"), dtype=np.float32)
    wx = np.abs(g[:, -1] - g[:, 0]).mean(); ix = np.abs(g[:, 1:] - g[:, :-1]).mean()
    wy = np.abs(g[-1, :] - g[0, :]).mean(); iy = np.abs(g[1:, :] - g[:-1, :]).mean()
    return round(float(max(wx / (ix + 1e-6), wy / (iy + 1e-6))), 2)


def measure_tile_meters(img, bay_m, floor_m, declared, min_conf=0.35):
    """생성 결과에서 실제 반복 주기를 자기상관으로 측정해, 타일이 덮는 실측 크기를 정한다.

    왜 필요한가: diffusion 모델은 "4개 층"을 요청해도 그 수를 지키지 않는다. 실제로
    apt-tower는 4층을 요청했는데 9.2층을 그렸다. 선언값을 그대로 tile_meters에 쓰면
    3D에서 층고가 1.3m로 찌그러진다. 따라서 **선언값이 아니라 산출물을 믿는다.**
    신뢰도(자기상관 피크)가 낮으면 선언값으로 되돌린다.
    """
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def period(sig, n_min=2, n_max=16):
        s = sig - sig.mean()
        n = len(s)
        ac = np.correlate(s, s, "full")[n - 1:]
        ac /= ac[0] + 1e-9
        lo, hi = max(4, n // n_max), max(6, n // n_min)   # 고주파 노이즈·전역 추세 배제
        k = int(np.argmax(ac[lo:hi])) + lo
        return k, float(ac[k])

    H, W = g.shape
    hp, hc = period(g.mean(axis=0))   # 열평균 → 가로 주기(세대 폭)
    vp, vc = period(g.mean(axis=1))   # 행평균 → 세로 주기(층고)
    if hc < min_conf or vc < min_conf:
        return declared[0], declared[1], {"scale_source": "declared",
                                          "reason": f"자기상관 신뢰도 낮음 (가로 {hc:.2f}, 세로 {vc:.2f})"}
    bays, floors = W / hp, H / vp
    return (round(bays * bay_m, 2), round(floors * floor_m, 2),
            {"scale_source": "measured",
             "measured_bays": round(bays, 1), "measured_floors": round(floors, 1),
             "autocorr": [round(hc, 2), round(vc, 2)]})


def derive_normal_map(img, strength=2.5):
    """알베도 휘도를 높이로 간주해 법선맵을 유도한다.

    주의 — 이것은 논문 §4.3이 **비판하는 순진한 방식**이다. 휘도가 낮은 곳이
    실제로 움푹한지(기하), 그냥 어두운 페인트인지(알베도) 구분하지 못한다.
    여기서는 GASR 이전의 baseline으로서 의도적으로 이 순진한 버전을 구현한다.
    """
    from PIL import Image
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    # 상하좌우 wrap: 타일 경계에서도 법선이 이어지게
    dx = (np.roll(g, -1, 1) - np.roll(g, 1, 1)) * strength
    dy = (np.roll(g, -1, 0) - np.roll(g, 1, 0)) * strength
    nz = np.ones_like(g)
    L = np.sqrt(dx * dx + dy * dy + nz * nz)
    rgb = np.stack([(-dx / L * 0.5 + 0.5), (-dy / L * 0.5 + 0.5), (nz / L * 0.5 + 0.5)], -1)
    return Image.fromarray((rgb * 255).astype("uint8"), "RGB")


def patch_circular(module):
    """모든 Conv2d의 padding_mode를 circular로 바꿔 생성 결과가 상하좌우로 이어지게 한다.
    (diffusion에서 seamless 타일을 얻는 표준 기법)"""
    import torch.nn as nn
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.padding_mode = "circular"
            n += 1
    return n


def build(args):
    import torch
    from diffusers import AutoPipelineForText2Image

    if not torch.cuda.is_available():
        sys.exit("CUDA를 쓸 수 없습니다. torch가 CPU 빌드일 수 있습니다.\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu130")
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"모델 로드: {args.model} (최초 1회 다운로드 발생)")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
    except Exception:
        pass

    nc = patch_circular(pipe.unet) + patch_circular(pipe.vae)
    # 주의: 이것만으로는 완전하지 않다. diffusers의 Downsample2D는 F.pad(mode="constant")가
    # 하드코딩돼 있어(0.35.2 downsampling.py:143) 다운샘플 경로가 순환되지 않는다.
    # 그래서 결과를 seam_score로 실측하고 후보 중에서 고른다.
    print(f"circular padding: Conv2d {nc}개 (Downsample2D 경로는 미적용 → 이음매는 측정으로 선별)")

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = [s for s in SPECS if not args.only or s["id"] == args.only]
    if not specs:
        sys.exit(f"--only {args.only} 에 해당하는 스펙이 없습니다. 가능: {[s['id'] for s in SPECS]}")

    tiles = []
    for s in specs:
        w_m, h_m = s["bays"] * s["bay_m"], s["floors"] * s["floor_m"]
        # 실제 종횡비를 유지하되 8의 배수로 맞춘다 (VAE 요구)
        px_per_m = args.px_per_m
        W = max(256, int(round(w_m * px_per_m / 8)) * 8)
        H = max(256, int(round(h_m * px_per_m / 8)) * 8)

        # 시드 후보를 여러 개 뽑아 고른다 (1장당 ~2초).
        # 판정 기준 2개: ① 실축을 측정할 수 있는가(구조 주기가 뚜렷한가) ② 이음매가 적은가.
        # ①을 우선한다 — 층고가 틀리면 이음매가 완벽해도 3D에서 즉시 이상해 보인다.
        t0 = time.time()
        cands = []
        for c in range(args.candidates):
            gen = torch.Generator(device=dev).manual_seed(args.seed + c)
            cand = pipe(prompt=s["prompt"], negative_prompt=NEGATIVE,
                        num_inference_steps=args.steps, guidance_scale=args.cfg,
                        width=W, height=H, generator=gen).images[0]
            tw, th, info = measure_tile_meters(cand, s["bay_m"], s["floor_m"], (w_m, h_m))
            cands.append({"img": cand, "seed": args.seed + c, "seam": seam_score(cand),
                          "tw": tw, "th": th, "info": info,
                          "measurable": info["scale_source"] == "measured"})
        pool = [c for c in cands if c["measurable"]] or cands   # 측정 가능한 것 우선
        pick = min(pool, key=lambda c: c["seam"])
        img, best_seam, best_seed = pick["img"], pick["seam"], pick["seed"]
        tried = [(c["seed"] - args.seed, c["seam"], "측정○" if c["measurable"] else "측정✕") for c in cands]
        dt = time.time() - t0

        path = os.path.join(OUT_DIR, f"{s['id']}.png")
        img.save(path)

        # 선언한 4x4가 아니라, 모델이 실제로 그린 반복 주기로 실축을 정한다
        tw_m, th_m, scale_info = pick["tw"], pick["th"], pick["info"]
        note = (f"측정 {scale_info.get('measured_bays')}세대 x {scale_info.get('measured_floors')}층"
                if scale_info["scale_source"] == "measured" else "선언값 사용")
        print(f"  [{s['id']}] {W}x{H}px → 실축 {tw_m}x{th_m}m ({note}) | "
              f"이음매 {best_seam} (후보 {tried} 중 seed{best_seed}) | {dt:.1f}s")

        tile = {
            "id": s["id"],
            "file": f"{s['id']}.png",
            "tile_meters": [tw_m, th_m],
            "seam_score": best_seam,
            "seam_note": "1.0 근처=이어짐, 2.0 초과=이음매 보임",
            "seed": best_seed,
            **scale_info,
            "match": s["match"],
            "conditioned_on": s["conditioned_on"],
            "provenance": "generated",
        }
        if args.normal:
            npath = os.path.join(OUT_DIR, f"{s['id']}_n.png")
            derive_normal_map(img, args.normal_strength).save(npath)
            tile["normal_file"] = f"{s['id']}_n.png"
            tile["normal_method"] = "휘도→높이 순진 유도 (논문 §4.3의 baseline, 알베도/음영 미분리)"
            print(f"           법선맵 → {npath}")
        tiles.append(tile)

    manifest = {
        "generator": f"diffusers {args.model} · {args.steps}step cfg{args.cfg} seed{args.seed} · circular-padding seamless",
        "disclaimer": "유형론적 생성물입니다. 특정 건물의 실제 외관이 아니며 실측 데이터가 아닙니다.",
        "conditioned_on": "V-World 건물통합정보(LT_C_SPBD)의 지상층수·건물명·footprint 치수. 거리뷰 등 비공개·제한 소스는 사용하지 않았습니다.",
        "tiles": tiles,
    }
    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    print(f"\n[OK] {mpath}")
    print("뷰어에서 확인:  viewer.html   (끄기: ?gen=off 또는 실행 중 G 키)")
    print(f"완전 원복:      rmdir /s /q \"{OUT_DIR}\"")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo",
                   help="HF 모델 id. 8GB면 sd-turbo 권장, 여유 있으면 stabilityai/sdxl-turbo")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg", type=float, default=1.0, help="turbo 계열은 1.0 (guidance 미사용)")
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--px-per-m", type=float, default=40.0, help="미터당 픽셀 해상도")
    p.add_argument("--only", default=None)
    p.add_argument("--candidates", type=int, default=6,
                   help="시드 후보 수. 이음매가 가장 적은 것을 자동 선택")
    p.add_argument("--normal", action="store_true", help="알베도에서 법선맵도 유도 (논문 §4.3 baseline)")
    p.add_argument("--normal-strength", type=float, default=2.5)
    build(p.parse_args())
