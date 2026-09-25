# 청구동 3D 데모 (V-World API)

GASR 논문 §5 파이프라인의 축소판: 브이월드 공개 API만으로 서울 중구 **청구동** 일대를 3D로 재현하고,
그 위에 생성형 파사드 레이어를 **껐다 켤 수 있게** 얹는다.

## 실행

```
# 1. 브이월드 인증키 발급(무료): https://www.vworld.kr → 오픈API → 인증키
# 2. 데이터 수집
set VWORLD_KEY=발급받은키
pip install requests pillow
python fetch_cheonggu.py

# 3. 뷰어
python -m http.server 8000
# → http://localhost:8000/viewer.html
#    특정 건물로 이동: viewer.html#focus=삼성아파트
```

## 2계층 구조

이 데모는 **실측 계층**과 **생성 계층**을 엄격히 분리한다. 둘을 섞지 않는 것이 설계 원칙이다.

### 계층 1 — baseline (실측 데이터만)
데이터에 실제로 있는 것만 그린다. 추측한 텍스처나 굴곡은 넣지 않는다.

| 요소 | 출처 | 비고 |
|---|---|---|
| 건물 형상 | V-World 건물통합정보 `LT_C_SPBD` footprint | 4,348동 |
| 건물 높이 | 같은 데이터의 지상층수 `gro_flo_co` × 3.1 m | 1~23층 실측 분포 |
| 지면 텍스처 | V-World 위성 WMTS z17 | ~1.2 m/px |
| 지형 표고 | AWS Terrarium DEM z14 | 25~98 m (경사지) |
| 벽·지붕 색 | — | 무늬 없는 중립 회색 (데이터에 없으므로) |


### 계층 2 — 생성형 파사드 (선택적)
`data/facades/manifest.json`이 **있을 때만** 적용된다. 없으면 자동으로 계층 1로 폴백한다.

**되돌리는 방법 3가지 — 어느 것도 코드 수정이 필요 없다:**
1. `data/facades/` 폴더 삭제 → 영구 원복
2. URL에 `?gen=off` → 그 세션만 끔 (예: `viewer.html?gen=off#focus=삼성아파트`)
3. 실행 중 **G 키** → 즉시 on/off 토글 (A/B 비교용)

생성 텍스처가 적용 중일 때는 HUD에 주황색 표식으로 **"생성형 모델 산출물(실측 아님)"**이 항상 표시된다.
사용자가 무엇이 데이터고 무엇이 생성물인지 화면에서 바로 구분할 수 있어야 한다.

#### manifest.json 스키마
```json
{
  "generator": "모델·파이프라인 이름",
  "conditioned_on": "생성을 조건화한 실측 입력 (근거 명시)",
  "tiles": [
    {
      "id": "apt-tower",
      "file": "apt-tower.png",
      "tile_meters": [3.5, 3.1],
      "match": { "name_regex": "삼성아파트", "min_floors": 10 }
    }
  ]
}
```
`match`는 먼저 매칭되는 규칙이 이긴다. `tile_meters`는 텍스처 1타일이 덮는 실제 미터 크기(가로, 세로)로,
뷰어가 UV를 미터 단위로 매핑하므로 값이 곧 실축 정합을 결정한다.

## GPU 환경 (검증됨)
- RTX 5050 Laptop (Blackwell, compute capability 12.0), VRAM 8.5 GB
- `torch 2.13.0+cu130` — `sm_120` 커널 포함, fp16 matmul 실연산 검증 완료

```
pip install torch==2.13.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130 --retries 20 --timeout 180
pip install "diffusers==0.35.2" "transformers==4.57.6" "huggingface_hub<1.0" "accelerate<2" safetensors
```

**버전을 반드시 고정할 것.** 이 조합에 도달하기까지 실제로 겪은 충돌:
- cu128 채널 torch는 `sm_120` 커널이 없다 → cu130 필요
- 3 GB 휠은 기본 타임아웃에서 끊긴다 → `--retries 20 --timeout 180`
- `pip install diffusers --upgrade`가 torch 메타데이터를 깨뜨렸다 → torch는 항상 마지막에 설치
- diffusers 0.40은 `huggingface-hub>=1.23`을, transformers 4.x는 `<1.0`을 요구해 **공존 불가** →
  diffusers 0.35.2로 내려야 transformers 4.x와 맞는다 (transformers 5.x는 diffusers가 `PreTrainedModel` 임포트에 실패)

## 알려진 한계 (= GASR가 풀려는 문제)
- 건물이 각진 LoD1 박스: 지붕 형상·파사드 요철 없음 → 논문 §4 기하 SR 대상
- 위성 텍스처 z17: 흐릿함 → 논문 §4 텍스처 SR 대상
- 실측 파사드 텍스처 부재: V-World `XDServer3d`는 `ERROR_DB_GENERAL` 반환(타일 인덱스 규약 불명),
  거리뷰 계열은 약관상 재가공 불가 → 계층 2가 이 공백을 생성으로 메우되, 실측이 아님을 명시한다
