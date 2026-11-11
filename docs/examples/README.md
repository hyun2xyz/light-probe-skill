# 실제 사진 실행 기록

모든 비교 이미지의 왼쪽은 입력 사진, 오른쪽은 실제 추론 결과에 주석을 얹은 도판이다.
표시용 크기 조절·흰 여백·JPEG 저장 외에 원본 사진을 새로 생성하지 않았다.
`input.png`는 실제 분석에 전달한 정규화 입력이다. `analysis.json`의 SHA256과 일치한다.

## 출처와 사진 라이선스

| 폴더 | 사진가 / 원본 | 라이선스 | 이미지 준비 |
|---|---|---|---|
| `bed` | [Sydney Moore / Unsplash](https://unsplash.com/photos/a-bed-with-a-blanket-on-it-in-a-bedroom-wJtEEDb4HBM) | [Unsplash License](https://unsplash.com/license) | [DiffusionLight의 정사각 크롭](https://github.com/DiffusionLight/DiffusionLight/blob/main/example/bed.png), 1024×1024 |
| `lounge` | [Tom Schönmann / Pexels](https://www.pexels.com/photo/armchairs-in-living-room-in-sunlight-26932519/) | [Pexels License](https://www.pexels.com/license/) | 비율 유지, 4032×3024 → 1024×768 |
| `facade` | [Mathias Reding / Pexels](https://www.pexels.com/photo/sunlight-and-shadow-on-building-12519360/) | [Pexels License](https://www.pexels.com/license/) | 비율 유지, 4146×5183 → 819×1024 |

2026-11-11에 원본 페이지와 라이선스를 확인했다. 사진 자체는 저장소의 코드 MIT 라이선스 대상이 아니다.
수업용 기능 설명과 원본/분석 결과 비교로 전시하며, 사진 판매·스톡 이미지 서비스·학습 데이터셋으로 제공하지 않는다.
재사용할 때는 각 원본 서비스의 현행 라이선스와 이용 조건을 따른다. 사진가가 이 도구를 보증한다는 뜻이 아니다.

## 실제 실행과 검수

| 항목 | 침실 | 실내 의자 | 외벽 |
|---|---|---|---|
| 결과 출처 | 기존 실제 추론을 재사용 | 새 실제 추론 | 새 실제 추론 |
| 계산 장치 | Apple MPS | Apple MPS | Apple MPS |
| 모델 경로 | DiffusionLight fast | DiffusionLight fast | DiffusionLight fast |
| 크롬볼 생성 | 512px / 12단계 / EV0 / seed 200 | 동일 | 동일 |
| 전체 분석 시간 | 47.381초 | 41.969초 | 9.299초 |
| 로딩 | 최초 로딩 포함 | 최초 로딩 포함 | 실내 실행 뒤 같은 프로세스의 모델 재사용 |
| 표시 | 방향·3단계 산란 | 방향·2단계 산란 | 주광·추가 방향 후보 |

실내와 외벽은 공개 배포 코드 `d794b5d92fcc7bc30292e4fcbe77012815c29087`로 순서대로 실행했다.
기존 로컬 모델 캐시를 사용했으며 이번 작업에서 모델을 다시 다운로드하지 않았다.
생성된 `ball_ev00.png`, `probe_preview.png`와 상태 완료를 확인했다.
라벨 배치는 모델 재추론 없이 조정했고, 최종 그림과 추론 기록의 광원 벡터가 같은지 확인했다.

윤곽·물성·2D 충돌 선분은 LLM 판독과 설명용 입력이다. 실제 검출·분할 모델이나 3D 기하 복원으로 만든 값이 아니다.
실행 성공과 시각적 검수를 확인한 예제이며, 알려진 조명 정답·실측 반사율과 비교한 정확도 검증은 아니다.
fast는 단일 LDR 노출이므로 full HDR 검증으로 확대하지 않는다.

## 재실행

저장소 루트에서 실행한다. 기본 설치 가이드에 따라 모델을 먼저 준비한다.
아래 명령은 실제 GPU 추론을 실행하며 출력 폴더는 새 경로를 사용해야 한다.

```sh
python skills/light-probe/scripts/run.py analyze docs/examples/bed/input.png --out outputs/bed-real --mode diffusion --quality fast --display all --anchor 0.5 0.65 --context docs/examples/bed/context.json
python skills/light-probe/scripts/run.py analyze docs/examples/lounge/input.png --out outputs/lounge-real --mode diffusion --quality fast --display all --anchor 0.52 0.87 --context docs/examples/lounge/context.json
python skills/light-probe/scripts/run.py analyze docs/examples/facade/input.png --out outputs/facade-real --mode diffusion --quality fast --display direction --anchor 0.60 0.59 --context docs/examples/facade/context.json
```

위 명령을 각각 새 프로세스로 실행하면 각 실행에 모델 로딩 시간이 포함된다. 하드웨어·소프트웨어 차이에 따라 생성 결과와 시간도 달라질 수 있다.
표시용 비교 이미지는 `scripts/export_readme_example.py --help`의 입력으로 재생성할 수 있다.
