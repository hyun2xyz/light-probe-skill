# 기본 설치: DiffusionLight

DiffusionLight는 링크 설치의 기본 준비 대상이며 단일 사진에 크롬볼을 생성하고 주변 조명을 추정한다. LLM은 [학생 설치 절차](student-install.md)에 따라 의존성과 가중치까지 준비하며, 추가 모델 설치 요청을 기다리지 않는다. quick만 요청한 경우에는 이 단계를 생략한다. 실제 광원 위치·조도(lux)·굴절률·장면의 3D 기하를 측정하지 않는다. 입력 사진과 추론 요청은 외부로 전송하지 않으며, 모델 다운로더는 설치 단계에서 네트워크를 사용한다.

## 1. Python과 의존성

저장소의 기본 설치를 먼저 끝내고 `skills/light-probe` 폴더에서 작업한다. 아래 `python`은 기본 설치에서 만든 가상환경의 Python을 뜻한다. macOS/Linux에서는 `source .venv/bin/activate`, Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`로 활성화할 수 있다. 활성화가 제한된 Windows에서는 `.venv\Scripts\python.exe`를 `python` 대신 사용한다.

먼저 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 OS와 GPU에 맞는 명령을 골라 같은 가상환경에 설치한다. NVIDIA 장치의 CUDA 빌드는 드라이버와 맞아야 하며 모든 학생에게 동일한 CUDA 명령을 사용하지 않는다. Apple Silicon은 PyTorch의 MPS 경로를 사용한다. CPU도 코드상 지원하지만 속도와 메모리 요구량을 보장하지 않는다. 현재 엔진은 MPS, CUDA, CPU를 이 순서로 자동 선택한다.

그다음 DiffusionLight 의존성을 설치한다.

```sh
python -m pip install -r requirements-diffusion.txt
python -m pip check
```

`diffusers`와 `transformers`는 확인한 로딩·LoRA API에 맞춰 각각 `>=0.33.1,<0.34`, `>=4.51.3,<4.52`로 제한했다. 기본 requirements에는 PyTorch·Transformers·Hugging Face가 포함되지 않는다. 선택 requirements의 `peft`가 PyTorch에 의존하므로 하드웨어용 PyTorch를 먼저 설치해야 한다.

## 2. 모델 캐시와 준비 상태

기본 위치는 Hugging Face 표준 `~/.cache/huggingface/hub`이다. 경로 선택 순서는 다음과 같다.

1. 명령의 `--hub-root` 옵션(다운로더에만 적용)
2. `LIGHT_PROBE_HUB_ROOT`
3. `LIGHT_PROBE_HF_HUB`(이전 설정 호환용)
4. `HF_HUB_CACHE`, 이어서 이전 표준 이름 `HUGGINGFACE_HUB_CACHE`
5. `HF_HOME/hub`
6. `XDG_CACHE_HOME/huggingface/hub`, 미설정 시 `~/.cache/huggingface/hub`

추론에서도 같은 캐시를 쓰려면 환경변수를 설정한 뒤 명령을 시작한다. `--hub-root`만 주면 다음 추론 명령에는 그 설정이 전달되지 않는다. 예를 들어 현재 작업 폴더의 별도 캐시를 쓰는 경우:

macOS/Linux:

```sh
export LIGHT_PROBE_HUB_ROOT="$PWD/model-cache"
```

Windows PowerShell:

```powershell
$env:LIGHT_PROBE_HUB_ROOT = Join-Path $PWD 'model-cache'
```

실제 모델 가중치 없이도 준비 상태를 확인할 수 있다.

```sh
python scripts/download_models.py --check
```

`--check`와 인수 없는 실행은 로컬 파일의 존재·크기만 읽는다. 인터넷 연결, 선택 패키지 import, 디렉터리 생성, 모델 다운로드를 하지 않는다. 준비 완료는 종료 코드 `0`, 누락 또는 크기 불일치는 `1`이다. JSON의 `ready`, `missing`, `cache_root`를 확인한다. `ready`는 파일 준비 상태이며 GPU 호환성이나 추론 성공을 의미하지 않는다.

```sh
python scripts/download_models.py --check --verify
```

`--verify`는 모든 파일을 읽어 해시까지 검사한다. 가중치는 SHA256, 작은 설정·토크나이저 파일은 Git blob SHA1을 고정 메타데이터와 비교한다. 약 11 GB를 읽으므로 빠른 준비 확인보다 오래 걸린다.

## 3. LLM이 수행하는 모델 다운로드

아래 모델의 이용 조건과 실제 저장 공간을 확인한 뒤 LLM이 설치 절차 안에서 실행한다. 별도 다운로드 요청을 기다리지 않는다. 총 6개 저장소의 **25개 파일, 11,024,885,148바이트(약 11.0 GB / 10.3 GiB)**가 필요하며, 임시 파일·Python 패키지·추론 메모리는 별도다. 모델 파일은 배포 패키지에 포함하지 않는다.

```sh
python scripts/download_models.py --download
```

공식 `huggingface_hub.hf_hub_download`가 전체 인벤토리의 정확한 파일명과 고정 commit revision을 사용한다. 캐시·잠금·중단된 다운로드 재개는 SDK가 관리한다. 기존 캐시는 재사용하고 각 파일의 크기와 해시를 확인한다. Windows의 일반 파일 캐시와 macOS/Linux의 심볼릭 링크 캐시를 모두 처리한다. Windows에서 심볼릭 링크를 사용할 수 없으면 SDK가 더 많은 디스크 공간을 사용할 수 있다. 운영체제별 동작은 [공식 캐시 설명](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/manage-cache#limitations)을 따른다.

상태 JSON의 기본 위치는 `<hub-root>/light-probe-download-status.json`이며 `--status-file`로 바꿀 수 있다. `running`은 완료가 아니다. `done`과 `ready: true`를 확인한다. 중단 시 같은 `--download` 명령을 다시 실행한다. 이 스크립트는 SDK의 `.incomplete` 파일을 임의로 지우지 않는다. 기존 완성 파일이 검증에 실패하면 보존한 채 중단하므로 오류에 나온 해당 파일만 확인한 뒤 별도 경로에 보관하고 재시도한다.

인증 또는 접근 권한이 필요한 경우 해당 모델 페이지의 절차와 Hugging Face의 표준 로그인·환경변수 설정을 사용한다. 토큰을 스킬 파일이나 채팅에 적지 않는다. `HF_HUB_OFFLINE=1`은 다운로드를 막을 수 있으므로 새 터미널에서 설정을 확인한다. 추론의 offline 설정은 끄지 않는다.

공식 설정·다운로드 문서: [환경변수](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables), [파일 다운로드](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/download).

## 4. 추론

README의 입력·출력 경로 안내를 따라 새 출력 폴더로 실행한다.

```sh
python scripts/light_probe.py analyze input.jpg --out result-diffusion --mode diffusion --quality fast
```

`fast`는 512px, 12단계, EV0 단일 노출의 LDR 근사다. HDR 측정이나 복원이라고 부르지 않는다. `full`은 1024px, 노출별 30단계, EV 0/−2.5/−5를 생성해 합친 상대 HDR이다. 두 모드 모두 생성 모델이 사진 밖 조명을 추정하므로 실제 조명의 증거가 아니다.

추론은 고정된 로컬 snapshot 경로와 `local_files_only=True`를 사용하고 HF/Transformers offline 설정을 켠다. 누락 모델을 자동으로 받지 않는다. 같은 사용자의 다른 Light Probe 프로세스가 모델을 보유하면 OS 잠금으로 중복 추론을 거부한다. 프로세스가 끝나면 잠금은 해제된다. 로딩 후 캐시를 바꾸려면 새 Python 프로세스를 시작한다.

## 모델 원본과 권리

| 구성 요소 | 공식 모델 저장소 | 고정 revision | 원본의 표시 라이선스 |
| --- | --- | --- | --- |
| SDXL base | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `462165984030d82259a11f4367a4eed129e94a7b` | CreativeML Open RAIL++-M |
| Depth ControlNet | [diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) | `17bb97973f29801224cd66f192c5ffacf82648b4` | CreativeML Open RAIL++-M |
| VAE | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | `207b116dae70ace3637169f1ddd2434b91b3a8cd` | MIT |
| DPT depth | [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) | `bc15f29aa3a80d532f2ed650b5e16ac48d8958f9` | Apache-2.0 |
| Turbo LoRA | [DiffusionLight/Flickr2K](https://huggingface.co/DiffusionLight/Flickr2K) | `ed635477e5a742e363e54a6ed2193a9a43153113` | MIT(저장소 메타데이터) |
| Exposure LoRA | [DiffusionLight/DiffusionLight](https://huggingface.co/DiffusionLight/DiffusionLight) | `bda34823e932601b3eb9aff0276f00925cdeea46` | MIT(저장소 메타데이터) |

코드의 MIT 라이선스가 모델 이용 조건을 대체하지 않는다. SDXL 기반 구성 요소를 함께 사용할 때 그 기반 모델의 조건도 확인한다. 저장소의 라이선스 표시는 모든 학습 데이터·생성 결과의 권리를 검증했다는 뜻이 아니다. 원본 코드 고지는 [THIRD_PARTY_LICENSE.txt](../THIRD_PARTY_LICENSE.txt)에 보존한다.

## 검증 범위

포팅 시 원본 환경에서 확인한 버전은 Python 런타임의 torch 2.15.0, diffusers 0.33.1, transformers 4.51.3, huggingface_hub 0.36.2, peft 0.21.2, accelerate 1.15.0, safetensors 0.8.0, filelock 4.2.10이다. 이는 모든 장치에서 이 조합의 설치·성능을 보증하는 표가 아니다.

원본 구현에는 Apple MPS에서 `fast` 추론이 성공한 기록이 있다. 공개 배포용 포팅에서는 기존 캐시의 읽기 전용 준비 상태, 파일 목록·해시 메타데이터, 작은 가상 파일 다운로드, 잠금·경로·offline 회귀 검사를 수행한다. 대형 모델을 새로 다운로드하거나 추론을 다시 실행하지 않는다. **새 Windows/Linux 컴퓨터의 실제 설치·GPU 추론, 새 캐시에서의 전체 모델 다운로드, `full` 세 노출 실제 생성은 미검증**이다. 사용 장치에서 별도로 확인해야 한다.
