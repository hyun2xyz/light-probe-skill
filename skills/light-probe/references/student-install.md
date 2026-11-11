# 수강생 링크 설치: DiffusionLight까지 준비

수강생이 사용·설치 목적으로 `https://github.com/hyun2xyz/light-probe-skill` 링크를 전달하면 읽는다.
기본 설치 범위는 **환경 확인 → 스킬 설치 → 적합한 PyTorch와 의존성 설치 → DiffusionLight 가중치 다운로드·검증**이다.
수강생이 별도로 “DiffusionLight도 설치해줘”라고 말할 때까지 기다리지 않는다.
링크의 요약·코드 검토만 요청한 경우에는 설치를 실행하지 않는다.
수강생이 quick만 원하거나 다운로드를 제한하면 그 최신 요청을 우선한다.

## LLM이 이어서 실행할 순서

1. 저장소를 새 폴더에 clone하거나 이미 있는 checkout을 확인한다. README와 `skills/light-probe/SKILL.md`, `references/diffusion.md`를 읽는다. 설치 대상은 사용하는 클라이언트에 맞춘다.
2. OS·CPU 아키텍처·Python 버전·GPU/드라이버·메모리·여유 디스크를 실제로 확인한다. Python 3.10+ 중 PyTorch와 의존성이 지원하는 버전을 사용한다. GPU가 없으면 CPU 실행의 제약을 알리고 가능한 설치 단계를 계속한다. 유료 클라우드로 자동 전환하지 않는다.
3. 모델이 새로 필요하면 약 11.0 GB(25개 파일)를 받는다는 진행 안내를 한 번 하고 계속한다. 같은 다운로드를 위한 별도 선택 질문은 하지 않는다. 기존 캐시의 누락 용량과 Python 패키지·임시 파일 여유를 고려한다. 용량이 부족하면 사용자 파일을 지우지 말고 필요한 공간을 구체적으로 알린다.
4. 저장소 루트에서 `python install.py --client codex --setup` 또는 `--client claude --setup`을 실행한다. 이 명령은 **스킬 복사와 기초 가상환경까지만** 준비한다. 여기서 설치 완료로 끝내지 않는다. 반환된 `skill` 경로를 이후 작업의 스킬 폴더로 사용한다. 기존 설치는 내용과 경로를 확인한 후 재사용하거나 백업 가능한 `--replace`를 사용한다.
5. 설치된 스킬의 `.venv` Python을 사용한다. OS·GPU에 맞는 PyTorch 명령을 [공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 확인해 실행한다. NVIDIA에는 드라이버와 맞는 CUDA 빌드, Apple Silicon에는 MPS 가능한 빌드를 선택한다. CUDA 버전을 모든 환경에 고정하지 않는다. GPU 드라이버·권한 설정을 임의로 바꾸지 않는다.
6. 같은 가상환경으로 `python -m pip install -r requirements-diffusion.txt`와 `python -m pip check`를 실행한다. `torch`, `diffusers`, `transformers`, `peft`의 import와 실제 장치 사용 가능 상태를 확인한다.
7. 같은 스킬 폴더에서 `python scripts/download_models.py --check --verify`로 기존 캐시를 검사한다. 누락되었으면 **LLM이 곧바로** `python scripts/download_models.py --download`를 실행한다. 다운로드를 사용자에게 다시 요청하도록 떠넘기지 않는다. 이미 검증된 파일은 재사용한다. 손상 파일은 보존하고 오류 원인을 해결한다.
8. 다운로드 완료 후 `python scripts/download_models.py --check --verify`가 종료 코드 0, `ready: true`, `available_files: 25`인지 확인한다. 의존성 검사가 통과하고 파일 검증까지 끝났을 때만 “DiffusionLight 설치·모델 준비 완료”로 보고한다. 다운로드 시작이나 기초 환경 설치만으로 완료라고 하지 않는다.
9. 사진이 이미 첨부되었으면 `--mode diffusion --quality fast`로 이어서 분석한다. 사진이 없으면 사진을 올리면 사용할 수 있다고 안내한다. 실제 추론을 아직 하지 않았다면 추론 검증까지 통과했다고 말하지 않는다.

명령의 `python`은 단계 5부터 가상환경의 실행 파일이다. macOS/Linux는 `<SKILL_DIR>/.venv/bin/python`, Windows는 `<SKILL_DIR>/.venv/Scripts/python.exe`를 사용하면 활성화 없이 실행할 수 있다.
`--hub-root`를 바꿨으면 추론에도 같은 캐시가 적용되도록 `LIGHT_PROBE_HUB_ROOT`를 일관되게 설정한다.
세부 명령·모델 라이선스·고정 revision·재개 방법은 공개 배포본의 `references/diffusion.md`를 따른다.

## 실패와 실행 환경

셸·파일 실행·네트워크 권한이 없는 웹 채팅은 링크만으로 설치를 실행할 수 없다. 그 한계를 구체적으로 안내한다.
실행 환경이 요구하는 승인, 계정 인증, 모델 이용 조건 동의는 이 문서로 우회하지 않는다.
실제 접근 차단·디스크 부족·호환되지 않는 Python/GPU가 있으면 막힌 단계와 필요한 조치만 알린다.
다운로드 중에는 프로세스·로그·상태 경로를 추적하고 완료까지 확인한다. 기존 SDK의 중단 파일을 보존해 재개한다.
DiffusionLight 준비가 실패했을 때 quick으로 몰래 대체하거나 quick 결과를 모델 추론이라고 부르지 않는다.
quick은 사용자가 명시했을 때 쓰는 가벼운 대안이며, 링크 설치의 기본 완료 지점은 DiffusionLight 준비 완료다.
