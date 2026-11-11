# 실행 환경

최초 설치·환경 진단 때 읽는다. 모델 설치는 [diffusion.md](diffusion.md)를 읽는다.

## 기본 모드

Python 3.10 이상과 셸·파일 접근이 필요하다. 이미지 판독을 위해 비전 기능이 있는 LLM을 사용한다.
스킬 폴더에서 다음을 실행한다. Windows에서는 `python` 대신 `py -3`를 사용할 수 있다.

```sh
python scripts/bootstrap.py
python scripts/bootstrap.py --check
python scripts/run.py analyze /path/to/image.png --out /path/to/new-output --mode quick
```

`bootstrap.py`는 스킬 내부 `.venv`에만 NumPy·Pillow·OpenCV를 설치한다.
전역 Python, LLM 설정, 권한 정책은 바꾸지 않는다. 기본 모드는 torch·GPU·Laya·API 키를 요구하지 않는다.
설치 후 가상환경 활성화 없이 `run.py`를 호출할 수 있다.
최초 pip 설치 실패 시 네트워크·Python 버전을 확인한 후 같은 bootstrap 명령으로 재시도한다.
Linux에서 venv 모듈이 없으면 해당 배포판의 Python venv 패키지가 필요하다.

## 다른 컴퓨터로 이동

저장소 또는 ZIP에는 모델·개인 사진·가상환경을 포함하지 않는다.
복사 후 bootstrap을 다시 실행한다. `.venv`는 다른 OS로 복사하지 않는다.
한글 폰트 NanumGothic을 동봉하며 라이선스는 `assets/fonts/OFL.txt`다.
시스템에 한국어 폰트가 있으면 그것을 먼저 사용하고 없으면 동봉 폰트로 표시한다.

## LLM과 Laya

장면과 요청 해석은 사용하는 LLM이 담당한다. Laya는 필수가 아니며 이 배포본에 포함하지 않는다.
기존 Laya 환경이 있으면 외부에서 제한된 CLI 옵션을 제안할 수 있으나 빛 추론 속도가 빨라지는 것은 아니다.
기본 실행기는 어떤 LLM API도 호출하지 않는다. 사용자가 이미지를 채팅에 첨부하면 해당 LLM 서비스의 처리는 별개다.

## 검증 범위

개발 Mac에서 기본 설치·분석·재렌더·출력 검수와 수학/마스크 회귀 검사를 수행한다.
저장소 CI는 macOS·Windows·Linux에서 기본 설치와 테스트를 수행한다. 각 커밋의 실제 Actions 결과를 확인한다.
DiffusionLight fast의 Apple MPS 실행 이력이 있으며, 다른 GPU 속도나 full 생성 결과를 보장하지 않는다.
