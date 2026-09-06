# Third-Party Notices

이 저장소가 사용하는 제3자 자산과 그 라이선스를 정리합니다.

---

## 1. Microsoft Fluent Emoji (3D style)

- 위치: `assets/illustrations/*.webp`
- 원본: https://github.com/microsoft/fluentui-emoji
- 라이선스: MIT License
- Copyright: Copyright (c) Microsoft Corporation.
- 원본 PNG(256x256)를 WebP(quality=90)로 변환해 저장했습니다(2026-07-16). 어떤 파일이
  어떤 원본 이모지에서 왔는지는 `assets/illustrations/SOURCE.md`에 파일별로 정리돼 있습니다.
- 라이선스 원문(2026-09-06 기준 원본 저장소 `LICENSE` 파일에서 직접 확인):

  ```
  MIT License

  Copyright (c) Microsoft Corporation.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

  참고: 이 저장소의 현재 화면(app.py)에서는 이 일러스트를 그리는 함수(`icon_3d`,
  `ui_icons.py`)가 실제로는 호출되지 않습니다(과거 디자인에서 쓰였던 대체 경로).
  라이선스가 확실히 확인되어 그대로 두었지만, 코드 검토 시 참고하세요.

## 2. Phosphor Icons (duotone SVG path data)

- 위치: `ui_icons.py`의 `svg_icon()` 함수 내 SVG path 문자열 (이미지 파일이 아니라
  코드에 직접 임베드된 벡터 경로 데이터)
- 원본: https://github.com/phosphor-icons/core (npm 패키지 `@phosphor-icons/core`)
- 라이선스: MIT License
- Copyright: Copyright (c) 2023 Phosphor Icons
- 페이지 헤더 아이콘 등 앱 전반의 네이비 duotone 아이콘이 이 경로 데이터를 그대로 사용합니다.

---

## 3. `assets/icons/*.png` (ChatGPT 이미지 생성)

- 이 프로젝트 전용으로 ChatGPT 이미지 생성 기능으로 직접 만든 아이콘입니다(제3자 배포 자산을
  가져온 것이 아님) - 출처 불명 자산이 아니라 자체 제작물이라 정리 대상에서 제외했다가,
  2026-09-06 사용자 확인 후 다시 포함했습니다. 실제 운영 화면(job_ai_v3)이 쓰는 32종 중 이
  저장소 코드가 실제로 참조하는 20종만 선별해 가져왔습니다(`ASSET_NOTES.md` 참고).
