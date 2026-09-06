"""
resume_input/job_card_component.py

STEP2 "공고 찾기" 목록의 공고 카드(2026-08-16, 사용자 확정) - CCv2로
전환한다. 이전에는 native st.container/st.columns를 여러 겹 쌓고
Streamlit이 만든 DOM에 `div[class*="st-key-...-"]` 셀렉터로 CSS를
덮어써서 "원문 보기+관심 버튼 폭 = 분석하기 버튼 폭"처럼 정교한
레이아웃을 맞추려 했다 - 그때그때 Streamlit이 실제로 어떤 DOM을
만드는지 실측하고, 안 맞으면 !important를 하나씩 더 쌓는 흐름이
반복됐다(사용자 지적 - "CCv2로 넘어간 이유와 반대"). 공고 분석 화면
(analysis_component.py)과 동일하게, 정교한 레이아웃이 필요한 반복
UI는 CCv2 컴포넌트 내부에서 CSS Grid/Flex로 직접 그린다 - 그러면
Streamlit이 생성하는 wrapper DOM을 추측/우회할 필요가 없다.

데이터는 전부 app.py가 이미 계산한 값(presentation_layer.
build_job_card_view() + ui_components.resolve_company_logo())을 그대로
JSON으로 넘기기만 한다 - 새 판단/새 문장을 만들지 않는다. 검색/정렬/
페이지네이션/저장/분석 전환 같은 실제 로직은 전부 app.py(Python)에
그대로 남아있다 - 이 컴포넌트는 표시와 클릭 트리거만 담당한다.

CCv2(Custom Components v2)만 쓴다 - v1은 deprecated라 새 코드에 쓰지
않는다(analysis_component.py와 동일 원칙). "원문 보기"는 단순 외부
링크라 Python 왕복이 필요 없다(target=_blank 앵커). "관심 공고(☆)"/
"목록에서 제외(×)"/"분석하기" 클릭만 setTriggerValue로 {jobId, type}
하나의 트리거에 실어 Python에 보낸다(type ∈ save|dismiss|analyze).
`×`(dismiss)는 공고를 삭제하는 게 아니라 이 사용자의 추천 목록에서만
빼는 표시다 - 실제 제외/보충/되돌리기 로직은 전부 app.py에 있다.

보안: 회사명/직무명은 크롤링된 외부 문자열이라 XSS 위험이 있다 - JS에서
innerHTML에 넣기 전 반드시 esc()로 이스케이프한다.
"""

import streamlit as st

_HTML = """
<div id="jc-root" class="jc-root"></div>
"""

_CSS = """
.jc-root {
  font-family: -apple-system, "Segoe UI", Inter, sans-serif;
  color: #0F1F4A;
}
.jc-root * { box-sizing: border-box; }
.jc-list { display: flex; flex-direction: column; gap: 12px; }
.jc-card {
  display: flex; align-items: flex-start; gap: 16px;
  background: linear-gradient(180deg, #FBFCFF 0%, #FFFFFF 65%);
  border: 1px solid #E5E7EB; border-radius: 16px; padding: 16px;
  box-shadow: 0 4px 14px rgba(15, 31, 74, 0.06), 0 1px 3px rgba(15, 31, 74, 0.04);
  transition: box-shadow .18s ease;
}
.jc-card:hover { box-shadow: 0 10px 24px rgba(15, 31, 74, 0.09); }
.jc-logo { width: 64px; height: 64px; flex-shrink: 0; border-radius: 10px; }
.jc-logo-initial {
  display: flex; align-items: center; justify-content: center;
  background: #0F1F4A; color: #FFFFFF; font-weight: 800; font-size: 28px;
}
.jc-logo-img { object-fit: contain; background: #FFFFFF; border: 1px solid #E5E7EB; padding: 5px; }
.jc-content { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.jc-company-row { display: flex; align-items: center; gap: 8px; }
.jc-company { font-size: 13px; color: #6B7280; }
.jc-title-row { display: flex; align-items: center; gap: 16px; flex-wrap: nowrap; min-width: 0; }
.jc-title {
  font-size: 16px; font-weight: 600; color: #0F1F4A;
  min-width: 0; flex: 0 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.jc-skills { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.jc-skill {
  display: inline-flex; padding: 3px 10px; border-radius: 999px; flex-shrink: 0;
  background: #F5F7FA; color: #0F1F4A; font-size: 12px; font-weight: 600; white-space: nowrap;
}
.jc-skill-more { font-size: 12px; color: #6B7280; flex-shrink: 0; }
.jc-meta { font-size: 14px; color: #6B7280; }
.jc-badge {
  display: inline-flex; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
}
.jc-badge-red { background: #FBE3E5; color: #E0555F; }
.jc-badge-orange { background: #FFFBEA; color: #B9702E; }
.jc-badge-gray { background: #F5F7FA; color: #6B7280; }
.jc-actions { width: 220px; flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; }
.jc-actions-top { display: grid; grid-template-columns: 1fr 36px 36px; gap: 6px; }
.jc-btn {
  display: inline-flex; align-items: center; justify-content: center;
  height: 36px; border-radius: 8px; font-size: 14px; font-weight: 600;
  cursor: pointer; text-decoration: none; border: 1px solid #E5E7EB;
  background: #FFFFFF; color: #0F1F4A; padding: 0 12px; white-space: nowrap;
}
.jc-btn:hover { background: #F5F8FF; }
.jc-btn-disabled { opacity: .5; cursor: default; pointer-events: none; }
.jc-btn-star { padding: 0; font-size: 18px; }
.jc-btn-x { padding: 0; font-size: 17px; color: #9CA3AF; font-weight: 400; }
.jc-btn-x:hover { background: #F5F7FA; color: #6B7280; }
.jc-btn-primary { background: #0F1F4A; border-color: #0F1F4A; color: #FFFFFF; height: 40px; }
.jc-btn-primary:hover { background: #1A2C63; border-color: #1A2C63; }
.jc-btn-block { width: 100%; }
@media (max-width: 720px) {
  .jc-card { flex-wrap: wrap; }
  .jc-actions { width: 100%; }
}
"""

_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector('#jc-root');
  if (!root || !data) return;

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = String(s ?? '');
    return d.innerHTML;
  }

  const cards = data.cards || [];

  function logoHtml(logo) {
    if (logo && logo.type === 'favicon') {
      return '<img class="jc-logo jc-logo-img" src="' + esc(logo.url) + '" alt="" />';
    }
    return '<div class="jc-logo jc-logo-initial">' + esc((logo && logo.letter) || '?') + '</div>';
  }

  function cardHtml(c) {
    const deadlineBadge = c.deadlineBadge
      ? '<span class="jc-badge jc-badge-' + esc(c.deadlineTone || 'gray') + '">' + esc(c.deadlineBadge) + '</span>'
      : '';
    const skillChips = (c.skills || []).map(function (s) {
      return '<span class="jc-skill">' + esc(s) + '</span>';
    }).join('') + (c.skillOverflow > 0 ? '<span class="jc-skill-more">+' + c.skillOverflow + '</span>' : '');
    const skillsHtml = skillChips ? '<div class="jc-skills">' + skillChips + '</div>' : '';
    const urlHtml = c.url
      ? '<a class="jc-btn" href="' + esc(c.url) + '" target="_blank" rel="noopener noreferrer">원문 보기 &#8599;</a>'
      : '<span class="jc-btn jc-btn-disabled">원문 보기 &#8599;</span>';
    return (
      '<div class="jc-card">' +
        logoHtml(c.logo) +
        '<div class="jc-content">' +
          '<div class="jc-company-row"><span class="jc-company">' + esc(c.company) + '</span>' + deadlineBadge + '</div>' +
          '<div class="jc-title-row"><span class="jc-title">' + esc(c.title) + '</span>' + skillsHtml + '</div>' +
          '<div class="jc-meta">' + esc(c.meta) + '</div>' +
        '</div>' +
        '<div class="jc-actions">' +
          '<div class="jc-actions-top">' +
            urlHtml +
            '<button type="button" class="jc-btn jc-btn-star" data-action="save" data-job-id="' + esc(c.jobId) + '" title="관심 공고 저장">' +
              (c.saved ? '&#9733;' : '&#9734;') +
            '</button>' +
            '<button type="button" class="jc-btn jc-btn-x" data-action="dismiss" data-job-id="' + esc(c.jobId) + '" title="목록에서 제외">&#215;</button>' +
          '</div>' +
          '<button type="button" class="jc-btn jc-btn-primary jc-btn-block" data-action="analyze" data-job-id="' + esc(c.jobId) + '">분석하기 &#8594;</button>' +
        '</div>' +
      '</div>'
    );
  }

  root.innerHTML = '<div class="jc-list">' + cards.map(cardHtml).join('') + '</div>';

  root.querySelectorAll('[data-action]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      setTriggerValue('action', { jobId: btn.getAttribute('data-job-id'), type: btn.getAttribute('data-action') });
    });
  });
}
"""

_COMPONENT = st.components.v2.component(
    "jobfit_job_card_list",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def job_list_view(cards: list[dict], *, key: str):
    """cards는 app.py가 이미 계산한 ViewModel(presentation_layer.
    build_job_card_view() + resolve_company_logo() + is_saved())의
    리스트만 담는다 - 이 함수는 그대로 컴포넌트에 전달할 뿐 새 판단을
    만들지 않는다. 반환값의 .action은 CCv2 trigger({jobId, type})라
    rerun 후 자동 리셋되므로 nonce 추적 없이 그대로 if문에 써도
    중복 실행되지 않는다."""
    return _COMPONENT(
        key=key,
        data={"cards": cards},
        on_action_change=lambda: None,
    )
