"""
resume_input/analysis_component.py

STEP3 "공고 분석" 화면 v5(2026-08-16, 사용자 확정 - "디자인보다 먼저
내용 중복을 줄이는 게 맞다") - 판단/한눈에보기/적합성/상세근거/CTA
화면을 st.components.v2.component()로 그린다. 데이터는 전부 app.py가
이미 계산해둔 값(analysis_engine/judge_engine/presentation_layer.build_*_v3)을
그대로 JSON으로 넘기기만 한다 - 이 파일은 새 판단/새 문장을 만들지 않는다,
Engine 결과를 HTML/CSS로 그리는 역할만 한다(Presentation != Explanation
Engine 원칙, presentation_layer.py 상단 docstring과 동일).

v4 -> v5 재개편 이유(사용자 지적, 전문 그대로 요약) - "같은 판단을 표현만
바꿔서 최소 3~4번 반복하는 부분이 있다": 상단 배너의 "직접 경험 부족" ->
나와의 적합성의 "직무 맥락/수준 차이" -> 지원 전 확인 필요의 "근거를
확인하지 못함" -> 지원 전략 "보완할 부분"이 전부 같은 이야기를 이름만
바꿔 반복했다. 이번 버전은 사용자가 위->아래로 읽으며 "① 이 회사/직무는
무엇인가 -> ② 그래서 지원하는 게 좋은가 -> ③ 무엇이 맞고 무엇이
부족한가 -> ④ 왜 그렇게 판단했는가"에 각각 한 번씩만 답을 얻도록
구성한다. 지원 전략(앞세울 경험/보완할 부분) 섹션은 완전히 삭제했다 -
앞세울 경험은 "잘 맞는 부분"+상세 근거와, 보완할 부분은 "지원 전 확인
필요"와 그대로 중복이었다.

CCv2(Custom Components v2)만 쓴다 - v1(components.v1.declare_component,
Streamlit.setComponentValue 등)은 deprecated라 새 코드에 쓰지 않는다
(developing-with-streamlit 공식 skill 확인). 상세 근거 아코디언/탭/행별
펼치기는 전부 순수 클라이언트 사이드 토글(JS)만으로 처리하고, "이력서
맞춤화 시작"/"관심 공고로 저장" 두 개만 setTriggerValue로 Python에
신호를 보낸다 - Python은 그 트리거를 받아 기존 _run_apply_flow/
add_saved_job/remove_saved_job을 그대로 재호출한다(새 로직 없음).

보안: JD 원문/이력서 인용문은 크롤링된 외부 문자열이라 XSS 위험이 있다
- JS에서 innerHTML에 넣기 전 반드시 esc()로 이스케이프한다(아래 JS의
esc() 참고, textContent 왕복으로 escape).
"""

import streamlit as st

_HTML = """
<div id="av-root" class="av-root"></div>
"""

_CSS = """
.av-root {
  font-family: -apple-system, "Segoe UI", Inter, sans-serif;
  color: #0F1F4A;
  background: #FFFFFF;
  max-width: 1080px;
}
.av-root * { box-sizing: border-box; }
.av-section-gap { margin-top: 24px; }
.av-header-row { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
.av-header-left { display:flex; flex-direction:column; gap:6px; min-width:0; }
.av-header-actions { display:flex; align-items:center; gap:8px; flex-shrink:0; }
.av-title { font-size: 22px; font-weight: 700; color:#0F1F4A; margin:0; }
.av-role { font-size: 14px; color:#6B7280; max-width: 760px; margin:0; }
.av-tag-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:2px; }
.av-tag { display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:999px;
  border:1px solid #E5E7EB; font-size:12px; color:#4B5563; background:#FFFFFF; }
.av-tag-label { color:#9AA3B2; }
.av-btn { display:inline-flex; align-items:center; justify-content:center; padding:9px 16px;
  border-radius:10px; font-size:13px; font-weight:500; cursor:pointer; border:1px solid #0F1F4A;
  background:#FFFFFF; color:#0F1F4A; text-decoration:none; white-space:nowrap; }
.av-btn:hover { background:#F5F8FF; }
.av-btn-filled { background:#0F1F4A; color:#FFFFFF; }
.av-btn-filled:hover { background:#1A2C63; }
.av-btn-block { width:100%; padding:14px 16px; font-size:15px; font-weight:600; border-radius:12px; }
.av-btn-saved { background:#0F1F4A; color:#FFFFFF; border-color:#0F1F4A; }
.av-banner { padding:18px 20px; border-radius:16px; }
.av-banner-top { display:flex; align-items:center; gap:16px; }
.av-banner-icon { font-size:20px; font-weight:700; color:#0F1F4A; flex-shrink:0; }
.av-banner-label-col { min-width:96px; display:flex; flex-direction:column; gap:2px; flex-shrink:0; }
.av-banner-label { font-size:12px; color:#4B5563; }
.av-banner-decision { font-size:18px; font-weight:700; }
.av-banner-divider { width:1px; align-self:stretch; background:rgba(15,31,74,0.12); }
.av-banner-body { display:flex; flex-direction:column; gap:4px; min-width:0; }
.av-banner-reason { font-size:14px; color:#4B5563; margin:0; }
.av-banner-action { display:inline-flex; align-self:flex-start; margin-top:2px; padding:3px 10px; border-radius:999px;
  font-size:12px; color:#0F1F4A; font-weight:700; background:rgba(15,31,74,0.08); }
.av-elig-table { margin-top:4px; border-top:1px solid rgba(15,31,74,0.1); padding-top:10px; display:flex; flex-direction:column; gap:6px; }
.av-elig-row { display:grid; grid-template-columns:110px 1fr; gap:8px; font-size:13px; }
.av-elig-key { color:#4B5563; }
.av-elig-val { color:#0F1F4A; font-weight:600; }
.av-section-title { font-size:20px; font-weight:700; color:#0F1F4A; margin:0 0 12px 0; }
.av-grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; align-items:start; }
.av-card { background:#FFFFFF; border:1px solid #E8EDF5; border-radius:16px; padding:18px 20px; }
.av-card-title { font-size:16px; font-weight:600; color:#0F1F4A; margin:0 0 10px 0; }
.av-card-head-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.av-card-desc { font-size:12.5px; color:#6B7280; margin:0 0 10px 0; line-height:1.5; }
.av-card-list { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; }
.av-card-list li { font-size:14px; color:#4B5563; line-height:1.55; }
.av-card-more { font-size:12px; color:#6B7280; margin:6px 0 0 0; }
.av-kw-label { font-size:13px; color:#6B7280; margin:16px 0 8px 0; }
.av-chip-row { display:flex; flex-wrap:wrap; gap:8px; }
.av-chip { padding:6px 12px; border-radius:999px; background:#F5F7FB; color:#0F1F4A; font-size:12px; font-weight:500; }
.av-badge { display:inline-flex; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:500; white-space:nowrap; }
.av-badge-green { background:#F2FFF6; color:#1F8F5E; }
.av-badge-orange { background:#FFFBEA; color:#B9702E; }
.av-badge-red { background:#FBE3E5; color:#E0555F; }
.av-badge-gray { background:#F5F7FB; color:#4B5563; }
.av-detail-btn-wrap { display:flex; justify-content:center; margin:20px 0; }
.av-detail-section { display:none; }
.av-detail-section.is-open { display:block; }
.av-detail-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:10px; }
.av-detail-note { font-size:12px; color:#6B7280; line-height:1.5; margin:0; max-width:420px; }
.av-tabs { display:flex; gap:8px; border-bottom:1px solid #E8EDF5; margin-bottom:8px; }
.av-tab { padding:8px 16px; border-radius:8px 8px 0 0; font-size:13px; color:#6B7280; cursor:pointer; background:transparent; border:none; }
.av-tab.is-active { background:#0F1F4A; color:#FFFFFF; font-weight:600; }
.av-tab-panel { display:none; }
.av-tab-panel.is-active { display:block; }
.av-row { display:flex; align-items:center; gap:16px; padding:16px 4px; border-bottom:1px solid #EEF2F7; cursor:pointer; }
.av-row-num { width:20px; font-size:14px; color:#6B7280; flex-shrink:0; }
.av-row-title-col { flex: 2 1 0; min-width:0; display:flex; flex-direction:column; align-items:flex-start; gap:6px; }
.av-row-title { font-size:15px; font-weight:600; color:#0F1F4A; }
.av-row-evidence { width:110px; font-size:13px; color:#6B7280; flex-shrink:0; }
.av-row-judgement { width:120px; flex-shrink:0; }
.av-row-chevron { width:20px; text-align:center; font-size:12px; color:#6B7280; flex-shrink:0; transition:transform .15s; }
.av-row.is-open .av-row-chevron { transform:rotate(180deg); }
.av-row-detail { display:none; padding:16px; margin:-1px 0 8px 0; background:#F8FAFC; border-radius:8px; }
.av-row-detail.is-open { display:block; }
.av-cmp-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.av-cmp-col h5 { font-size:12px; font-weight:600; color:#6B7280; margin:0 0 6px 0; text-transform:none; }
.av-cmp-col p { font-size:13.5px; color:#0F1F4A; line-height:1.5; margin:0 0 4px 0; }
.av-cmp-col .av-cmp-tag { font-size:11px; color:#6B7280; font-weight:600; margin-right:4px; }
.av-cmp-empty { color:#9AA3B2; }
.av-footer-col { display:flex; flex-direction:column; gap:6px; align-items:center; }
.av-footer-caption { font-size:12px; color:#6B7280; text-align:center; margin:0; }
.av-footer-decline-note { font-size:12.5px; color:#B9702E; text-align:center; margin:0 0 2px 0; }
@media (max-width: 860px) {
  .av-grid3 { grid-template-columns:1fr; }
  .av-cmp-grid { grid-template-columns:1fr; }
  .av-row { flex-wrap:wrap; }
  .av-header-row { flex-direction:column; }
}
"""

_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector('#av-root');
  if (!root || !data) return;

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = String(s ?? '');
    return d.innerHTML;
  }
  function badgeHtml(label, tone) {
    return '<span class="av-badge av-badge-' + esc(tone || 'gray') + '">' + esc(label) + '</span>';
  }
  function cardListHtml(items) {
    return '<ul class="av-card-list">' + (items || []).map(function (t) {
      return '<li>&bull; ' + esc(t) + '</li>';
    }).join('') + '</ul>';
  }
  function fitListHtml(items) {
    return '<ul class="av-card-list">' + (items || []).map(function (t) {
      return '<li>&#10003; ' + esc(t && t.title) + '</li>';
    }).join('') + '</ul>';
  }

  const overview = data.overview || {};
  const fit = data.fit || {};
  const detailGroups = data.detailGroups || {};
  const hardElig = data.hardEligibility || [];

  // ── 헤더: 회사/직무명, 역할 한 줄 소개, 도메인/직무영역 태그(좌) +
  // 원문보기/관심공고 저장(우, 상단 고정) ─────────────────────────
  const tagsHtml =
    (data.domain || data.roleArea) ?
    '<div class="av-tag-row">' +
      (data.domain ? '<span class="av-tag"><span class="av-tag-label">도메인</span>' + esc(data.domain) + '</span>' : '') +
      (data.roleArea ? '<span class="av-tag"><span class="av-tag-label">직무영역</span>' + esc(data.roleArea) + '</span>' : '') +
    '</div>' : '';
  const headerHtml =
    '<div class="av-header-row">' +
      '<div class="av-header-left">' +
        '<div class="av-title">' + esc(data.company) + '</div>' +
        '<div class="av-role">' + esc(data.roleIntro) + '</div>' +
        tagsHtml +
      '</div>' +
      '<div class="av-header-actions">' +
        (data.url ? '<a class="av-btn" href="' + esc(data.url) + '" target="_blank" rel="noopener noreferrer">공고 원문 보기 &#8599;</a>' : '') +
        '<button type="button" class="av-btn' + (data.saved ? ' av-btn-saved' : '') + '" id="av-btn-save">' +
          (data.saved ? '&#9733; 저장됨' : '&#9734; 관심 공고 저장') + '</button>' +
      '</div>' +
    '</div>';

  // ── 지원 판단 배너: 이유는 딱 1개만. Hard Eligibility 미충족이면
  // 일반 사유 대신 요구/이력서 확인 비교 표를 보여준다(우선순위) ─────
  const eligHtml = hardElig.length ?
    '<div class="av-elig-table">' + hardElig.map(function (e) {
      return '<div class="av-elig-row"><span class="av-elig-key">' + esc(e.requirement) + '</span>' +
        '<span class="av-elig-val">요구 ' + esc(e.required) + (e.actual ? ' / 이력서 확인 ' + esc(e.actual) : '') + '</span></div>';
    }).join('') + '</div>' : '';
  const bannerHtml =
    '<div class="av-banner av-section-gap" style="background:' + esc(data.bannerBg) + ';">' +
      '<div class="av-banner-top">' +
        '<div class="av-banner-icon">!</div>' +
        '<div class="av-banner-label-col">' +
          '<div class="av-banner-label">지원 판단</div>' +
          '<div class="av-banner-decision" style="color:' + esc(data.bannerFg) + ';">' + esc(data.decision) + '</div>' +
        '</div>' +
        '<div class="av-banner-divider"></div>' +
        '<div class="av-banner-body">' +
          '<p class="av-banner-reason">' + esc(data.reason) + '</p>' +
          (data.action ? '<p class="av-banner-action">' + esc(data.action) + '</p>' : '') +
        '</div>' +
      '</div>' +
      eligHtml +
    '</div>';

  // ── 공고 한눈에 보기: JD Understanding 전용, 이력서 비교는 섞지 않음 ──
  const ovCards = [
    ['핵심 업무', overview.tasks],
    ['중요 역량', overview.keyCompetencies],
    ['주요 업무 영역', overview.focusAreas],
  ];
  const overviewHtml =
    '<div class="av-section-gap">' +
      '<h4 class="av-section-title">공고 한눈에 보기</h4>' +
      '<div class="av-grid3">' +
        ovCards.map(function (c) {
          return '<div class="av-card"><div class="av-card-title">' + esc(c[0]) + '</div>' + cardListHtml(c[1]) + '</div>';
        }).join('') +
      '</div>' +
      (overview.keywords && overview.keywords.length ?
        '<div class="av-kw-label">키워드</div><div class="av-chip-row">' +
          overview.keywords.map(function (k) { return '<span class="av-chip">' + esc(k) + '</span>'; }).join('') +
        '</div>' : '') +
    '</div>';

  // ── 나와의 적합성: "잘 맞는 부분/관련 경험은 있음/지원 전 확인 필요"
  // 세 카드는 서로 다른 의미를 갖는다 - 설명 문구도 카드마다 고정.
  // 카드당 최대 4개, 넘으면 "+N개 더"만 표시(전체는 상세 근거에서). ──
  const fitCardsDef = [
    ['strong', '잘 맞는 부분', 'green', '요구사항을 보여줄 이력서 근거가 충분합니다.'],
    ['partial', '관련 경험은 있음', 'orange', '유사한 경험은 있지만 직무 맥락이나 수준에 차이가 있습니다.'],
    ['risk', '지원 전 확인 필요', 'red', '현재 이력서에서는 직접적인 근거를 확인하지 못했습니다.'],
  ];
  const FIT_CARD_CAP = 4;
  const fitHtml =
    '<div class="av-section-gap">' +
      '<h4 class="av-section-title">나와의 적합성</h4>' +
      '<div class="av-grid3">' +
        fitCardsDef.map(function (d) {
          const key = d[0], title = d[1], tone = d[2], desc = d[3];
          const group = fit[key] || { count: 0, items: [] };
          const shown = (group.items || []).slice(0, FIT_CARD_CAP);
          const rest = group.count - shown.length;
          return '<div class="av-card">' +
            '<div class="av-card-head-row"><div class="av-card-title" style="margin:0;">' + esc(title) + '</div>' +
            badgeHtml(String(group.count) + '개', tone) + '</div>' +
            '<p class="av-card-desc">' + esc(desc) + '</p>' +
            fitListHtml(shown) +
            (rest > 0 ? '<p class="av-card-more">+' + rest + '개 더</p>' : '') +
            '</div>';
        }).join('') +
      '</div>' +
    '</div>';

  // ── 상세 근거: 기본 숨김. 탭(그룹)별 행 목록 -> 클릭한 행만 3열
  // 비교(JD 요구 | 내 이력서 근거 | 연결된 요소/차이/확인사항)로 펼침.
  // "확인되지 않음 = 경험 없음 아님" 안내는 섹션 전체에 한 번만. ──────
  const tabDefs = [
    ['strong', '잘 맞는 부분', 'green'],
    ['partial', '관련 경험은 있음', 'orange'],
    ['risk', '지원 전 확인 필요', 'red'],
  ];
  function cmpColHtml(label, bodyHtml) {
    return '<div class="av-cmp-col"><h5>' + esc(label) + '</h5>' + bodyHtml + '</div>';
  }
  function resumeEvidenceHtml(r) {
    if (!(r.resumeEvidence || []).length) {
      return '<p class="av-cmp-empty">이력서 근거 없음</p>';
    }
    return r.resumeEvidence.map(function (e) {
      return '<p>' + (e.project ? '<span class="av-cmp-tag">' + esc(e.project) + '</span>' : '') + '&ldquo;' + esc(e.text) + '&rdquo;</p>';
    }).join('');
  }
  function missingHtml(r, emptyText) {
    const list = r.missingList || [];
    if (!list.length) return '<p class="av-cmp-empty">' + esc(emptyText) + '</p>';
    return list.map(function (m) { return '<p>' + esc(m.text) + '</p>'; }).join('');
  }
  function rowDetailHtml(r, groupKey) {
    const jdCol = cmpColHtml('JD 요구', '<p>' + (r.jdEvidence ? '&ldquo;' + esc(r.jdEvidence) + '&rdquo;' : '원문 근거를 찾지 못했습니다.') + '</p>');
    let col2, col3;
    if (groupKey === 'strong') {
      col2 = cmpColHtml('내 이력서에서 확인된 근거', resumeEvidenceHtml(r));
      const dims = (r.matchedDims && r.matchedDims.length) ? '<p>' + r.matchedDims.map(esc).join(', ') + '</p>' : '<p class="av-cmp-empty">-</p>';
      col3 = cmpColHtml('연결된 요소', dims);
    } else if (groupKey === 'partial') {
      col2 = cmpColHtml('내 경험', resumeEvidenceHtml(r));
      col3 = cmpColHtml('차이 / 확인사항', missingHtml(r, '-'));
    } else {
      col2 = cmpColHtml('현재 확인 상태', '<p class="av-cmp-empty">이력서 근거 없음</p>');
      col3 = cmpColHtml('확인하면 좋은 경험', missingHtml(r, '추가로 확인할 항목이 없습니다.'));
    }
    return '<div class="av-row-detail" data-detail-for="' + esc(r.id) + '"><div class="av-cmp-grid">' + jdCol + col2 + col3 + '</div></div>';
  }
  function rowHtml(r, groupKey, groupLabel, groupTone, isOpen) {
    return '<div>' +
      '<div class="av-row' + (isOpen ? ' is-open' : '') + '" data-row-id="' + esc(r.id) + '">' +
        '<div class="av-row-num">' + r.n + '</div>' +
        '<div class="av-row-title-col"><div class="av-row-title">' + esc(r.title) + '</div>' + badgeHtml(r.importanceBadge, 'gray') + '</div>' +
        '<div class="av-row-evidence">' + esc(r.evidenceLabel) + '</div>' +
        '<div class="av-row-judgement">' + badgeHtml(groupLabel, groupTone) + '</div>' +
        '<div class="av-row-chevron">&#9660;</div>' +
      '</div>' +
      rowDetailHtml(r, groupKey).replace('av-row-detail"', 'av-row-detail' + (isOpen ? ' is-open' : '') + '"') +
    '</div>';
  }
  const detailTotal = data.detailTotal || 0;
  const detailHtml =
    '<div class="av-detail-btn-wrap">' +
      '<button type="button" class="av-btn" id="av-detail-toggle">상세 근거 보기 (' + detailTotal + '개) &#9662;</button>' +
    '</div>' +
    '<div class="av-detail-section" id="av-detail-section">' +
      '<div class="av-detail-head">' +
        '<h4 class="av-section-title" style="margin:0;">상세 근거 (' + detailTotal + '개)</h4>' +
        '<p class="av-detail-note">&#8251; &lsquo;확인되지 않음&rsquo;은 경험이 없다는 의미가 아니라, 현재 이력서에서 판단 가능한 근거를 찾지 못했다는 의미입니다.</p>' +
      '</div>' +
      '<div class="av-tabs">' +
        tabDefs.map(function (t, i) {
          const count = (detailGroups[t[0]] || []).length;
          return '<button type="button" class="av-tab' + (i === 0 ? ' is-active' : '') + '" data-tab="' + t[0] + '">' + t[1] + ' ' + count + '</button>';
        }).join('') +
      '</div>' +
      tabDefs.map(function (t, i) {
        const rows = detailGroups[t[0]] || [];
        const body = rows.length
          ? rows.map(function (r, ri) { return rowHtml(r, t[0], t[1], t[2], i === 0 && ri === 0); }).join('')
          : '<p class="av-cmp-empty" style="padding:16px 4px;">해당하는 항목이 없습니다.</p>';
        return '<div class="av-tab-panel' + (i === 0 ? ' is-active' : '') + '" data-tab-panel="' + t[0] + '">' + body + '</div>';
      }).join('') +
    '</div>';

  // ── 최종 CTA: 이력서 맞춤화 시작 하나만. 관심 공고 저장은 상단으로 이동 ──
  // 2026-08-16(사용자 확정 - "비추천이어도 체크박스 게이트를 다시 넣지
  // 않는다, 이 서비스는 지원을 막는 시스템이 아니라 판단을 돕는
  // 시스템") - CTA는 decision과 무관하게 항상 눌린다. 비추천일 때만
  // CTA 위에 한 줄 더 붙이되, 이유는 위 배너("비추천 - 필수 조건
  // 미충족" 등)가 이미 설명하므로 여기서 그 이유를 다시 반복하지
  // 않는다(2026-08-16 사용자 지적 - "새로운 설명을 또 하나 만드는
  // 느낌이 되면 안 된다") - CTA를 눌러도 된다는 사실 하나만 짧게.
  const declineNoteHtml = data.decision === '비추천'
    ? '<p class="av-footer-decline-note">비추천 공고도 이력서 수정 제안은 확인할 수 있습니다.</p>'
    : '';
  const footerHtml =
    '<div class="av-section-gap av-footer-col">' +
      declineNoteHtml +
      '<button type="button" class="av-btn av-btn-filled av-btn-block" id="av-btn-apply">이력서 맞춤화 시작 &#8594;</button>' +
      '<p class="av-footer-caption">이 공고의 요구사항에 맞춰 이력서 수정 제안을 확인합니다.</p>' +
    '</div>';

  root.innerHTML = headerHtml + bannerHtml + overviewHtml + fitHtml + detailHtml + footerHtml;

  const detailToggle = root.querySelector('#av-detail-toggle');
  const detailSection = root.querySelector('#av-detail-section');
  if (detailToggle && detailSection) {
    detailToggle.addEventListener('click', function () {
      const open = detailSection.classList.toggle('is-open');
      detailToggle.innerHTML = open ? '상세 근거 접기 &#9652;' : '상세 근거 보기 (' + detailTotal + '개) &#9662;';
    });
  }
  root.querySelectorAll('.av-tab').forEach(function (tabBtn) {
    tabBtn.addEventListener('click', function () {
      const key = tabBtn.getAttribute('data-tab');
      root.querySelectorAll('.av-tab').forEach(function (b) { b.classList.toggle('is-active', b === tabBtn); });
      root.querySelectorAll('.av-tab-panel').forEach(function (p) {
        p.classList.toggle('is-active', p.getAttribute('data-tab-panel') === key);
      });
    });
  });
  root.querySelectorAll('.av-row').forEach(function (rowEl) {
    rowEl.addEventListener('click', function () {
      const id = rowEl.getAttribute('data-row-id');
      const detailEl = root.querySelector('.av-row-detail[data-detail-for="' + CSS.escape(id) + '"]');
      const open = rowEl.classList.toggle('is-open');
      if (detailEl) detailEl.classList.toggle('is-open', open);
    });
  });
  const applyBtn = root.querySelector('#av-btn-apply');
  if (applyBtn) applyBtn.addEventListener('click', function () { setTriggerValue('apply', true); });
  const saveBtn = root.querySelector('#av-btn-save');
  if (saveBtn) saveBtn.addEventListener('click', function () { setTriggerValue('save', true); });
}
"""

_COMPONENT = st.components.v2.component(
    "jobfit_analysis_view",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def analysis_view(payload: dict, *, key: str):
    """payload는 app.py가 이미 계산한 값(analysis_engine/judge_engine/
    presentation_layer.build_*_v3)만 담는다 - 이 함수는 그대로 컴포넌트에
    전달할 뿐 새 판단을 만들지 않는다. 반환값의 .apply/.save는 CCv2
    trigger(1회성, rerun 후 자동 리셋)라 nonce 추적 없이 그대로 if문에
    써도 중복 실행되지 않는다."""
    return _COMPONENT(
        key=key,
        data=payload,
        on_apply_change=lambda: None,
        on_save_change=lambda: None,
    )
