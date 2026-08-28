import json
import pandas as pd
import logging
logger = logging.getLogger('keiba_ebye')


def _build_gemini_html(r: dict) -> str:
    """GeminiアナリストコメントがあればHTML断片を返す。なければ空文字。（旧txt/後方互換用）"""
    h = r.get('gemini_honmei', {})
    a = r.get('gemini_ana', {})
    model = r.get('gemini_model', '')
    if not h and not a:
        return ""
    hc = h.get('comment', '')
    hb = h.get('bet', '')
    ac = a.get('comment', '')
    ab = a.get('bet', '')
    return f"""
<div class="gemini-block">
  <div class="gemini-title">🤖 AI思考モード（{model}）</div>
  <div class="gemini-row">
    <div class="gemini-honmei">
      <div class="gemini-analyst">🎯 本命党「伊藤ホンメ」</div>
      <p>{hc}</p>
      <div class="gemini-bet">💰 買い目: {hb}</div>
    </div>
    <div class="gemini-ana">
      <div class="gemini-analyst">💣 穴党「風穴あけるズ」</div>
      <p>{ac}</p>
      <div class="gemini-bet">🎰 買い目: {ab}</div>
    </div>
  </div>
</div>"""


# ════════════════════════════════════════════════════════════════════════
#  朝刊HTML（スマホ最適・オフライン自己完結・並べ替え/絞り込み）  2026-08-29
#  __DATA__ に races_data(JSON)、__DATE__ に日付を差し込む。外部通信ゼロ。
# ════════════════════════════════════════════════════════════════════════
_MORNING_TEMPLATE = """<title>keiba-ebye 朝刊 __DATE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root{
  --bg:#eceef2; --card:#ffffff; --ink:#171a24; --muted:#616b7b;
  --line:#e3e6ec; --line2:#eef0f4; --accent:#7a1f3d; --accent-soft:#f3e7ec;
  --fire:#c0392b; --fire-bg:#fbe9e7; --calm:#2f6fb0; --calm-bg:#e8f1fb;
  --warn:#b0791b; --warn-bg:#fbf1dc; --money:#1a7a4c; --money-bg:#e6f4ec;
  --bar:#c9d2de; --bar-hi:#7a1f3d;
  --shadow:0 1px 2px rgba(20,25,40,.06),0 4px 14px rgba(20,25,40,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12141b; --card:#1b1e28; --ink:#e9ecf3; --muted:#99a2b2;
  --line:#2a2e3b; --line2:#232733; --accent:#e79bb2; --accent-soft:#33202a;
  --fire:#ff7a6b; --fire-bg:#3a2320; --calm:#7db4ec; --calm-bg:#1f2a38;
  --warn:#e3b25a; --warn-bg:#332a17; --money:#5cc88c; --money-bg:#182a20;
  --bar:#333a49; --bar-hi:#e79bb2;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 18px rgba(0,0,0,.28);
}}
:root[data-theme="dark"]{
  --bg:#12141b; --card:#1b1e28; --ink:#e9ecf3; --muted:#99a2b2;
  --line:#2a2e3b; --line2:#232733; --accent:#e79bb2; --accent-soft:#33202a;
  --fire:#ff7a6b; --fire-bg:#3a2320; --calm:#7db4ec; --calm-bg:#1f2a38;
  --warn:#e3b25a; --warn-bg:#332a17; --money:#5cc88c; --money-bg:#182a20;
  --bar:#333a49; --bar-hi:#e79bb2;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 18px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);
  font-family:'Hiragino Kaku Gothic ProN','Hiragino Sans','Noto Sans JP','Meiryo',system-ui,sans-serif;
  font-size:15px;line-height:1.5;-webkit-text-size-adjust:100%;padding-bottom:40px;}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}
.wrap{max-width:680px;margin:0 auto;padding:0 12px}
header.top{position:sticky;top:0;z-index:20;background:var(--bg);padding:10px 0 6px;
  border-bottom:1px solid var(--line);box-shadow:0 6px 12px -12px rgba(20,25,40,.5);}
.brand{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.brand h1{font-size:19px;margin:0;letter-spacing:.01em;font-weight:800}
.brand .date{font-size:12.5px;color:var(--muted);font-weight:600}
.brand .themebtn{margin-left:auto;border:1px solid var(--line);background:var(--card);
  color:var(--muted);border-radius:8px;padding:4px 9px;font-size:13px;cursor:pointer}
.controls{margin-top:8px;display:flex;flex-direction:column;gap:7px}
.ctl-row{display:flex;align-items:center;gap:7px;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}
.ctl-row::-webkit-scrollbar{display:none}
.ctl-lab{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;flex:0 0 auto;width:34px}
.chip{flex:0 0 auto;border:1px solid var(--line);background:var(--card);color:var(--ink);
  border-radius:999px;padding:6px 12px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;
  transition:background .12s,border-color .12s,color .12s}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.chip.fire[aria-pressed="true"]{background:var(--fire);border-color:var(--fire);color:#fff}
.count{color:var(--muted);font-size:12px;font-weight:600;padding-left:4px;flex:0 0 auto}
.summary{background:linear-gradient(180deg,var(--accent-soft),var(--card));border:1px solid var(--line);
  border-radius:14px;box-shadow:var(--shadow);padding:12px 13px;margin-top:12px}
.summary h2{margin:0 0 9px;font-size:13px;letter-spacing:.02em;display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.summary .s-counts{margin-left:auto;font-size:11px;color:var(--muted);font-weight:700}
.s-list{display:flex;flex-direction:column;gap:7px}
.s-item{display:flex;align-items:center;gap:8px;font-size:13px}
.s-item .s-mk{font-weight:800;width:18px;text-align:center;color:var(--accent)}
.s-item .s-rc{color:var(--muted);font-weight:700;font-size:11px;flex:0 0 auto;width:60px}
.s-item .s-nm{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.s-item .s-ev{margin-left:auto;font-weight:800;color:var(--money);font-size:12.5px;flex:0 0 auto}
main{margin-top:12px;display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);overflow:hidden}
.card.hidden{display:none}
.chead{display:flex;align-items:center;gap:9px;padding:11px 13px 8px;position:relative}
.chead:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--calm)}
.card[data-conf="勝負"] .chead:before{background:var(--fire)}
.card[data-conf="回避"] .chead:before{background:var(--warn)}
.venue{font-weight:800;font-size:16px;letter-spacing:.01em}
.rno{font-weight:800;font-size:16px;color:var(--accent)}
.meta{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12.5px;font-weight:600}
.meta .dot{width:3px;height:3px;border-radius:50%;background:var(--muted);opacity:.5}
.badge{margin-left:auto;flex:0 0 auto;font-size:11.5px;font-weight:800;padding:4px 9px;border-radius:999px;letter-spacing:.02em}
.badge.勝負{background:var(--fire-bg);color:var(--fire)}
.badge.通常{background:var(--calm-bg);color:var(--calm)}
.badge.回避{background:var(--warn-bg);color:var(--warn)}
.rtitle{padding:0 13px 4px;font-size:12px;color:var(--muted);font-weight:600}
.pace{padding:0 13px 9px;color:var(--muted);font-size:12px;line-height:1.45}
.hlist{border-top:1px solid var(--line2)}
.hrow{display:grid;grid-template-columns:26px 1fr auto;gap:9px;align-items:center;padding:8px 13px;border-top:1px solid var(--line2)}
.hrow:first-child{border-top:none}
.hrow.extra{display:none}
.card.open .hrow.extra{display:grid}
.mk{font-size:16px;font-weight:800;text-align:center;line-height:1}
.hmid{min-width:0}
.hname{font-weight:700;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;gap:5px}
.tag{flex:0 0 auto;font-size:10px;font-weight:700;padding:1px 5px;border-radius:5px;background:var(--accent-soft);color:var(--accent)}
.tag.d{background:var(--warn-bg);color:var(--warn)}
.sub{display:flex;align-items:center;gap:7px;margin-top:3px}
.style{font-size:11px;color:var(--muted);font-weight:600;flex:0 0 auto}
.barwrap{position:relative;height:7px;background:var(--bar);border-radius:4px;flex:1 1 auto;min-width:44px;max-width:150px;overflow:hidden}
.bar{position:absolute;left:0;top:0;bottom:0;background:var(--bar-hi);border-radius:4px}
.wr{font-size:11.5px;color:var(--muted);font-weight:700;flex:0 0 auto}
.hend{text-align:right;flex:0 0 auto}
.odds{font-size:13px;font-weight:700}
.odds small{color:var(--muted);font-weight:600;font-size:10px}
.evchip{display:inline-block;margin-top:3px;font-size:11.5px;font-weight:800;padding:2px 7px;border-radius:6px;background:var(--line2);color:var(--muted)}
.evchip.hot{background:var(--money-bg);color:var(--money)}
.fukline{font-size:10.5px;color:var(--muted);font-weight:600;margin-top:2px}
.dchip{display:inline-block;font-size:10px;font-weight:700;padding:1px 6px;border-radius:5px;background:var(--warn-bg);color:var(--warn)}
.cfoot{display:flex;flex-direction:column;gap:8px;padding:10px 13px 12px;border-top:1px solid var(--line2)}
.toggle{align-self:flex-start;border:1px solid var(--line);background:var(--card);color:var(--muted);border-radius:8px;padding:5px 11px;font-size:12.5px;font-weight:600;cursor:pointer}
.reason{font-size:11.5px;color:var(--muted);line-height:1.45}
.reason b{color:var(--ink);font-weight:700}
.memoblock{background:var(--line2);border-radius:9px;padding:8px 10px;font-size:11.5px;line-height:1.5}
.memoblock .mh{font-weight:800;color:var(--accent)}
.memoblock .mt{color:var(--muted);font-weight:600}
.reco{background:var(--accent-soft);border-radius:9px;padding:9px 11px;font-size:12.5px;line-height:1.5;color:var(--ink)}
.reco b{color:var(--accent)}
.gemini{display:flex;flex-direction:column;gap:6px;background:var(--calm-bg);border-radius:9px;padding:9px 11px}
.gtitle{font-size:11.5px;font-weight:800;color:var(--calm)}
.gcard{font-size:12px;line-height:1.5}
.gcard b{color:var(--ink)}
.gbet{font-size:11px;color:var(--muted);font-weight:700;margin-top:2px}
.empty{text-align:center;color:var(--muted);padding:34px 12px;font-size:14px}
footer{max-width:680px;margin:20px auto 0;padding:0 14px;color:var(--muted);font-size:11px;line-height:1.5}
@media (min-width:760px){ main{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start} }
</style>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <h1>🐴 keiba-ebye 朝刊</h1>
      <span class="date num">__DATE__</span>
      <button class="themebtn" id="themebtn" type="button">◐ 表示</button>
    </div>
    <div class="controls">
      <div class="ctl-row" id="sortRow">
        <span class="ctl-lab">並替</span>
        <button class="chip" data-sort="venue" aria-pressed="true" type="button">競馬場順</button>
        <button class="chip" data-sort="time" aria-pressed="false" type="button">発走時刻</button>
        <button class="chip" data-sort="conf" aria-pressed="false" type="button">信頼度</button>
        <button class="chip" data-sort="ev" aria-pressed="false" type="button">最高EV</button>
      </div>
      <div class="ctl-row" id="filterRow">
        <span class="ctl-lab">絞込</span>
        <button class="chip" data-filter="all" aria-pressed="true" type="button">すべて</button>
        <button class="chip fire" data-filter="勝負" aria-pressed="false" type="button">🔥 勝負のみ</button>
        <button class="chip" data-filter="ev" aria-pressed="false" type="button">EV1.5+ 含む</button>
        <span class="count num" id="count"></span>
      </div>
    </div>
  </header>
  <section id="summary" class="summary"></section>
  <main id="main"></main>
  <div class="empty" id="empty" style="display:none">該当するレースがありません</div>
</div>
<footer>
  複勝率＝キャリブレーション済み（勝率＜複勝率を保証）。AI勝率バーは頭内相対。数値は参考情報で的中を保証しません。<br>
  keiba-ebye — オフライン表示OK / スマホ最適
</footer>
<script>
const R = __DATA__;
const CONF_ORDER={"勝負":0,"通常":1,"回避":2};
const fmtPct=x=>(x||0).toFixed(1)+"%";
let sort="venue", filter="all";
function topEV(rc){return rc.horses.length?Math.max.apply(null,rc.horses.map(h=>h.ev||0)):0;}
function maxWin(rc){return rc.horses.length?Math.max(1,Math.max.apply(null,rc.horses.map(h=>h.w||0))):1;}
function isDanger(rc,h){
  const top2=rc.horses.map(x=>x.w).sort((a,b)=>b-a).slice(0,2);
  return h.o<=4.0 && h.w<10 && top2.indexOf(h.w)<0;
}
function buildVenueChips(){
  const seen=[]; R.forEach(r=>{if(seen.indexOf(r.v)<0)seen.push(r.v);});
  const row=document.getElementById("filterRow"), cnt=document.getElementById("count");
  seen.forEach(v=>{
    const b=document.createElement("button");
    b.className="chip"; b.type="button"; b.dataset.filter=v;
    b.setAttribute("aria-pressed","false"); b.textContent=v;
    row.insertBefore(b,cnt);
  });
}
function buildSummary(){
  const el=document.getElementById("summary");
  const picks=[];
  R.forEach(rc=>rc.horses.forEach(h=>picks.push({rc:rc,mk:h.mk,nm:h.nm,ev:h.ev||0})));
  picks.sort((a,b)=>b.ev-a.ev);
  const top=picks.slice(0,5);
  const nFire=R.filter(r=>r.conf==="勝負").length, nAvoid=R.filter(r=>r.conf==="回避").length;
  el.innerHTML='<h2>🎯 本日のベスト <span class="s-counts">🔥勝負 '+nFire+' ・ ⚠️回避 '+nAvoid+' ・ 全'+R.length+'R</span></h2>'
    +'<div class="s-list">'+top.map(function(p){return '<div class="s-item"><span class="s-mk">'+(p.mk||'‥')+'</span>'
      +'<span class="s-rc">'+p.rc.v+p.rc.r+'R</span><span class="s-nm">'+p.nm+'</span>'
      +'<span class="s-ev num">EV '+p.ev.toFixed(2)+'</span></div>';}).join('')+'</div>';
}
function build(){
  const main=document.getElementById("main");
  main.innerHTML="";
  let list=R.slice();
  list=list.filter(function(rc){
    if(filter==="all")return true;
    if(filter==="勝負")return rc.conf==="勝負";
    if(filter==="ev")return topEV(rc)>=1.5;
    return rc.v===filter;
  });
  list.sort(function(a,b){
    if(sort==="venue")return a.v===b.v ? a.r-b.r : String(a.v).localeCompare(String(b.v),"ja");
    if(sort==="time")return String(a.t).localeCompare(String(b.t)) || a.r-b.r;
    if(sort==="conf")return CONF_ORDER[a.conf]-CONF_ORDER[b.conf] || String(a.t).localeCompare(String(b.t));
    if(sort==="ev")return topEV(b)-topEV(a);
    return 0;
  });
  document.getElementById("count").textContent=list.length+" レース";
  document.getElementById("empty").style.display=list.length?"none":"block";
  for(const rc of list){
    const mw=maxWin(rc);
    const card=document.createElement("article");
    card.className="card"; card.dataset.conf=rc.conf;
    const rows=rc.horses.map(function(h,i){
      const place=h.p, barW=Math.max(4,Math.min(100,h.w/mw*100));
      const evHot=h.ev>=1.5, danger=isDanger(rc,h);
      const tags=(h.ana?'<span class="tag d">🎯穴</span>':'')+(h.memo?'<span class="tag">📝メモ</span>':'');
      return '<div class="hrow '+(i>=5?'extra':'')+'">'
        +'<div class="mk">'+(h.mk||'<span style="visibility:hidden">•</span>')+'</div>'
        +'<div class="hmid"><div class="hname">'+h.nm+' '+tags+'</div>'
        +'<div class="sub"><span class="style">'+(h.st||'')+'・'+h.no+'番</span>'
        +'<span class="barwrap"><span class="bar" style="width:'+barW+'%"></span></span>'
        +'<span class="wr num">'+fmtPct(h.w)+'</span></div>'
        +'<div class="fukline num">複勝率 '+fmtPct(place)+(danger?' <span class="dchip">⚠️人気だがAI低評価</span>':'')+'</div></div>'
        +'<div class="hend"><div class="odds num">'+(h.o||0).toFixed(1)+'<small>倍</small></div>'
        +'<span class="evchip num '+(evHot?'hot':'')+'">EV '+(h.ev||0).toFixed(2)+'</span></div></div>';
    }).join("");
    const memoHtml=(rc.memos||[]).map(function(m){
      return '<div class="memoblock">📝 <span class="mh">'+m.nm+'</span> <span class="mt">'+m.tag+'・'+m.dt+'</span>'+(m.tx?('<br>'+m.tx):'')+'</div>';
    }).join("");
    const g=rc.gemini;
    const gemHtml=g?('<div class="gemini"><div class="gtitle">🤖 AI思考モード '+(g.model?('('+g.model+')'):'')+'</div>'
      +((g.h&&g.h.c)?('<div class="gcard"><b>🎯 本命党</b> '+g.h.c+(g.h.b?('<div class="gbet">💰 '+g.h.b+'</div>'):'')+'</div>'):'')
      +((g.a&&g.a.c)?('<div class="gcard"><b>💣 穴党</b> '+g.a.c+(g.a.b?('<div class="gbet">🎰 '+g.a.b+'</div>'):'')+'</div>'):'')
      +'</div>'):'';
    const extra=rc.horses.length>5;
    const distLine=(rc.track||'')+(rc.dist?rc.dist+'m':'');
    card.innerHTML=
      '<div class="chead"><span class="venue">'+rc.v+'</span><span class="rno num">'+rc.r+'R</span>'
      +'<span class="meta">'+(rc.t?('<span class="num">'+rc.t+'</span><span class="dot"></span>'):'')+'<span>'+distLine+'</span></span>'
      +'<span class="badge '+rc.conf+'">'+(rc.conf==="勝負"?"🔥 勝負":rc.conf==="回避"?"⚠️ 回避":"🟡 通常")+'</span></div>'
      +(rc.title?('<div class="rtitle">'+rc.title+'</div>'):'')
      +(rc.pace?('<div class="pace">'+rc.pace+'</div>'):'')
      +'<div class="hlist">'+rows+'</div>'
      +'<div class="cfoot">'
      +(extra?('<button class="toggle" type="button">＋ 全'+rc.horses.length+'頭を表示</button>'):'')
      +((rc.reason&&rc.reason!=='—')?('<div class="reason">🧠 <b>◎の根拠</b>：'+rc.reason+'</div>'):'')
      +memoHtml
      +'<div class="reco">💰 <b>買い目</b>：'+(rc.reco||'—')+'</div>'
      +gemHtml
      +'</div>';
    if(extra){
      const btn=card.querySelector(".toggle");
      btn.addEventListener("click",function(){
        card.classList.toggle("open");
        btn.textContent=card.classList.contains("open")?"− 上位のみ表示":("＋ 全"+rc.horses.length+"頭を表示");
      });
    }
    main.appendChild(card);
  }
}
function wireChips(rowId,set){
  document.getElementById(rowId).addEventListener("click",function(e){
    const b=e.target.closest(".chip"); if(!b)return;
    const chips=document.querySelectorAll("#"+rowId+" .chip");
    for(const c of chips)c.setAttribute("aria-pressed","false");
    b.setAttribute("aria-pressed","true"); set(b.dataset.sort||b.dataset.filter); build();
  });
}
wireChips("sortRow",function(v){sort=v;});
wireChips("filterRow",function(v){filter=v;});
document.getElementById("themebtn").addEventListener("click",function(){
  const r=document.documentElement, cur=r.getAttribute("data-theme");
  const dark=cur?cur==="dark":matchMedia("(prefers-color-scheme:dark)").matches;
  r.setAttribute("data-theme",dark?"light":"dark");
});
buildVenueChips(); buildSummary(); build();
</script>"""


def _race_confidence_category(conf: str) -> str:
    """confidence_text から 勝負/回避/通常 を判定。"""
    first = (conf or '').split('\n')[0]
    if '勝負' in first or '鉄板' in (conf or ''):
        return '勝負'
    if '回避' in first:
        return '回避'
    return '通常'


def _race_reason(topics: list) -> str:
    """topics の「AIの推し理由」を1行テキストに整形。"""
    for t in (topics or []):
        if 'AIの推し理由' in t:
            return (t.replace('AIの推し理由:', '').replace('AIの推し理由', '')
                     .replace('\n', '　').replace('**', '').strip())
    return ''


def generate_pdf_report(results_list, ev_threshold=1.5, all_memos: dict = None):
    """
    朝刊予想をスマホ最適・オフライン自己完結HTML（bytes）で返す。
    - 外部通信ゼロ（インラインCSS/JS）→ ダウンロードすればオフラインで閲覧可
    - 並べ替え（競馬場/発走時刻/信頼度/最高EV）・絞り込み（勝負/EV/競馬場）をJSで実行
    - 各カード: ◎〇▲…をAI勝率バー付き表示・全頭折りたたみ・買い目/根拠/馬券メモ/危険人気馬
    ※ 複勝率は inference 側でキャリブレーション済み（勝率<複勝率を保証）の値をそのまま使用。
    """
    if all_memos is None:
        all_memos = {}
    try:
        races_data = []
        for r in results_list:
            df = r.get('df')
            if df is None or getattr(df, 'empty', True):
                continue
            has_ana = '穴馬マーク' in df.columns
            horses, memos = [], []
            for _, row in df.iterrows():
                name = str(row.get('馬名', ''))
                has_memo = name in all_memos
                try:
                    no = int(float(row.get('馬番', 0) or 0))
                except Exception:
                    no = 0
                horses.append({
                    'mk': str(row.get('印', '') or ''),
                    'no': no,
                    'nm': name,
                    'st': str(row.get('脚質カテゴリ', '') or '').replace('nan', ''),
                    'o':  round(float(row.get('単勝オッズ', 0) or 0), 1),
                    'w':  round(float(row.get('勝率(AI予測)', 0) or 0) * 100, 1),
                    'p':  round(float(row.get('複勝率(AI予測)', 0) or 0) * 100, 1),
                    'ev': round(float(row.get('期待値', 0) or 0), 2),
                    'fev': round(float(row.get('複勝期待値', 0) or 0), 2),
                    'ana': 1 if (has_ana and str(row.get('穴馬マーク', '')) == '🎯') else 0,
                    'memo': 1 if has_memo else 0,
                })
                if has_memo:
                    latest = sorted(all_memos[name], key=lambda x: x.get('日付', ''), reverse=True)[0]
                    memos.append({'nm': name, 'tag': latest.get('タグ', ''),
                                  'dt': latest.get('日付', ''), 'tx': latest.get('メモ', '') or ''})
            # 買い目（AIの推し理由以降は除去）
            reco_raw = r.get('reco', '') or ''
            if 'AIの推し理由' in reco_raw:
                reco_raw = reco_raw[:reco_raw.index('AIの推し理由')].strip()
            reco = reco_raw.replace('**', '').replace('\n', ' ').strip()
            # 距離を整数文字列に
            try:
                dist_s = str(int(float(r.get('dist', 0))))
            except Exception:
                dist_s = str(r.get('dist', '') or '')
            gh, ga = r.get('gemini_honmei'), r.get('gemini_ana')
            gem = None
            if gh or ga:
                gem = {'model': r.get('gemini_model', ''),
                       'h': {'c': (gh or {}).get('comment', ''), 'b': (gh or {}).get('bet', '')},
                       'a': {'c': (ga or {}).get('comment', ''), 'b': (ga or {}).get('bet', '')}}
            races_data.append({
                'v': r.get('place', ''), 'r': r.get('num', ''),
                't': r.get('time', '') or '', 'title': r.get('title', '') or '',
                'track': r.get('track', '') or '', 'dist': dist_s,
                'conf': _race_confidence_category(r.get('confidence', '')),
                'pace': (r.get('pace', '') or '').replace('**', ''),
                'reason': _race_reason(r.get('topics', [])),
                'reco': reco, 'horses': horses, 'memos': memos, 'gemini': gem,
            })

        date_str = results_list[0].get('date', '') if results_list else ''
        html = (_MORNING_TEMPLATE
                .replace('__DATA__', json.dumps(races_data, ensure_ascii=False))
                .replace('__DATE__', str(date_str)))
        return html.encode('utf-8')
    except Exception as e:
        logger.warning(f'generate_pdf_report error: {e}')
        return None


def generate_txt_report(results_list, ev_threshold=1.5, all_memos: dict = None):
    """noteに貼りやすい・読みやすいプレーンテキスト形式のレポートを生成"""
    if all_memos is None:
        all_memos = {}
    out = []
    ev_summary = []

    date_str = results_list[0].get("date", "") if results_list else ""
    out.append(f"🏇 keiba-ebye AI予想レポート  {date_str}")
    out.append("keiba-ebye（ebi × AI × Eye）によるAI競馬予想")
    out.append("")

    for r in results_list:
        place = r.get("place", "")
        num   = r.get("num", "")
        track = r.get("track", "")
        dist  = r.get("dist", "")
        conf  = r.get("confidence", "")
        pace  = r.get("pace", "")

        out.append("=" * 44)
        out.append(f"■ {place} {num}R  {track}{dist}m")
        if conf: out.append(conf)
        out.append("=" * 44)

        if pace:
            out.append(f"[展開予想] {pace}")
            out.append("")

        # 予想表（固定幅テキスト）
        has_ana = '穴馬マーク' in r["df"].columns
        if has_ana:
            out.append(f"{'印':<3} {'穴':<2} {'馬番':>3} {'馬名':<12} {'脚質':<5} {'オッズ':>7} {'勝率':>6} {'複勝率':>7} {'EV':>5} {'複EV':>5}")
            out.append("-" * 67)
        else:
            out.append(f"{'印':<3} {'馬番':>3} {'馬名':<12} {'脚質':<5} {'オッズ':>7} {'勝率':>6} {'複勝率':>7} {'EV':>5} {'複EV':>5}")
            out.append("-" * 63)
        for rank, row in r["df"].iterrows():
            try:
                imp  = str(row["印"] or "").ljust(2)
                num_ = int(row["馬番"])
                name = str(row["馬名"])[:10]
                stle = str(row.get("脚質カテゴリ", ""))[:4]
                odds = float(row["単勝オッズ"])
                wp   = row["勝率(AI予測)"] * 100
                fp   = row["複勝率(AI予測)"] * 100
                ev   = float(row.get("期待値", 0) or 0)
                fev  = float(row.get("複勝期待値", 0) or 0)
                ev_str = f"{ev:4.2f}"
                fev_str = f"{fev:4.2f}"
                mark = " ★" if ev >= ev_threshold else ""
                if has_ana:
                    ana = str(row.get("穴馬マーク", "") or "")
                    out.append(f"{imp:<3} {ana:<2} {num_:>3} {name:<12} {stle:<5} {odds:>6.1f}倍 {wp:>5.1f}% {fp:>6.1f}% {ev_str} {fev_str}{mark}")
                else:
                    out.append(f"{imp:<3} {num_:>3} {name:<12} {stle:<5} {odds:>6.1f}倍 {wp:>5.1f}% {fp:>6.1f}% {ev_str} {fev_str}{mark}")
                if rank < 5 and ev >= ev_threshold:
                    ev_summary.append({"レース": f"{place}{num}R", "印": row["印"],
                                       "馬番": num_, "馬名": row["馬名"],
                                       "EV": ev, "勝率": f"{wp:.1f}%", "オッズ": odds})
            except Exception as _e:
                logger.debug(f'generate_txt_report行処理スキップ: {_e}')
        sep = "-" * (60 if has_ana else 56)
        out.append(sep)
        out.append("★ = 期待値" + str(ev_threshold) + "以上の注目馬" + ("  🎯 = 穴馬マーク" if has_ana else ""))
        out.append("")

        # AIアナリスト（Gemini）コメント
        _gh = r.get('gemini_honmei', {})
        _ga = r.get('gemini_ana', {})
        if _gh or _ga:
            out.append(f"[🤖 AI思考モード ({r.get('gemini_model','')})]")
            if _gh.get('comment'):
                out.append(f"  🎯 伊藤ホンメ（本命党）:")
                out.append(f"    {_gh['comment']}")
            if _gh.get('bet'):
                out.append(f"    💰 買い目: {_gh['bet']}")
            out.append("")
            if _ga.get('comment'):
                out.append(f"  💣 風穴あけるズ（穴党）:")
                out.append(f"    {_ga['comment']}")
            if _ga.get('bet'):
                out.append(f"    🎰 買い目: {_ga['bet']}")
            out.append("")

        # 注目トピック
        if r.get("topics"):
            out.append("[注目トピック]")
            for t in r["topics"]:
                clean_t = t.replace("**", "")
                out.append(f"  {clean_t}")
            out.append("")

        # 推奨買い目
        reco_raw = r.get("reco", "")
        if "AIの推し理由" in reco_raw:
            reco_raw = reco_raw[:reco_raw.index("AIの推し理由")].strip()
        if reco_raw:
            out.append("[推奨買い目]")
            out.append(f"  {reco_raw.strip()}")
            out.append("")

        # 馬券メモ
        memo_lines = []
        for _, row in r["df"].iterrows():
            name = str(row.get("馬名", ""))
            if name in all_memos:
                imp = str(row.get("印", "")).strip()
                latest = sorted(all_memos[name], key=lambda x: x.get("日付", ""), reverse=True)[0]
                tag    = latest.get("タグ", "")
                date_  = latest.get("日付", "")
                memo_t = latest.get("メモ", "")
                memo_lines.append(
                    f"  📝 {name}（{imp or '印なし'}）: {tag} {date_}"
                    + (f" — {memo_t}" if memo_t else "")
                )
        if memo_lines:
            out.append("[馬券メモあり]")
            out.extend(memo_lines)
            out.append("")

    # 横断EV注目馬まとめ
    if ev_summary:
        out.append("")
        out.append("=" * 44)
        out.append(f"💰 本日の注目馬（期待値{ev_threshold}以上・印あり）")
        out.append("=" * 44)
        for row in sorted(ev_summary, key=lambda x: x["EV"], reverse=True):
            out.append(f"  {row['印']} {row['レース']} {row['馬番']}番 {row['馬名']}"
                       f"  EV:{row['EV']:.2f}  オッズ:{row['オッズ']}倍  勝率:{row['勝率']}")
        out.append("")

    out.append("=" * 44)
    out.append("keiba-ebye / 予想はAI参考情報です。馬券は自己責任でお願いします。")
    return "\n".join(out)
