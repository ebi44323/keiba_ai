"""
formation_view.py — 想定隊列シミュレーター（アニメーション表示）

src/pace_model.py が出した「想定隊列」を、ゲートから1角までの動きとして見せる。
アプリ（Streamlit）と朝刊HTML（完全オフライン）で**同じ部品**を使う。

★設計方針
  - **外部ライブラリ・外部通信ゼロ**（インラインCSS/JSのみ）。朝刊はDL後オフラインで
    見るため、CDNを1つでも使うと競馬場で動かなくなる。
  - **点ではなく帯**。位置の実測ばらつきは SD 0.275（14頭立てで±3.6頭分）あるので、
    馬の後ろにぶれ幅のハローを敷き、確定情報のように見せない。
  - 縦の並びは**馬番順**（上が内枠）＝真上から見た俯瞰。内外の並びも同時に読める。
  - 横は走行位置。ゲートで固まって出て、1角にかけて隊列が縦長に散っていく。

使い方:
    from src.formation_view import build_formation_html, FORMATION_SCRIPT
    # 単体（アプリ）
    html = build_formation_html(rows, autoplay=True, include_script=True)
    # 朝刊のように何レースも並べるとき: スクリプトは1回だけ出す
    head = FORMATION_SCRIPT
    body = "".join(build_formation_html(r, autoplay=False, include_script=False) for r in races)

rows: 各馬 dict のリスト
    馬番, 馬名, 印, mid（0=前〜1=後）, lo, hi, 勝率, zone
"""

import json
import html as _html

# ── 共有スクリプト（朝刊では1回だけ埋め込む）──────────────────────────────
# requestAnimationFrame でゲート→1角の位置を補間する。CSSトランジションではなく
# JSで描くのは、ぶれ幅ハローを進行に合わせて広げたいため（序盤は確定的、
# 隊列が決まるにつれ不確実性が見えてくる、という見せ方）。
FORMATION_SCRIPT = """
<style>
.kbfm{position:relative;background:linear-gradient(180deg,#eef6ee,#e3efe3);border:1px solid #cfe0cf;
 border-radius:10px;padding:8px 10px 6px;margin:8px 0;overflow:hidden}
.kbfm .kbfm-turf{position:absolute;inset:0;background:repeating-linear-gradient(90deg,
 rgba(255,255,255,.35) 0 2px,transparent 2px 46px);pointer-events:none}
.kbfm-hd{display:flex;justify-content:space-between;align-items:center;font-size:11px;
 color:#4a6b4a;margin-bottom:4px;position:relative}
.kbfm-btn{cursor:pointer;border:1px solid #9bbf9b;background:#fff;color:#2f5d2f;border-radius:12px;
 padding:1px 10px;font-size:11px;line-height:1.6}
.kbfm-btn:active{transform:scale(.96)}
.kbfm-lane{position:relative;height:17px;margin:1px 0}
.kbfm-halo{position:absolute;top:2px;height:13px;border-radius:7px;opacity:0;transition:opacity .3s}
.kbfm-h{position:absolute;top:0;height:17px;display:flex;align-items:center;gap:3px;
 white-space:nowrap;transform:translateX(-50%)}
.kbfm-pill{display:inline-flex;align-items:center;justify-content:center;min-width:19px;height:15px;
 border-radius:8px;color:#fff;font-size:10px;font-weight:700;padding:0 3px;
 box-shadow:0 1px 2px rgba(0,0,0,.25)}
.kbfm-nm{font-size:10px;color:#33513a;max-width:82px;overflow:hidden;text-overflow:ellipsis}
.kbfm-goal{position:absolute;top:0;bottom:0;width:0;border-left:2px dashed #b04a4a;opacity:.5}
.kbfm-ft{font-size:10px;color:#6b7d6b;margin-top:3px;position:relative}
@media (prefers-color-scheme:dark){
 .kbfm{background:linear-gradient(180deg,#26332a,#1e2a22);border-color:#3c5240}
 .kbfm-hd,.kbfm-ft{color:#9ec49e}.kbfm-nm{color:#c8dcc8}
 .kbfm-btn{background:#2c3d30;color:#bfe0bf;border-color:#4e6b52}}
</style>
<script>
(function(){
 if(window.__kbfmInit) return; window.__kbfmInit=1;
 function run(el){
  var data=JSON.parse(el.getAttribute('data-h')||'[]'); if(!data.length) return;
  var lanes=el.querySelectorAll('.kbfm-h'), halos=el.querySelectorAll('.kbfm-halo');
  var t0=null, DUR=2600;
  el.classList.add('kbfm-playing');
  function frame(ts){
   if(t0===null) t0=ts;
   var p=Math.min((ts-t0)/DUR,1);
   // ゲートで少し溜めてから伸びる（イーズアウト）
   var e=p<0.12?0:1-Math.pow(1-(p-0.12)/0.88,3);
   for(var i=0;i<data.length;i++){
    var d=data[i];
    // x: 8%（ゲート）→ 最終位置。pos=0(想定先頭)が右端側に来るよう反転
    var fx=20+(1-d.pos)*68, x=8+(fx-8)*e;
    lanes[i].style.left=x+'%';
    // ぶれ幅。枠外にはみ出すと切れて見えるので [0,100] に収める
    var hw=Math.min(d.band*62,46), L=Math.max(0,x-hw/2), R=Math.min(100,x+hw/2);
    halos[i].style.left=L+'%';
    halos[i].style.width=Math.max(R-L,1)+'%';
    halos[i].style.opacity=(e*0.22).toFixed(3);
   }
   if(p<1){requestAnimationFrame(frame);} else {el.classList.remove('kbfm-playing');}
  }
  requestAnimationFrame(frame);
 }
 window.kbfmPlay=function(btn){var el=btn.closest('.kbfm');el.__done=1;run(el);};
 var io=('IntersectionObserver' in window)?new IntersectionObserver(function(es){
   es.forEach(function(x){
     if(x.isIntersecting&&!x.target.__done){x.target.__done=1;run(x.target);io.unobserve(x.target);}
   });},{threshold:.35}):null;
 // 朝刊はカードをJSで作り直すので、描画後に再スキャンできるよう関数を公開する。
 // 画面に入ったレースだけ1回再生（36レース同時再生で重くならないように）。
 window.kbfmScan=function(){
  document.querySelectorAll('.kbfm').forEach(function(el){
   if(el.__seen) return; el.__seen=1;
   if(el.getAttribute('data-auto')==='1'){el.__done=1;run(el);}
   else if(io){io.observe(el);}
  });
 };
 if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',window.kbfmScan);}
 else{window.kbfmScan();}
})();
</script>
"""

_MARK_COLOR = {'◎': '#c0392b', '○': '#2471a3', '〇': '#2471a3', '▲': '#1e8449',
               '△': '#8e6a1f', '☆': '#7d3c98'}


def mark_color(mark: str, win: float) -> str:
    """印とAI勝率から表示色を決める（朝刊HTML側からも使う）。"""
    return _color(mark, win)


def _color(mark: str, win: float) -> str:
    m = (mark or '').strip()
    if m in _MARK_COLOR:
        return _MARK_COLOR[m]
    # 無印はAI勝率が高いほど濃いグレー
    return '#8a9aa5' if win < 0.08 else '#5d6d78'


def build_formation_html(rows: list, race_label: str = '', pace_text: str = '',
                         autoplay: bool = True, include_script: bool = False) -> str:
    """想定隊列アニメーションのHTMLを返す。

    autoplay=True  … 表示と同時に再生（アプリ用・1レースずつ表示するため）
    autoplay=False … 画面に入ったとき1回だけ再生（朝刊用・36レース分の負荷対策）
    include_script … 共有CSS/JSを同梱するか（朝刊では最初の1回だけTrue）
    """
    if not rows:
        return ''
    # ── 横位置は「想定順位」で等間隔に配る ──────────────────────────────
    # mid（回帰の条件付き平均）は中央へ圧縮され、14頭でも 0.26〜0.68 にしか散らない。
    # そのまま描くと全馬が画面中央に団子になって隊列が読めない。
    # 我々が本当に知っているのは *順序* （前走位置率との相関 +0.441）なので、順位で配る。
    # 一方 **ぶれ幅（ハロー）は実測SDのまま**描くので、「順番は目安・かなり入れ替わる」
    # という情報は失われない（ハロー同士が大きく重なるのが正しい絵）。
    valid = []
    for r in rows:
        try:
            m = float(r.get('mid', 0.5))
        except (TypeError, ValueError):
            continue
        valid.append((m, r))
    valid.sort(key=lambda x: x[0])
    n_v = max(len(valid) - 1, 1)
    pct_by_id = {id(r): i / n_v for i, (_, r) in enumerate(valid)}

    data, lanes = [], []
    # 縦は馬番順＝真上から見た内→外
    for r in sorted((r for _, r in valid), key=lambda x: (x.get('馬番') or 0)):
        try:
            mid = float(r.get('mid', 0.5)); lo = float(r.get('lo', mid)); hi = float(r.get('hi', mid))
        except (TypeError, ValueError):
            continue
        pct = pct_by_id.get(id(r), 0.5)
        win = float(r.get('勝率', 0) or 0)
        mark = str(r.get('印', '') or '')
        col = _color(mark, win)
        num = int(r.get('馬番') or 0)
        name = _html.escape(str(r.get('馬名', ''))[:7])
        # pos = 描画位置（順位ベース）/ band = ぶれ幅（実測SDのまま）
        data.append({'pos': round(pct, 4), 'band': round(max(hi - lo, 0.02), 4)})
        lanes.append(
            f'<div class="kbfm-lane">'
            f'<div class="kbfm-halo" style="background:{col}"></div>'
            f'<div class="kbfm-h" style="left:8%">'
            f'<span class="kbfm-pill" style="background:{col}">{num}</span>'
            f'<span class="kbfm-nm">{mark}{name}</span></div></div>')

    head = (f'<span>🐎 想定隊列シミュレーション{" ｜ " + _html.escape(race_label) if race_label else ""}'
            f'</span><span class="kbfm-btn" onclick="kbfmPlay(this)">▶ 再生</span>')
    foot = ('ゲート→1角の想定。<b>薄い帯＝位置のぶれ幅</b>（実測SD±1・14頭立てで±3.6頭分）。'
            '位置は目安で、帯が重なる馬同士は先行争いになりやすい読みです。')
    if pace_text:
        foot = _html.escape(pace_text) + '<br>' + foot
    return ((FORMATION_SCRIPT if include_script else '') +
            f'<div class="kbfm" data-auto="{1 if autoplay else 0}" '
            f'data-h=\'{json.dumps(data, separators=(",", ":"))}\'>'
            f'<div class="kbfm-turf"></div>'
            f'<div class="kbfm-hd">{head}</div>'
            f'<div style="position:relative">'
            f'<div class="kbfm-goal" style="left:90%"></div>'
            + ''.join(lanes) +
            f'</div><div class="kbfm-ft">{foot}</div></div>')


def rows_from_df(df, limit: int = 18) -> list:
    """res_df（inference の戻り）から build_formation_html 用の rows を作る。"""
    need = '想定位置率'
    if df is None or need not in getattr(df, 'columns', []):
        return []
    out = []
    for _, r in df.head(limit).iterrows():
        if r.get(need) is None:
            continue
        try:
            mid = float(r[need])
        except (TypeError, ValueError):
            continue
        out.append({
            '馬番': r.get('馬番'), '馬名': r.get('馬名', ''), '印': r.get('印', ''),
            'mid': mid,
            'lo': float(r.get('想定位置帯lo', mid) or mid),
            'hi': float(r.get('想定位置帯hi', mid) or mid),
            '勝率': float(r.get('勝率(AI予測)', 0) or 0),
            'zone': r.get('想定ゾーン', ''),
        })
    return out
