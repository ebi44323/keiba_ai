"""
formation_view.py — レースシミュレーター（競馬中継風アニメーション）

ゲート → 想定隊列 → 直線の追い比べ → ゴール までを走らせて見せる。
アプリ（Streamlit）と朝刊HTML（完全オフライン）で**同じ部品**を使う。

★設計方針
  - **外部ライブラリ・外部通信ゼロ**（インラインCSS/JS）。朝刊はDL後オフラインで
    見るため、CDNを1つでも使うと競馬場で動かなくなる。
  - **着順は毎回 AI勝率にもとづく抽選**（Plackett-Luce）。同じレースでも再生のたびに
    結果が変わる。これは演出であると同時に**正直さの担保**でもある:
    勝率20%の馬は10回まわせば約2回しか勝たない、が体感で分かる。
    「AIが選んだ1着」を毎回同じに見せると、確率予測を確定予想のように誤解させてしまう。
  - **道中の位置は pace_model の想定隊列**（前走位置率との相関 +0.441・実測）。
    直線での動きは着順への補間であり、そこは演出であることを画面にも明記する。

使い方:
    from src.formation_view import build_race_sim_html, RACE_SIM_ASSETS
    # 単体（アプリ）
    html = build_race_sim_html(rows, distance=1600, autoplay=True, include_assets=True)
    # 朝刊のように何レースも並べるとき: アセットは1回だけ
    head = RACE_SIM_ASSETS
    body = "".join(build_race_sim_html(r, include_assets=False) for r in races)

rows: 各馬 dict のリスト
    馬番, 馬名, 印, mid（0=前〜1=後・想定隊列）, 勝率
"""

import json
import html as _html

_MARK_COLOR = {'◎': '#c0392b', '○': '#2471a3', '〇': '#2471a3', '▲': '#1e8449',
               '△': '#8e6a1f', '☆': '#7d3c98'}


def mark_color(mark: str, win: float = 0.0) -> str:
    """印とAI勝率から表示色を決める（朝刊HTML側からも使う）。"""
    m = (mark or '').strip()
    if m in _MARK_COLOR:
        return _MARK_COLOR[m]
    return '#8a9aa5' if (win or 0) < 0.08 else '#5d6d78'


# ── 共有アセット（朝刊では1回だけ埋め込む）──────────────────────────────
RACE_SIM_ASSETS = """
<style>
.kbrs{position:relative;background:linear-gradient(180deg,#dff0e0,#cfe6d2 62%,#b9d8be);
 border:1px solid #bcd6bf;border-radius:10px;padding:8px 10px 6px;margin:8px 0;overflow:hidden}
.kbrs-hd{display:flex;align-items:center;gap:8px;font-size:11px;color:#31543a;margin-bottom:5px}
.kbrs-rem{font-weight:800;font-size:15px;color:#1d4427;font-variant-numeric:tabular-nums;
 background:rgba(255,255,255,.7);border-radius:6px;padding:0 7px}
.kbrs-btn{margin-left:auto;cursor:pointer;border:1px solid #8fb894;background:#fff;color:#245c2f;
 border-radius:12px;padding:1px 11px;font-size:11px;line-height:1.7;white-space:nowrap}
.kbrs-btn:active{transform:scale(.96)}
.kbrs-track{position:relative;border-radius:6px;overflow:hidden;
 background:repeating-linear-gradient(90deg,rgba(255,255,255,.28) 0 2px,transparent 2px 52px)}
.kbrs-goal{position:absolute;top:0;bottom:0;left:94%;width:5px;
 background:repeating-linear-gradient(180deg,#fff 0 5px,#111 5px 10px);opacity:.85}
.kbrs-lane{position:relative;height:16px;margin:1px 0}
.kbrs-h{position:absolute;top:0;height:16px;display:flex;align-items:center;gap:3px;
 white-space:nowrap;transform:translateX(-100%);will-change:left}
.kbrs-pill{display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:14px;
 border-radius:7px;color:#fff;font-size:10px;font-weight:800;padding:0 3px;
 box-shadow:0 1px 2px rgba(0,0,0,.28)}
.kbrs-nm{font-size:10px;color:#2c4a33;max-width:74px;overflow:hidden;text-overflow:ellipsis}
.kbrs-ord{font-size:9.5px;font-weight:800;color:#fff;background:#1d4427;border-radius:6px;
 padding:0 4px;opacity:0;transition:opacity .25s}
.kbrs-podium{margin-top:6px;display:flex;flex-direction:column;gap:3px;min-height:18px}
.kbrs-pod{display:flex;align-items:center;gap:7px;background:rgba(255,255,255,.78);
 border-radius:7px;padding:3px 8px;opacity:0;transform:translateY(5px);
 transition:opacity .32s,transform .32s}
.kbrs-pod.show{opacity:1;transform:none}
.kbrs-pod .rk{font-weight:900;font-size:14px;color:#1d4427;width:34px;flex:0 0 auto}
.kbrs-pod.p1 .rk{color:#b8860b;font-size:16px}
.kbrs-pod .bn{display:inline-flex;align-items:center;justify-content:center;min-width:22px;
 height:18px;border-radius:5px;color:#fff;font-weight:800;font-size:12px;flex:0 0 auto}
.kbrs-pod .hn{font-weight:800;font-size:14px;color:#1b3323;flex:1 1 auto;min-width:0;
 overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kbrs-pod.p1 .hn{font-size:15.5px}
.kbrs-pod .od{font-weight:800;font-size:13px;color:#8a4b12;flex:0 0 auto;
 font-variant-numeric:tabular-nums}
.kbrs-pay{font-size:12px;font-weight:800;color:#1a7a4c;margin-top:3px;opacity:0;
 transition:opacity .35s}
.kbrs-pay.show{opacity:1}
.kbrs-res{font-size:10.5px;color:#2c4a33;margin-top:3px;min-height:13px;font-weight:700}
.kbrs-ft{font-size:9.5px;color:#5c7361;margin-top:2px;line-height:1.45}
@media (prefers-color-scheme:dark){
 .kbrs-pod{background:rgba(0,0,0,.34)}
 .kbrs-pod .rk{color:#bfe0bf}.kbrs-pod.p1 .rk{color:#e8c46a}
 .kbrs-pod .hn{color:#dcecdf}.kbrs-pod .od{color:#e8b77a}.kbrs-pay{color:#5cc88c}}
@media (prefers-color-scheme:dark){
 .kbrs{background:linear-gradient(180deg,#22322a,#1a2720 62%,#152019);border-color:#3a5240}
 .kbrs-hd,.kbrs-res{color:#a8cdad}.kbrs-nm{color:#bcd8c0}.kbrs-ft{color:#7e9a84}
 .kbrs-rem{color:#d6f0d9;background:rgba(0,0,0,.35)}
 .kbrs-btn{background:#2a3d30;color:#bfe0bf;border-color:#4c6a52}
 .kbrs-ord{background:#bfe0bf;color:#1a2720}}
</style>
<script>
(function(){
 if(window.__kbrsInit) return; window.__kbrsInit=1;
 var EASE=function(x){return 1-Math.pow(1-x,3);};

 // AI勝率を重みにした Plackett-Luce 抽選。1着の出現率が勝率どおりになる。
 function sampleOrder(ws){
  var idx=ws.map(function(_,i){return i;}), w=ws.slice(), out=[];
  while(idx.length){
   var s=0,i; for(i=0;i<idx.length;i++) s+=w[idx[i]];
   var r=Math.random()*s, a=0, pick=idx.length-1;
   for(i=0;i<idx.length;i++){a+=w[idx[i]]; if(r<=a){pick=i;break;}}
   out.push(idx[pick]); idx.splice(pick,1);
  }
  return out; // out[k] = k着の馬インデックス
 }

 function build(el){
  var hs=JSON.parse(el.getAttribute('data-h')||'[]');
  if(!hs.length){el.style.display='none';return null;}
  var dist=parseInt(el.getAttribute('data-dist')||'0',10)||1600;
  var lanes=hs.map(function(h){
   return '<div class="kbrs-lane"><div class="kbrs-h" style="left:6%">'
    +'<span class="kbrs-ord"></span>'
    +'<span class="kbrs-pill" style="background:'+h.c+'">'+h.no+'</span>'
    +'<span class="kbrs-nm">'+h.mk+h.nm+'</span></div></div>';
  }).join('');
  el.innerHTML='<div class="kbrs-hd"><span>🏇 レースシミュレーション</span>'
   +'<span class="kbrs-rem">残り'+dist+'m</span>'
   +'<span class="kbrs-btn">▶ 出走</span></div>'
   +'<div class="kbrs-track"><div class="kbrs-goal"></div>'+lanes+'</div>'
   +'<div class="kbrs-podium">'
   +'<div class="kbrs-pod p1"><span class="rk">1着</span><span class="bn"></span>'
   +'<span class="hn"></span><span class="od"></span></div>'
   +'<div class="kbrs-pod p2"><span class="rk">2着</span><span class="bn"></span>'
   +'<span class="hn"></span><span class="od"></span></div>'
   +'<div class="kbrs-pod p3"><span class="rk">3着</span><span class="bn"></span>'
   +'<span class="hn"></span><span class="od"></span></div></div>'
   +'<div class="kbrs-pay"></div>'
   +'<div class="kbrs-res"></div>'
   +'<div class="kbrs-ft">道中の位置は想定隊列（実測にもとづく）、着順は<b>AI勝率による抽選</b>です。'
   +'再生するたび結果が変わります＝それが確率予想の実際の姿です。</div>';
  var st={hs:hs,dist:dist,el:el,wins:{},runs:0,
   lanes:el.querySelectorAll('.kbrs-h'),ords:el.querySelectorAll('.kbrs-ord'),
   pods:el.querySelectorAll('.kbrs-pod'),pay:el.querySelector('.kbrs-pay'),
   rem:el.querySelector('.kbrs-rem'),res:el.querySelector('.kbrs-res'),
   btn:el.querySelector('.kbrs-btn')};
  st.btn.addEventListener('click',function(){play(st);});
  el.__st=st; return st;
 }

 function play(st){
  if(st.running) return; st.running=true; st.runs++;
  var hs=st.hs,n=hs.length,i,k;
  var order=sampleOrder(hs.map(function(h){return Math.max(h.w,0.001);}));
  var place=new Array(n); order.forEach(function(hi,kk){place[hi]=kk;});

  // ── 最終x: 前は詰まって後ろほど離れる（実際の入線＝ハナ差の叩き合い＋後方は大差）──
  var gaps=[0],acc=0;
  for(k=1;k<n;k++){ acc+=1.0+k*0.22; gaps.push(acc); }
  var scale=Math.min(58/Math.max(acc,1),1.6);
  var xf=hs.map(function(_,idx){return 94-gaps[place[idx]]*scale;});

  // ── 道中x: 想定隊列（e: 0=前）を 24〜80% に展開 ──
  var xm=hs.map(function(h){return 24+(1-h.e)*56;});

  // ── 直線の仕掛け: 後ろの馬ほど遅く動き出して鋭く伸びる（差し・追込の見せ場）──
  //    毎回わずかに乱数を混ぜるので、同じ着順でも運び方が変わる。
  var kick=hs.map(function(h){return 0.60+h.e*0.13+Math.random()*0.04;});
  var sharp=hs.map(function(h){return 1.7+h.e*1.5;});   // 後方馬ほど加速が鋭い

  st.ords.forEach(function(o){o.style.opacity=0;o.textContent='';});
  st.pods.forEach(function(p){p.classList.remove('show');});
  st.pay.classList.remove('show'); st.res.textContent='';
  var t0=null,DUR=8200;
  function frame(ts){
   if(t0===null)t0=ts;
   var p=Math.min((ts-t0)/DUR,1);
   st.rem.textContent='残り'+(Math.ceil(st.dist*(1-p)/50)*50)+'m';
   for(i=0;i<n;i++){
    var x;
    if(p<0.14){ x=6+(xm[i]-6)*EASE(p/0.14); }               // ゲート→隊列形成
    else if(p<kick[i]){                                      // 道中（息を入れる・小さな出入り）
     x=xm[i]+Math.sin((p-0.14)*9+i*1.7)*0.6;
    }else{                                                   // 直線: 仕掛けてから鋭く伸びる
     var q=(p-kick[i])/(1-kick[i]);
     q=Math.pow(q,1.0)*0.25+Math.pow(q,sharp[i])*0.75;       // 序盤ゆるく→末脚で一気に
     x=xm[i]+(xf[i]-xm[i])*q;
    }
    st.lanes[i].style.left=x+'%';
   }
   if(p<1){requestAnimationFrame(frame);}
   else{finish(st,order);}
  }
  requestAnimationFrame(frame);
 }

 function finish(st,order){
  var hs=st.hs,n=hs.length,k;
  for(k=0;k<Math.min(3,n);k++){
   var hi=order[k];
   st.ords[hi].textContent=(k+1)+'着'; st.ords[hi].style.opacity=1;
   var pod=st.pods[k],h=hs[hi];
   pod.querySelector('.bn').textContent=h.no;
   pod.querySelector('.bn').style.background=h.c;
   pod.querySelector('.hn').textContent=(h.mk?h.mk+' ':'')+h.nm;
   pod.querySelector('.od').textContent=(h.o>0?h.o.toFixed(1)+'倍':'—');
   (function(el,d){setTimeout(function(){el.classList.add('show');},d);})(pod,k*220);
  }
  for(k=n;k<3;k++){ if(st.pods[k]) st.pods[k].style.display='none'; }
  var w=hs[order[0]];
  if(w.o>0){
   st.pay.textContent='💰 単勝 '+Math.round(w.o*100).toLocaleString()+'円'
     +(hs[order[1]]&&hs[order[1]].o>0?'　／　馬連の目安 '+w.no+'-'+hs[order[1]].no:'');
   setTimeout(function(){st.pay.classList.add('show');},700);
  }
  st.wins[w.no]=(st.wins[w.no]||0)+1;
  var tally=Object.keys(st.wins).sort(function(a,b){return st.wins[b]-st.wins[a];})
    .slice(0,3).map(function(kk){return kk+'番 '+st.wins[kk]+'回';}).join(' / ');
  st.res.textContent='この馬のAI勝率 '+(w.w*100).toFixed(1)+'%　▶ '+st.runs+'回中: '+tally;
  st.btn.textContent='▶ もう一度';
  st.running=false;
 }

 var io=('IntersectionObserver' in window)?new IntersectionObserver(function(es){
   es.forEach(function(x){
    if(x.isIntersecting&&!x.target.__done){x.target.__done=1;
     if(x.target.__st)play(x.target.__st); io.unobserve(x.target);}
   });},{threshold:.4}):null;

 // 朝刊はカードをJSで作り直すので、描画後に再スキャンできるよう公開する。
 window.kbrsScan=function(){
  document.querySelectorAll('.kbrs').forEach(function(el){
   if(el.__seen)return; el.__seen=1;
   var st=build(el); if(!st)return;
   if(el.getAttribute('data-auto')==='1'){el.__done=1;play(st);}
   else if(io){io.observe(el);}
  });
 };
 if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',window.kbrsScan);}
 else{window.kbrsScan();}
})();
</script>
"""


def sim_rows_from_df(df, limit: int = 18) -> list:
    """res_df（inference の戻り）からシミュレーター用の行データを作る。"""
    if df is None or '想定位置率' not in getattr(df, 'columns', []):
        return []
    out = []
    for _, r in df.head(limit).iterrows():
        try:
            mid = float(r['想定位置率'])
        except (TypeError, ValueError):
            continue
        win = float(r.get('勝率(AI予測)', 0) or 0)
        mark = str(r.get('印', '') or '')
        try:
            no = int(float(r.get('馬番', 0) or 0))
        except (TypeError, ValueError):
            no = 0
        try:
            odds = round(float(r.get('単勝オッズ', 0) or 0), 1)
        except (TypeError, ValueError):
            odds = 0.0
        out.append({'no': no, 'nm': str(r.get('馬名', ''))[:9], 'mk': mark,
                    'c': mark_color(mark, win), 'w': round(win, 4),
                    'e': round(mid, 4), 'o': odds})
    # 縦は馬番順＝上が内枠
    return sorted(out, key=lambda x: x['no'])


def build_race_sim_html(rows: list, distance=1600, autoplay: bool = True,
                        include_assets: bool = False) -> str:
    """レースシミュレーターのHTMLを返す。

    autoplay=True  … 表示と同時に出走（アプリ用・1レースずつ表示するため）
    autoplay=False … 画面に入ったとき1回だけ出走（朝刊用・36レース分の負荷対策）
    include_assets … 共有CSS/JSを同梱するか（朝刊では最初の1回だけTrue）
    """
    if not rows:
        return ''
    try:
        d = int(float(distance or 1600))
    except (TypeError, ValueError):
        d = 1600
    safe = [{'no': r['no'], 'nm': _html.escape(str(r['nm'])), 'mk': _html.escape(str(r['mk'])),
             'c': r['c'], 'w': r['w'], 'e': r['e'], 'o': r.get('o', 0)} for r in rows]
    return ((RACE_SIM_ASSETS if include_assets else '') +
            f'<div class="kbrs" data-auto="{1 if autoplay else 0}" data-dist="{d}" '
            f'data-h=\'{json.dumps(safe, ensure_ascii=False, separators=(",", ":"))}\'></div>')
