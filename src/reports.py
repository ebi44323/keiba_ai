import pandas as pd
import logging
logger = logging.getLogger('keiba_ebye')


def _build_gemini_html(r: dict) -> str:
    """GeminiアナリストコメントがあればHTML断片を返す。なければ空文字。"""
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


def _build_ana_detail_html(df) -> str:
    """穴馬マーク付き馬の詳細HTMLブロック。なければ空文字。"""
    if '穴馬マーク' not in df.columns:
        return ""
    ana_horses = df[df['穴馬マーク'] == '🎯']
    if ana_horses.empty:
        return ""
    rows_html = ""
    for _, ar in ana_horses.iterrows():
        try:
            score = float(ar.get('穴馬スコア', 0) or 0)
            odds  = float(ar.get('単勝オッズ', 0) or 0)
            prob  = float(ar.get('勝率(AI予測)', 0) or 0) * 100
            ev    = float(ar.get('期待値', 0) or 0)
            imp   = str(ar.get('印', ''))
            name  = str(ar.get('馬名', ''))
            ev_class = ' style="color:#cc0000;font-weight:bold"' if ev >= 1.5 else ''
            rows_html += f"""<div class="ana-horse">
              🎯 <strong>{name}</strong>（{imp}）&nbsp;
              オッズ <strong>{odds:.1f}倍</strong> /
              AI勝率 {prob:.1f}% /
              EV <span{ev_class}>{ev:.2f}</span> /
              穴馬スコア {score:.3f}
            </div>"""
        except Exception:
            pass
    return f"""<div class="ana-section">
  <div class="ana-title">🎯 穴馬マーク付き馬（人気薄激走パターン適合）</div>
  {rows_html}
</div>"""


def _build_topics_html(topics: list) -> str:
    """注目トピックがあればHTMLリストを返す。"""
    if not topics:
        return ""
    items = "".join(f"<li>{t.replace('**', '')}</li>" for t in topics)
    return f'<ul class="topics">{items}</ul>'


def generate_pdf_report(results_list, ev_threshold=1.5):
    """
    予想レポートをHTML形式で生成してbytesを返す。
    ブラウザで開いてCtrl+P(Cmd+P)で印刷・PDF保存できる。
    reportlabのフォント問題を回避するためHTML方式に変更。
    """
    try:
        ev_summary = []
        races_html = ""
        for r in results_list:
            df = r['df']
            has_ana = '穴馬マーク' in df.columns
            rows_html = ""
            for rank, row in df.iterrows():
                try:
                    ev_val = float(row.get('期待値', 0) or 0)
                except Exception:
                    ev_val = 0.0
                bg = ' style="background:#fff0f0"' if ev_val >= ev_threshold else (' style="background:#fffde7"' if rank == 0 else '')
                ev_color = ' color:#cc0000;font-weight:bold' if ev_val >= ev_threshold else ''
                ana_mark = str(row.get('穴馬マーク', '')) if has_ana else ''
                ana_td   = f'<td style="color:#e65100">{ana_mark}</td>' if has_ana else ''
                rows_html += f"""<tr{bg}>
                  <td>{row.get('印','')}</td>
                  {ana_td}
                  <td>{int(float(row.get('馬番',0)))}</td>
                  <td style="text-align:left">{row.get('馬名','')}</td>
                  <td>{row.get('脚質カテゴリ','')}</td>
                  <td>{float(row.get('単勝オッズ',0)):.1f}</td>
                  <td>{float(row.get('勝率(AI予測)',0))*100:.1f}%</td>
                  <td>{float(row.get('複勝率(AI予測)',0))*100:.1f}%</td>
                  <td style="{ev_color}">{ev_val:.2f}</td>
                </tr>"""
                if rank < 5 and ev_val >= ev_threshold:
                    ev_summary.append({'レース': f"{r['place']}{r['num']}R", '印': row.get('印',''), '馬名': row.get('馬名',''), 'EV': ev_val})

            ana_th = '<th>穴</th>' if has_ana else ''
            # 信頼度に応じた背景色
            conf_text = r.get('confidence', '')
            if '鉄板' in conf_text:
                conf_cls = 'conf-solid'
            elif '波乱' in conf_text:
                conf_cls = 'conf-upset'
            else:
                conf_cls = 'conf-normal'

            # 推奨買い目: "AIの推し理由"以降を除去
            reco_raw = r.get('reco', '')
            if 'AIの推し理由' in reco_raw:
                reco_raw = reco_raw[:reco_raw.index('AIの推し理由')].strip()
            reco_lines = reco_raw.replace('\n', '<br>') if reco_raw else ''

            # Geminiは11Rのみ
            gemini_html = _build_gemini_html(r) if str(r.get('num', '')) == '11' else ''

            races_html += f"""
            <div class="race-block">
              <h3>■ {r['place']} {r['num']}R &nbsp;<span class="track-badge">{r['track']}{r['dist']}m</span></h3>
              <p class="{conf_cls}">{conf_text}</p>
              <p class="pace">📐 展開: {r.get('pace','')}</p>
              {_build_topics_html(r.get('topics', []))}
              <table>
                <thead><tr><th>印</th>{ana_th}<th>馬番</th><th>馬名</th><th>脚質</th><th>オッズ</th><th>勝率</th><th>複勝率</th><th>EV</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
              {_build_ana_detail_html(df)}
              <div class="reco">💰 推奨: {reco_lines}</div>
              {gemini_html}
            </div>"""

        ev_html = ""
        if ev_summary:
            ev_rows = "".join(f"<tr><td>{r['印']}</td><td>{r['レース']}</td><td>{r['馬名']}</td><td style='color:#cc0000;font-weight:bold'>{r['EV']:.2f}</td></tr>"
                             for r in sorted(ev_summary, key=lambda x: x['EV'], reverse=True))
            ev_html = f"""<div class="ev-summary">
              <h3>💰 本日の注目馬 (EV{ev_threshold}以上・上位5頭内)</h3>
              <table><thead><tr><th>印</th><th>レース</th><th>馬名</th><th>EV</th></tr></thead>
              <tbody>{ev_rows}</tbody></table></div>"""

        date_str = results_list[0]['date'] if results_list else ''
        html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>keiba-ebye 予想レポート {date_str}</title>
<style>
  body {{ font-family: 'Hiragino Kaku Gothic ProN','Noto Sans JP','Meiryo',sans-serif; font-size:11px; margin:10px; color:#222; }}
  h1 {{ font-size:16px; border-bottom:2px solid #2c3e7a; padding-bottom:4px; color:#2c3e7a; }}
  h3 {{ font-size:13px; margin:14px 0 4px; color:#2c3e7a; border-left:3px solid #2c3e7a; padding-left:6px; }}
  .track-badge {{ font-size:10px; background:#e8eaf6; color:#3949ab; padding:1px 6px; border-radius:10px; font-weight:normal; }}
  .race-block {{ margin-bottom:20px; page-break-inside:avoid; border:1px solid #e0e0e0; border-radius:6px; padding:8px 10px; }}
  .conf-solid  {{ font-size:10px; color:#1b5e20; background:#e8f5e9; padding:3px 6px; border-radius:3px; margin:2px 0; }}
  .conf-upset  {{ font-size:10px; color:#b71c1c; background:#ffebee; padding:3px 6px; border-radius:3px; margin:2px 0; }}
  .conf-normal {{ font-size:10px; color:#0d47a1; background:#e3f2fd; padding:3px 6px; border-radius:3px; margin:2px 0; }}
  .pace {{ font-size:10px; color:#444; margin:2px 0; }}
  .topics {{ font-size:10px; color:#555; margin:4px 0 4px 14px; padding:0; }}
  .topics li {{ margin:1px 0; }}
  .reco {{ font-size:10px; background:#f0f8ff; padding:5px 8px; border-radius:3px; margin-top:6px; line-height:1.6; }}
  table {{ border-collapse:collapse; width:100%; margin:6px 0; font-size:10px; }}
  th {{ background:#2c3e7a; color:white; padding:3px 5px; text-align:center; }}
  td {{ border:1px solid #ddd; padding:2px 4px; text-align:center; }}
  tr:nth-child(even) {{ background:#f8f8f8; }}
  .ev-summary {{ margin-top:16px; padding:8px; background:#fff8f0; border:1px solid #ffd0a0; border-radius:4px; }}
  .ana-section {{ margin-top:6px; padding:6px 8px; background:#fff3e0; border:1px solid #ffb74d; border-radius:4px; }}
  .ana-title {{ font-size:10px; font-weight:bold; color:#e65100; margin-bottom:4px; }}
  .ana-horse {{ font-size:10px; color:#333; margin:2px 0; padding:2px 0; border-bottom:1px solid #ffe0b2; }}
  .ana-horse:last-child {{ border-bottom:none; }}
  .gemini-block {{ margin-top:8px; padding:8px; background:#f0f4ff; border:1px solid #b0c4ff; border-radius:4px; }}
  .gemini-title {{ font-size:10px; font-weight:bold; color:#3a5bc7; margin-bottom:6px; }}
  .gemini-row {{ display:flex; gap:8px; }}
  .gemini-honmei, .gemini-ana {{ flex:1; padding:6px; border-radius:3px; font-size:10px; }}
  .gemini-honmei {{ background:#e8f4e8; border-left:3px solid #4caf50; }}
  .gemini-ana    {{ background:#fff3e0; border-left:3px solid #ff9800; }}
  .gemini-analyst {{ font-weight:bold; margin-bottom:3px; }}
  .gemini-bet {{ font-size:9px; margin-top:4px; font-weight:bold; color:#555; }}
  @media print {{ body {{ margin:5mm; }} .race-block {{ page-break-inside:avoid; }} }}
</style></head>
<body>
<h1>🐴 keiba-ebye AI予想レポート — {date_str}</h1>
{races_html}
{ev_html}
<p style="font-size:9px;color:#888;margin-top:20px">Generated by keiba-ebye / ブラウザでCtrl+P→PDFで保存できます</p>
</body></html>"""
        return html.encode('utf-8')
    except Exception as e:
        logger.warning(f'generate_pdf_report error: {e}')
        return None



def generate_txt_report(results_list, ev_threshold=1.5):
    """noteに貼りやすい・読みやすいプレーンテキスト形式のレポートを生成"""
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
            out.append(f"{'印':<3} {'穴':<2} {'馬番':>3} {'馬名':<12} {'脚質':<5} {'オッズ':>7} {'勝率':>6} {'複勝率':>7} {'EV':>5}")
            out.append("-" * 60)
        else:
            out.append(f"{'印':<3} {'馬番':>3} {'馬名':<12} {'脚質':<5} {'オッズ':>7} {'勝率':>6} {'複勝率':>7} {'EV':>5}")
            out.append("-" * 56)
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
                ev_str = f"{ev:4.2f}"
                mark = " ★" if ev >= ev_threshold else ""
                if has_ana:
                    ana = str(row.get("穴馬マーク", "") or "")
                    out.append(f"{imp:<3} {ana:<2} {num_:>3} {name:<12} {stle:<5} {odds:>6.1f}倍 {wp:>5.1f}% {fp:>6.1f}% {ev_str}{mark}")
                else:
                    out.append(f"{imp:<3} {num_:>3} {name:<12} {stle:<5} {odds:>6.1f}倍 {wp:>5.1f}% {fp:>6.1f}% {ev_str}{mark}")
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
