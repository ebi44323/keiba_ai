import os
import re
import json
import logging

logger = logging.getLogger('keiba_ebye')

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# 試すモデルの優先順（新しいほど上）
_CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]


def check_gemini_available() -> bool:
    return bool(_GEMINI_API_KEY)


def _make_model(model_name: str):
    """指定モデルで GenerativeModel を返す。失敗時は候補を順番に試す。"""
    import google.generativeai as genai
    genai.configure(api_key=_GEMINI_API_KEY)
    candidates = [model_name] + [m for m in _CANDIDATE_MODELS if m != model_name]
    last_err = None
    for m in candidates:
        try:
            model = genai.GenerativeModel(m)
            return model, m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"利用可能な Gemini モデルが見つかりません: {last_err}")


def _parse_json(text: str) -> dict:
    """レスポンスからJSONを抽出してパース。コードブロックを除去して再試行。"""
    text = text.strip()
    # コードブロック除去
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _build_horses_text(df_res, n: int = 7) -> str:
    lines = []
    for _, r in df_res.head(n).iterrows():
        imp  = str(r.get('印', ''))
        mark = str(r.get('穴馬マーク', ''))
        name = str(r.get('馬名', ''))
        num  = str(r.get('馬番', ''))
        odds = float(r.get('単勝オッズ', 0))
        wp   = float(r.get('勝率(AI予測)', 0)) * 100
        fp   = float(r.get('複勝率(AI予測)', 0)) * 100
        ev   = float(r.get('期待値', 0) or 0)
        lines.append(
            f"  {imp}{mark} {num}番 {name}　{odds:.1f}倍 / AI勝率{wp:.1f}% / AI複勝率{fp:.1f}% / EV{ev:.2f}"
        )
    return "\n".join(lines)


def generate_two_analysts(df_res, pace_text: str, confidence_text: str,
                           topics: list, reco: str,
                           model_name: str = "gemini-2.0-flash") -> dict | None:
    """
    本命党「伊藤ホンメ」と穴党「風穴あけるズ」の2アナリストによる
    レース展望コメント＋具体的買い目を生成する。

    戻り値:
      {
        'honmei': {'comment': str, 'bet': str},
        'ana':    {'comment': str, 'bet': str},
        'model':  str,
      }
    失敗時は None
    """
    if not _GEMINI_API_KEY:
        return None

    try:
        model, used_model = _make_model(model_name)
        horses_text = _build_horses_text(df_res, n=7)
        topics_text = "\n".join(topics[:3]) if topics else "なし"

        # EV・穴馬情報を抽出
        ev_horses  = df_res[df_res['期待値'].astype(float) >= 1.5] if '期待値' in df_res.columns else df_res.iloc[0:0]
        ana_horses = df_res[df_res['穴馬マーク'] == '🎯'] if '穴馬マーク' in df_res.columns else df_res.iloc[0:0]
        ev_note    = "、".join(f"{r.get('馬番','')}番{r.get('馬名','')}(EV{float(r.get('期待値',0)):.2f})" for _, r in ev_horses.iterrows()) or "なし"
        ana_note   = "、".join(f"{r.get('馬番','')}番{r.get('馬名','')}" for _, r in ana_horses.iterrows()) or "なし"
        top_horse  = df_res.iloc[0] if len(df_res) > 0 else None
        top_name   = str(top_horse.get('馬名','')) if top_horse is not None else ''
        top_num    = str(top_horse.get('馬番','')) if top_horse is not None else ''

        prompt = f"""あなたは競馬予想番組の2人の個性的なコメンテーターです。
以下のLightGBM統計AIの予測データをベースに、それぞれ**独自の視点と個性**でコメントしてください。
AIの数字を参考にしつつも、そのまま読み上げるのではなく、キャラクターとしての解釈・感情・こだわりを出してください。

【ML予測データ（上位7頭）】
{horses_text}
【展開予測】{pace_text}
【信頼度判定】{confidence_text}
【注目トピック】{topics_text}
【EV1.5以上の馬】{ev_note}
【穴馬マーク🎯の馬】{ana_note}

━━━━━━━━━━━━━━━━━━━━━━
■ アナリスト①「伊藤ホンメ」（本命党）
【キャラ設定】
- 元JRA調教助手という経歴を持つ実績派。「数字より馬を見ろ」が口癖だが、実はAI予測を信頼している
- AI最高勝率馬を基本軸にしつつ、展開・ペース適性・距離経験から"確信の度合い"を語る
- 本命が人気薄でも「数字が出てるなら買うしかない」と言い切る剛胆さもある
- 単勝・馬連・ワイドの本命軸を好む。口調はベテランらしく落ち着いているが時々熱くなる
- 必ず最後に「{top_num}番{top_name}から入ります」と締める

■ アナリスト②「風穴あけるズ」（穴党）
【キャラ設定】
- 競馬歴20年のロマン追求型。「穴を当てるのは芸術だ」が口癖
- EV1.5以上・穴馬マーク🎯の馬を溺愛。AI本命には冷たいことも多い
- 展開の綾・コース形態・人気の盲点を独自の言葉で語る（例:「この外枠は実は有利なんですよ！」）
- 穴馬がいない場合でも「この人気では市場が間違っている」とひねった視点を見つける
- ワイド・3連複の穴絡みを愛する。口調はテンション高め、ときどき暴走気味
━━━━━━━━━━━━━━━━━━━━━━

以下のJSON形式のみで返答してください（コードブロック・余分なテキスト不要）:
{{
  "honmei_comment": "伊藤ホンメのコメント（150〜180字、キャラの個性と展開推論を含む）",
  "honmei_bet": "伊藤ホンメの買い目（馬番・馬券種を明記・60字以内）",
  "ana_comment": "風穴あけるズのコメント（150〜180字、キャラの個性とロマン・穴の根拠を含む）",
  "ana_bet": "風穴あけるズの買い目（馬番・馬券種を明記・60字以内）"
}}"""

        response = model.generate_content(prompt)
        text = response.text or ""
        data = _parse_json(text)

        result = {
            'honmei': {
                'comment': data.get('honmei_comment', ''),
                'bet':     data.get('honmei_bet', ''),
            },
            'ana': {
                'comment': data.get('ana_comment', ''),
                'bet':     data.get('ana_bet', ''),
            },
            'model': used_model,
        }
        logger.info(f"2アナリスト生成完了 ({used_model})")
        return result

    except Exception as e:
        logger.warning(f"2アナリスト生成失敗: {e}")
        return None
