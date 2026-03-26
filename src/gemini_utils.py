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
    本命党「鉄板師・剛三」と穴党「穴師・乱丸」の2アナリストによる
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
        horses_text  = _build_horses_text(df_res, n=7)
        topics_text  = "\n".join(topics[:3]) if topics else "なし"

        # EV1.5以上馬・穴馬マークを抽出してプロンプトを補強
        ev_horses = df_res[df_res['期待値'].astype(float) >= 1.5] if '期待値' in df_res.columns else df_res.iloc[0:0]
        ana_horses = df_res[df_res.get('穴馬マーク', '') == '🎯'] if '穴馬マーク' in df_res.columns else df_res.iloc[0:0]
        ev_note  = "、".join(f"{r.get('馬番','')}番{r.get('馬名','')}(EV{float(r.get('期待値',0)):.2f})" for _, r in ev_horses.iterrows()) or "なし"
        ana_note = "、".join(f"{r.get('馬番','')}番{r.get('馬名','')}" for _, r in ana_horses.iterrows()) or "なし"

        prompt = f"""あなたは競馬番組に登場する2人のコメンテーターです。
以下のLightGBM統計AIの予測データを見て、それぞれの視点で短いコメントと具体的な買い目を述べてください。

【ML予測上位7頭（馬番・AI勝率・期待値EV）】
{horses_text}
【展開予測】{pace_text}
【信頼度】{confidence_text}
【注目トピック】{topics_text}
【EV1.5以上の馬】{ev_note}
【穴馬マーク🎯の馬】{ana_note}

━━━━━━━━━━━━━━━━━━━━━━
■ アナリスト①「鉄板師・剛三」（本命党）
スタイル: AIの最高勝率馬を軸に堅実回収を狙う。単勝・馬連・ワイドの本命軸流しを好む。口調は冷静・論理的。

■ アナリスト②「穴師・乱丸」（穴党）
スタイル: EV1.5以上と穴馬マーク🎯に注目。市場が見落としたロマンを掘り起こす。ワイド・3連複の穴絡みや大穴単勝を好む。口調は情熱的。
━━━━━━━━━━━━━━━━━━━━━━

以下のJSON形式のみで返答してください（コードブロック・余分なテキスト不要）:
{{
  "honmei_comment": "鉄板師のレース展望（150字程度、なぜその馬が有力かの推論を含む）",
  "honmei_bet": "鉄板師の買い目（馬番・馬券種を明記・60字以内）",
  "ana_comment": "穴師のレース展望（150字程度、EV・穴馬マーク・展開妙味の推論を含む）",
  "ana_bet": "穴師の買い目（馬番・馬券種を明記・60字以内）"
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
