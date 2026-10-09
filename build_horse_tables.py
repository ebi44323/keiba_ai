"""
build_horse_tables.py — 馬ごとの過去走テーブルを最新データで作り直して HF Hub に置く
==========================================================================

推論が使う「各馬の最新走（前走着順・通過順＝脚質・スピード指数…）」の表は、これまで
モデル bundle の中にしか無く、**再学習しない限り更新されなかった**。Phase 2 の測定で再学習を
止めた結果、9月以降にデビューした2歳馬は脚質もスピード指数も空になっていた（2026-10-10 判明:
2歳未勝利の脚質不明 57%）。

このスクリプトはモデル本体には触らず、馬のテーブルだけを作り直して `horse_tables.pkl` として
HF Hub に置く。推論側（src/core_model._try_load_model_from_hub）はこれがあればモデル内の表と
差し替える。**再学習ではない**（モデル・キャリブレータ・各種辞書は不変）。

    python build_horse_tables.py            # 作成して HF Hub にアップロード → HF Space を再起動
    python build_horse_tables.py --no-upload # 作成のみ（horse_tables.pkl をローカルに保存）

weekly_update.yml がデータ更新の直後に実行する。
"""
import argparse
import io
import logging
import os
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_horse_tables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true", help="HF Hub にアップロードしない")
    ap.add_argument("--out", default="horse_tables.pkl")
    ap.add_argument("--no-restart", action="store_true", help="アップロード後に HF Space を再起動しない")
    args = ap.parse_args()

    from src.features_engine import create_features
    from src.core_model import build_horse_tables, _HORSE_TABLES_FILE

    t0 = time.time()
    df = pd.read_csv("learning_data_perfect_tier.zip", compression="zip", dtype=str)
    # 学習時(prepare_model_and_data)と同じ前処理
    if "調教師" in df.columns:
        df["調教師"] = df["調教師"].str.replace(r"^\[.+?\]\s*", "", regex=True)
    df, _ = create_features(df)
    tables = build_horse_tables(df)
    logger.info(f"作成完了: データ {tables['data_last_date']} まで / {tables['n_rows']:,}行 / "
                f"{time.time() - t0:.0f}秒")

    blob = pickle.dumps(tables, protocol=4)
    with open(args.out, "wb") as f:
        f.write(blob)
    logger.info(f"{args.out} を保存（{len(blob) / 1e6:.1f}MB）")

    if args.no_upload:
        return
    token, repo = os.environ.get("HF_TOKEN", ""), os.environ.get("HF_REPO_ID", "")
    if not token or not repo:
        logger.error("HF_TOKEN / HF_REPO_ID が未設定のためアップロードできません")
        sys.exit(1)
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=io.BytesIO(blob), path_in_repo=_HORSE_TABLES_FILE,
        repo_id=repo, repo_type="dataset", token=token,
        commit_message=f"馬テーブル更新（データ {tables['data_last_date']} まで）",
    )
    logger.info(f"HF Hub {repo}/{_HORSE_TABLES_FILE} にアップロードしました")

    # アプリ(HF Space)は起動時に1回だけ表を読むので、再起動して新しい表を読ませる。
    # 失敗しても表のアップロード自体は済んでいるので致命ではない（次の再起動で反映）。
    space = os.environ.get("HF_SPACE_ID", "ebi44323/keiba-ebye")
    if space and not args.no_restart:
        try:
            HfApi(token=token).restart_space(space)
            logger.info(f"HF Space {space} を再起動しました（新しい馬テーブルを読み込ませるため）")
        except Exception as e:
            logger.warning(f"HF Space の再起動に失敗（次の再起動で反映）: {e}")


if __name__ == "__main__":
    main()
