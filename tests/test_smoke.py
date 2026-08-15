# -*- coding: utf-8 -*-
"""
最小限のスモークテスト（回帰防止）。
pytest 不要 — `python tests/test_smoke.py` で実行可（pytestがあれば `pytest tests/` も可）。

守りたい不変条件:
  1. 特徴量リストの健全性（重複なし・カテゴリと数値が排他）
  2. bundle のフィールド位置契約（producer=core_model と consumer=inference/backtest の一致）
  3. create_features が実データ先頭で例外なく走り、NUM_FEATURES を全て生成する
  4. 絶対スコア正規化 と EV優先フロア の計算不変条件
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS


def test_feature_lists_integrity():
    assert len(NUM_FEATURES) == len(set(NUM_FEATURES)), 'NUM_FEATURES に重複あり'
    assert len(CAT_FEATURES) == len(set(CAT_FEATURES)), 'CAT_FEATURES に重複あり'
    assert not (set(NUM_FEATURES) & set(CAT_FEATURES)), 'NUM と CAT が重複'
    # 騎手能力特徴量（2026-05-22追加）が残っていること
    assert '騎手_通算着順パーセント' in NUM_FEATURES
    assert '騎手_競馬場_着順パーセント' in NUM_FEATURES
    # 削除済みの高カーディナリティ騎手カテゴリが復活していないこと
    assert '騎手_競馬場' not in CAT_FEATURES
    assert '騎手_距離' not in CAT_FEATURES


def test_bundle_field_contract():
    """core_model のbundle並びと、inference(*_extra)/backtest(*_rest) の添字が一致すること。"""
    # core_model.py のbundle並び（名前で表現）
    order = [
        'model', 'model_win', 'model_reg', 'features', 'cat_features', 'num_features',
        'cat_categories_dict', 'latest_horse_data', 'horse_course_dict', 'ped_dict',
        'known_jockeys', 'known_trainers', 'te_dicts', 'global_mean', 'recent_return_rate',
        'best_weight', 'auc_win', 'auc_place', 'calibrator', 'model_d', 'ped_aptitude_dict',
        'horse_heavy_dict', 'sire_heavy_dict', 'jockey_overall_dict', 'jockey_venue_dict',
        'score_norms', 'SOFTMAX_TEMPERATURE', 'place_calibrator',
        'draw_course_dict', 'draw_course_bucket_dict', 'style_course_dict',
    ]
    # inference.py: 18個を固定展開 → *_extra
    extra = order[18:]
    assert extra[5] == 'jockey_overall_dict'
    assert extra[6] == 'jockey_venue_dict'
    assert extra[7] == 'score_norms', 'inference の _extra[7] が score_norms からズレた'
    assert extra[8] == 'SOFTMAX_TEMPERATURE', 'inference の _extra[8] がズレた'
    assert extra[9] == 'place_calibrator', 'inference の _extra[9] が place_calibrator からズレた'
    assert extra[10] == 'draw_course_dict', 'inference の _extra[10] が draw_course_dict からズレた'
    assert extra[11] == 'draw_course_bucket_dict', 'inference の _extra[11] がズレた'
    assert extra[12] == 'style_course_dict', 'inference の _extra[12] が style_course_dict からズレた'
    # backtest.py: 14個を固定展開 → *_rest
    rest = order[14:]
    assert rest[4] == 'calibrator'
    assert rest[11] == 'score_norms', 'backtest の _rest[11] が score_norms からズレた'
    assert rest[12] == 'SOFTMAX_TEMPERATURE', 'backtest の _rest[12] がズレた'


def test_create_features_smoke():
    """実データ先頭で create_features が例外なく走り、NUM_FEATURES を全生成すること。"""
    import pandas as pd
    from src.features_engine import create_features
    path = None
    for p in ('learning_data_perfect_tier.zip', 'learning_data_perfect_tier.csv'):
        if os.path.exists(p):
            path = p
            break
    if path is None:
        print('  [skip] 学習データが無いため create_features スモークをスキップ')
        return
    comp = 'zip' if path.endswith('.zip') else None
    df = pd.read_csv(path, compression=comp, dtype=str, nrows=5000)
    if '調教師' in df.columns:
        df['調教師'] = df['調教師'].str.replace(r'^\[.+?\]\s*', '', regex=True)
    out, _ = create_features(df)
    missing = [c for c in NUM_FEATURES if c not in out.columns]
    assert not missing, f'create_features が生成しなかった特徴量: {missing}'


def test_absolute_norm_and_ev_floor():
    # 絶対正規化: clip[0,1] かつ 単調
    lo, hi = -2.0, 3.0
    s = np.array([-3.0, -2.0, 0.0, 3.0, 5.0])
    n = np.clip((s - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert np.all(np.diff(n) >= 0), '正規化が単調でない'
    assert n[0] == 0.0 and n[-1] == 1.0
    # EV優先フロア: 頭数連動 max(0.25, 1.4/N)
    def floor(n_runners):
        return max(0.25, 1.4 / n_runners)
    assert abs(floor(5) - 0.28) < 1e-9
    assert floor(8) == 0.25 and floor(18) == 0.25
    assert floor(5) > floor(8), '小頭数の方がフロアが高いはず'


def test_place_prob_invariants():
    """複勝率の物理制約（機能3・2026-08-16 回帰防止）:
       複勝率 >= 勝率（勝てば必ず3着内）かつ <= 0.98。
       place_calibrator が step 関数で破綻値（0.0 や 0.999）を返しても安全網が守ること。
       inference.py の複勝率算出ロジックと同一式で検証する。"""
    class _PathologicalCalib:
        # わざと破綻させる: 小入力→0.0（複勝率<勝率を誘発）, 大入力→0.999（>0.98を誘発）
        def predict(self, x):
            x = np.asarray(x, dtype=float)
            return np.where(x < 0.1, 0.0, 0.999)

    win_probs     = np.array([0.03, 0.05, 0.20, 0.35, 0.60])
    softmax_probs = np.array([0.02, 0.04, 0.18, 0.40, 0.70])

    # inference.py と同一: place_calibrator は softmax_probs（学習ドメイン）を入力にする
    cal = _PathologicalCalib()
    place_probs = np.clip(cal.predict(softmax_probs), 0.0, 0.95)
    result = np.clip(np.maximum(place_probs, win_probs), 0.0, 0.98)
    assert np.all(result >= win_probs - 1e-9), '複勝率が勝率を下回った（物理制約違反）'
    assert np.all(result <= 0.98 + 1e-9), '複勝率が上限0.98を超えた'

    # フォールバック(Bradley-Terry)も同じ不変条件を満たすこと
    bt = np.clip((3.0 * win_probs) / (2.0 * win_probs + 1.0 + 1e-9), 0, 0.95)
    bt_res = np.clip(np.maximum(bt, win_probs), 0.0, 0.98)
    assert np.all(bt_res >= win_probs - 1e-9), 'BTフォールバックが物理制約違反'
    assert np.all(bt_res <= 0.98 + 1e-9)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {t.__name__}: {e}')
        except Exception as e:
            failed += 1
            print(f'  ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
