"""
race_sim.py — レースのモンテカルロ・シミュレーション

AI勝率を重みとした Plackett-Luce で着順を大量にサンプリングし、
  - 各馬の 1着率 / 連対率 / 複勝圏率
  - 馬連・ワイド・三連複といった**券種ごとの的中確率**
を出す。買い目推奨を「印の順番」から「確率計算」に変えるための土台。

★実装メモ
  Gumbel-top-k トリック: key_i = log(w_i) + Gumbel(0,1) を降順ソートすると、
  重み w の Plackett-Luce 順列がそのまま得られる。逐次サンプリングのループが要らず
  numpy で一括計算できるので、2000試行×18頭でも一瞬で終わる。

★正直に書いておく前提（画面にも明記すること）
  サンプリングの入力は **単勝の勝率だけ**。「1着になる確率」から全着順を組み立てる
  モデル（Plackett-Luce）なので、複勝圏率や連対率はその仮定の上の推定値になる。
  モデルが別に持っている `複勝率(AI予測)`（実データでキャリブレーション済み）とは
  別物で、そちらの方が複勝単体の推定としては信頼できる。
  組み合わせ馬券（馬連・ワイド・三連複）は他に推定手段が無いのでここで出す。
"""

import numpy as np

DEFAULT_SIMS = 4000


def sample_orders(win_probs, n_sims: int = DEFAULT_SIMS, seed=None) -> np.ndarray:
    """Plackett-Luce で着順を n_sims 回サンプリングする。

    戻り値: shape (n_sims, n_horses) の配列。[:, 0] が1着の馬インデックス。
    """
    w = np.asarray(win_probs, dtype=float)
    w = np.clip(np.nan_to_num(w, nan=0.0), 1e-9, None)
    rng = np.random.default_rng(seed)
    # Gumbel(0,1) = -log(-log(U))
    u = rng.random((n_sims, len(w)))
    keys = np.log(w)[None, :] - np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
    return np.argsort(-keys, axis=1)


def horse_stats(orders: np.ndarray, n_horses: int) -> dict:
    """各馬の 1着率 / 連対率(2着以内) / 複勝圏率(3着以内)。"""
    n_sims = orders.shape[0]
    win = np.zeros(n_horses)
    top2 = np.zeros(n_horses)
    top3 = np.zeros(n_horses)
    np.add.at(win, orders[:, 0], 1)
    for k in range(min(2, orders.shape[1])):
        np.add.at(top2, orders[:, k], 1)
    for k in range(min(3, orders.shape[1])):
        np.add.at(top3, orders[:, k], 1)
    return {'win': win / n_sims, 'top2': top2 / n_sims, 'top3': top3 / n_sims}


def _rank_matrix(orders: np.ndarray, n_horses: int) -> np.ndarray:
    """rank[s, i] = 試行sでの馬iの着順(0始まり)。"""
    n_sims, n = orders.shape
    rank = np.empty((n_sims, n), dtype=np.int16)
    cols = np.arange(n, dtype=np.int16)[None, :].repeat(n_sims, axis=0)
    np.put_along_axis(rank, orders, cols, axis=1)
    return rank


def bet_probabilities(orders: np.ndarray, axis_idx: int, himo_idx: list) -> dict:
    """◎を軸にした券種別の的中確率と点数を返す。

    馬連    … ◎ と 相手のどれかが 2着以内（相手の頭数ぶん）
    ワイド  … ◎ と 相手のどれかが 3着以内
    三連複  … ◎ と 相手のうち2頭が 3着以内（相手4頭なら C(4,2)=6点）
    """
    n_sims, n = orders.shape
    rank = _rank_matrix(orders, n)
    himo = [h for h in himo_idx if 0 <= h < n and h != axis_idx]
    if not himo:
        return {}
    a2 = rank[:, axis_idx] < 2
    a3 = rank[:, axis_idx] < 3
    h2 = rank[:, himo] < 2          # (n_sims, len(himo))
    h3 = rank[:, himo] < 3
    umaren = float((a2 & h2.any(axis=1)).mean())
    wide = float((a3 & h3.any(axis=1)).mean())
    # 三連複: ◎が3着以内 かつ 相手が2頭以上3着以内
    sanren = float((a3 & (h3.sum(axis=1) >= 2)).mean())
    n_h = len(himo)
    return {
        '馬連':   {'prob': umaren, 'points': n_h},
        'ワイド': {'prob': wide,   'points': n_h},
        '三連複': {'prob': sanren, 'points': n_h * (n_h - 1) // 2},
    }


def simulate_race(df, n_sims: int = DEFAULT_SIMS, n_himo: int = 4, seed=None) -> dict:
    """res_df（inference の戻り・AI順位で並んでいる）からシミュレーション結果を返す。

    戻り値:
      horses … [{馬番, 馬名, 印, win, top2, top3, odds, ev_tan, ev_fuku}, ...]（AI順位順）
      bets   … 券種別の {prob, points, 期待払戻の目安}
      n_sims … 試行回数
    """
    if df is None or '勝率(AI予測)' not in getattr(df, 'columns', []):
        return {}
    probs = df['勝率(AI予測)'].astype(float).to_numpy()
    n = len(probs)
    if n < 3:
        return {}
    orders = sample_orders(probs, n_sims=n_sims, seed=seed)
    st = horse_stats(orders, n)

    horses = []
    for i in range(n):
        r = df.iloc[i]
        try:
            odds = float(r.get('単勝オッズ', 0) or 0)
        except (TypeError, ValueError):
            odds = 0.0
        horses.append({
            '馬番': r.get('馬番'), '馬名': r.get('馬名', ''), '印': r.get('印', '') or '',
            'win': float(st['win'][i]), 'top2': float(st['top2'][i]),
            'top3': float(st['top3'][i]), 'odds': odds,
        })

    bets = bet_probabilities(orders, 0, list(range(1, min(1 + n_himo, n))))
    return {'horses': horses, 'bets': bets, 'n_sims': n_sims}
