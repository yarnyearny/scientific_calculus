# 数値科学計算

## 面隣接グラフの共有経路最適化

矩形分割の各面を頂点、辺を共有する面同士を隣接辺として、複数のスタート面から
ゴール面までを結ぶ共有経路木を近似計算します。辺重みは面の重心間の L1 距離です。
複数経路が共有する辺は総コストへ一度だけ加算されます。

一般の Steiner tree 問題は NP 困難であるため、次を組み合わせて候補を探索します。

- 終端距離閉包の最小全域木
- 現在の木への逐次最短路接続
- スタート順列のランダム再始動付き局所探索

詳しい責務分割と設計判断は [`docs/architecture.md`](docs/architecture.md) を参照して
ください。

### 実行

Python 3.11 以上を使用します。実行時依存パッケージはありません。

```bash
python3 -m pip install -e .
python3 -m scientific_calculus examples/shared_trunk.json
```

標準入力・標準出力も利用できます。

```bash
python3 -m scientific_calculus < problem.json
```

入力例:

```json
{
  "faces": [
    {"id": "g", "centroid": [0, 0]},
    {"id": "j", "centroid": [1, 0]},
    {"id": "a", "centroid": [2, 1]},
    {"id": "b", "centroid": [2, -1]}
  ],
  "adjacencies": [["g", "j"], ["j", "a"], ["j", "b"]],
  "goal": "g",
  "starts": ["a", "b"],
  "config": {
    "random_restarts": 16,
    "local_iterations": 200,
    "seed": 0
  }
}
```

出力の `edges` は `parent` から `child` へ、ゴールを根として向き付けられます。

### テスト

```bash
python3 -m unittest discover -s tests -v
```
