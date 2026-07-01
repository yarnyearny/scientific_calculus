# 数値科学計算

## L1 障害物付きボロノイ格子分割

`l1_voronoi` は、軸平行な線分障害物を持つ矩形領域で L1 距離のボロノイ分割を計算する小さな Python 実装です。

アルゴリズムは次の段階に分かれています。

1. 問題定義: 領域、サイト、軸平行障害物を受け取る
2. 伝播: サイト・障害物端点・領域境界から作った格子上で多始点 Dijkstra を行う
3. 単位セル分割: 各セルの4隅の距離だけから、重み付き L1 距離の下包絡を半平面クリップで求める
4. 貼り合わせ: 全セルの分割片を同じ座標系で返し、隣接セルをまとめて扱える形にする

同点の格子頂点では、勝者を1つに潰さず、最短距離を達成するサイト集合を保持します。

```python
from l1_voronoi import L1VoronoiProblem, Point, Segment, compute_diagram

problem = L1VoronoiProblem(
    domain=(0, 0, 4, 3),
    sites={
        "a": Point(0, 0),
        "b": Point(4, 3),
    },
    obstacles=(
        Segment(Point(2, 0), Point(2, 2)),
    ),
)

diagram = compute_diagram(problem)
print(diagram.nearest_sites_at(Point(3, 1)))
```

テストは標準ライブラリの `unittest` で実行できます。

```bash
python3 -m unittest discover -s tests
```
