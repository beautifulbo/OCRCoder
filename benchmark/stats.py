"""统计 benchmark 数据的基本信息"""
import json, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/home/10358884/10358884/tools/benchmark/benchmark_data.json"

with open(path) as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

repos = Counter(s["source_graph"].split("#")[0] for s in data)
print("\nRepository distribution:")
for repo, cnt in repos.most_common():
    print(f"  {repo}: {cnt}")

nt = Counter(s["seed_node_type"] for s in data)
print(f"\nSeed node types: {dict(nt)}")

mp = Counter()
for s in data:
    for f in s["masked_fields"]:
        mp[f] += 1
print(f"Masked fields: {dict(mp)}")

sizes = [len(s["subgraph"]["nodes"]) for s in data]
print(f"\nSubgraph nodes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

esizes = [len(s["subgraph"]["edges"]) for s in data]
print(f"Subgraph edges: min={min(esizes)}, max={max(esizes)}, avg={sum(esizes)/len(esizes):.1f}")

text_lens = [len(s["ground_truth"]["text"]) for s in data if "text" in s["ground_truth"]]
sl = sorted(text_lens)
n = len(sl)
print(f"\nGT text length:")
print(f"  min={sl[0]}, max={sl[-1]}, avg={sum(sl)/n:.1f}")
for p in [25, 50, 75, 90, 95]:
    print(f"  P{p}: {sl[int(n*p/100)]}")
