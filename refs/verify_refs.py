"""Confere cada arXiv id: baixa o título pela API do arXiv. Saída: refs/verified.json"""
import json, re, time, urllib.request, xml.etree.ElementTree as ET, sys
IDS = """2203.05482 2212.04089 2306.01708 2311.03099 1803.05407 1912.05671 2209.04836 2111.09832 2212.09849 2109.01903 2307.13269
2403.13257 2403.13187 2408.07666 2412.15115 2502.02737 2304.01373 1803.05457 1905.07830 1907.10641 1609.07843 2409.12186 2411.00640
1612.01474 2306.03241 2306.04488 2212.10445 2410.03617 2310.02575 2403.19390 2310.04799 2305.12827 1902.10811 1411.2664 2309.15698""".split()
out = {}
for i in range(0, len(IDS), 10):
    ch = IDS[i:i+10]; u = "http://export.arxiv.org/api/query?max_results=10&id_list=" + ",".join(ch)
    x = urllib.request.urlopen(u, timeout=60).read(); root = ET.fromstring(x); ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns):
        idu = e.find("a:id", ns).text.split("/abs/")[-1]; idu = re.sub(r"v\d+$", "", idu)
        t = " ".join(e.find("a:title", ns).text.split()); au = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        y = e.find("a:published", ns).text[:4]; out[idu] = {"title": t, "authors": au[:4], "n_authors": len(au), "year": y}
    time.sleep(3)
json.dump(out, open("refs/verified.json", "w"), indent=1, ensure_ascii=False)
miss = [i for i in IDS if i not in out]; print(len(out), "ok; faltam:", miss)
for k, v in out.items(): print(k, v["year"], v["title"][:90], "|", v["authors"][0], v["n_authors"])
