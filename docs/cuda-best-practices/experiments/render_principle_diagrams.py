from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def wrap_svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<defs><style>\n"
        ".bg{fill:#f8fafc;}\n"
        ".panel{fill:#ffffff;stroke:#cbd5e1;stroke-width:2;rx:18;ry:18;}\n"
        '.title{font:700 28px "DejaVu Sans", sans-serif;fill:#0f172a;}\n'
        '.label{font:700 18px "DejaVu Sans", sans-serif;fill:#1e293b;}\n'
        '.body{font:400 15px "DejaVu Sans", sans-serif;fill:#334155;}\n'
        '.small{font:400 13px "DejaVu Sans", sans-serif;fill:#64748b;}\n'
        ".cpu{fill:#dbeafe;stroke:#2563eb;stroke-width:2;}\n"
        ".gpu{fill:#dcfce7;stroke:#16a34a;stroke-width:2;}\n"
        ".warn{fill:#fef3c7;stroke:#d97706;stroke-width:2;}\n"
        ".hot{fill:#fee2e2;stroke:#dc2626;stroke-width:2;}\n"
        ".accent{fill:#ede9fe;stroke:#7c3aed;stroke-width:2;}\n"
        ".line{stroke:#94a3b8;stroke-width:2;}\n"
        ".dash{stroke:#94a3b8;stroke-width:2;stroke-dasharray:7 6;}\n"
        ".arrow{stroke:#0f172a;stroke-width:2.5;marker-end:url(#arrow);fill:none;}\n"
        ".grid{stroke:#e2e8f0;stroke-width:1.5;}\n"
        "</style>\n"
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 z" fill="#0f172a"/></marker></defs>\n'
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>\n'
        f"{body}\n</svg>\n"
    )


def write(name: str, body: str, width: int = 1400, height: int = 560) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / name).write_text(wrap_svg(width, height, body))


def timing_baseline() -> None:
    body = """
<text class="title" x="44" y="48">Timing Baseline: CPU/GPU Timeline</text>
<text class="body" x="44" y="78">host_submit 只量 CPU 提交；kernel_ms 只量 GPU 执行；end_to_end 覆盖调用方真正等待完成的时间。</text>

<rect class="panel" x="40" y="110" width="1320" height="400"/>
<text class="label" x="88" y="180">CPU</text>
<line class="line" x1="180" y1="170" x2="1260" y2="170"/>
<rect class="cpu" x="300" y="142" width="210" height="56" rx="12" ry="12"/>
<text class="body" x="328" y="176">record start + launch</text>
<rect class="warn" x="890" y="142" width="220" height="56" rx="12" ry="12"/>
<text class="body" x="930" y="176">cudaDeviceSynchronize</text>

<text class="label" x="84" y="330">GPU</text>
<line class="line" x1="180" y1="320" x2="1260" y2="320"/>
<rect class="gpu" x="470" y="292" width="380" height="56" rx="12" ry="12"/>
<text class="body" x="590" y="326">kernel execution</text>
<line class="dash" x1="470" y1="212" x2="470" y2="376"/>
<line class="dash" x1="850" y1="212" x2="850" y2="376"/>

<line class="arrow" x1="300" y1="108" x2="510" y2="108"/>
<text class="body" x="332" y="96">host_submit_ms</text>

<line class="arrow" x1="470" y1="392" x2="850" y2="392"/>
<text class="body" x="602" y="422">kernel_ms</text>

<line class="arrow" x1="300" y1="470" x2="1110" y2="470"/>
<text class="body" x="640" y="500">end_to_end_ms</text>

<text class="small" x="270" y="236">CPU 很快返回，GPU 之后才真正开始跑 kernel。</text>
<text class="small" x="878" y="378">同步等待把“提交”与“GPU 实际完成”连接起来。</text>
"""
    write("timing-baseline-cpu-gpu-timeline.svg", body)


def pinned_pageable() -> None:
    body = """
<text class="title" x="44" y="48">Pinned vs Pageable: Transfer Path</text>
<text class="body" x="44" y="78">pageable 内存往往要先拷到临时 pinned staging buffer；pinned 可以直接交给 DMA。</text>

<rect class="panel" x="40" y="110" width="630" height="400"/>
<text class="label" x="70" y="150">Pageable</text>
<rect class="cpu" x="80" y="210" width="170" height="64" rx="12" ry="12"/>
<text class="body" x="115" y="248">host pageable</text>
<rect class="warn" x="300" y="210" width="190" height="64" rx="12" ry="12"/>
<text class="body" x="326" y="248">driver staging pinned</text>
<rect class="gpu" x="540" y="210" width="90" height="64" rx="12" ry="12"/>
<text class="body" x="562" y="248">GPU</text>
<line class="arrow" x1="250" y1="242" x2="300" y2="242"/>
<line class="arrow" x1="490" y1="242" x2="540" y2="242"/>
<text class="small" x="146" y="320">多了一次 host-side staging copy</text>
<text class="small" x="168" y="344">小包传输时成本尤其显眼</text>

<rect class="panel" x="700" y="110" width="660" height="400"/>
<text class="label" x="730" y="150">Pinned</text>
<rect class="accent" x="760" y="210" width="180" height="64" rx="12" ry="12"/>
<text class="body" x="809" y="248">host pinned</text>
<rect class="gpu" x="1110" y="210" width="120" height="64" rx="12" ry="12"/>
<text class="body" x="1148" y="248">GPU</text>
<line class="arrow" x1="940" y1="242" x2="1110" y2="242"/>
<text class="body" x="986" y="228">DMA</text>
<text class="small" x="806" y="320">直接走 DMA，路径更短</text>
<text class="small" x="818" y="344">更容易接近链路带宽上限</text>
"""
    write("pinned-pageable-transfer-path.svg", body)


def multi_stream_overlap() -> None:
    body = """
<text class="title" x="44" y="48">Multi Stream Overlap: Serial vs Overlapped</text>
<text class="body" x="44" y="78">单 stream 把 H2D / kernel / D2H 串起来；多 stream 把不同 chunk 错峰排队，copy 与 compute 可重叠。</text>

<rect class="panel" x="40" y="110" width="630" height="400"/>
<text class="label" x="76" y="150">Single stream</text>
<line class="line" x1="110" y1="230" x2="610" y2="230"/>
<rect class="cpu" x="130" y="198" width="120" height="64" rx="12" ry="12"/><text class="body" x="170" y="236">H2D</text>
<rect class="gpu" x="260" y="198" width="150" height="64" rx="12" ry="12"/><text class="body" x="312" y="236">kernel</text>
<rect class="warn" x="420" y="198" width="120" height="64" rx="12" ry="12"/><text class="body" x="460" y="236">D2H</text>
<text class="small" x="168" y="318">chunk0 全串行</text>
<text class="small" x="170" y="342">资源利用有空洞</text>

<rect class="panel" x="700" y="110" width="660" height="400"/>
<text class="label" x="736" y="150">Four streams</text>
<text class="small" x="736" y="184">stream0</text><line class="line" x1="820" y1="180" x2="1310" y2="180"/>
<text class="small" x="736" y="234">stream1</text><line class="line" x1="820" y1="230" x2="1310" y2="230"/>
<text class="small" x="736" y="284">stream2</text><line class="line" x1="820" y1="280" x2="1310" y2="280"/>
<text class="small" x="736" y="334">stream3</text><line class="line" x1="820" y1="330" x2="1310" y2="330"/>
<rect class="cpu" x="840" y="150" width="90" height="40" rx="10" ry="10"/><rect class="gpu" x="930" y="150" width="120" height="40" rx="10" ry="10"/><rect class="warn" x="1050" y="150" width="90" height="40" rx="10" ry="10"/>
<rect class="cpu" x="900" y="200" width="90" height="40" rx="10" ry="10"/><rect class="gpu" x="990" y="200" width="120" height="40" rx="10" ry="10"/><rect class="warn" x="1110" y="200" width="90" height="40" rx="10" ry="10"/>
<rect class="cpu" x="960" y="250" width="90" height="40" rx="10" ry="10"/><rect class="gpu" x="1050" y="250" width="120" height="40" rx="10" ry="10"/><rect class="warn" x="1170" y="250" width="90" height="40" rx="10" ry="10"/>
<rect class="cpu" x="1020" y="300" width="90" height="40" rx="10" ry="10"/><rect class="gpu" x="1110" y="300" width="120" height="40" rx="10" ry="10"/><rect class="warn" x="1230" y="300" width="90" height="40" rx="10" ry="10"/>
<text class="small" x="910" y="388">不同 chunk 在不同 stream 上错峰，copy 与 compute 可以并行占用不同硬件队列。</text>
"""
    write("multi-stream-overlap-timeline.svg", body)


def offset_stride() -> None:
    body = """
<text class="title" x="44" y="48">Offset vs Stride: Access Pattern</text>
<text class="body" x="44" y="78">coalescing 看的是同一拍 warp 里 32 个线程的地址分布，不只是单线程自己是不是连续。</text>

<rect class="panel" x="40" y="110" width="410" height="400"/>
<text class="label" x="72" y="150">Contiguous</text>
<text class="small" x="72" y="178">lane0..lane7 访问相邻地址</text>
<rect class="gpu" x="80" y="240" width="36" height="44"/><rect class="gpu" x="118" y="240" width="36" height="44"/><rect class="gpu" x="156" y="240" width="36" height="44"/><rect class="gpu" x="194" y="240" width="36" height="44"/><rect class="gpu" x="232" y="240" width="36" height="44"/><rect class="gpu" x="270" y="240" width="36" height="44"/><rect class="gpu" x="308" y="240" width="36" height="44"/><rect class="gpu" x="346" y="240" width="36" height="44"/>
<text class="small" x="140" y="336">一个或少数几个 segment 就能覆盖</text>

<rect class="panel" x="495" y="110" width="410" height="400"/>
<text class="label" x="527" y="150">Offset</text>
<text class="small" x="527" y="178">整体平移一格，但 lane 之间仍基本相邻</text>
<rect x="535" y="240" width="36" height="44" fill="#e5e7eb"/><rect class="warn" x="573" y="240" width="36" height="44"/><rect class="warn" x="611" y="240" width="36" height="44"/><rect class="warn" x="649" y="240" width="36" height="44"/><rect class="warn" x="687" y="240" width="36" height="44"/><rect class="warn" x="725" y="240" width="36" height="44"/><rect class="warn" x="763" y="240" width="36" height="44"/><rect class="warn" x="801" y="240" width="36" height="44"/>
<text class="small" x="598" y="336">可能多碰到一个额外 segment</text>

<rect class="panel" x="950" y="110" width="410" height="400"/>
<text class="label" x="982" y="150">Stride</text>
<text class="small" x="982" y="178">lane 间地址相隔很大</text>
<rect class="hot" x="990" y="240" width="36" height="44"/><rect x="1028" y="240" width="36" height="44" fill="#e5e7eb"/><rect x="1066" y="240" width="36" height="44" fill="#e5e7eb"/><rect class="hot" x="1104" y="240" width="36" height="44"/><rect x="1142" y="240" width="36" height="44" fill="#e5e7eb"/><rect x="1180" y="240" width="36" height="44" fill="#e5e7eb"/><rect class="hot" x="1218" y="240" width="36" height="44"/><rect x="1256" y="240" width="36" height="44" fill="#e5e7eb"/>
<text class="small" x="1015" y="336">一次 warp load 被拆成更多离散请求</text>
"""
    write("offset-stride-access-pattern.svg", body)


def shared_gemm() -> None:
    body = """
<text class="title" x="44" y="48">Shared Memory GEMM: Tile Reuse and Padding</text>
<text class="body" x="44" y="78">按 Best Practices Guide 的矩阵图思路重画：先看 tile 如何减少 global load，再看 C=AAT 里 transpose tile 为什么需要 +1 padding。</text>

<rect class="panel" x="40" y="110" width="430" height="400"/>
<text class="label" x="70" y="150">C = AB: one A tile, one B tile, one C tile</text>
<rect x="78" y="206" width="92" height="210" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="78" y="272" width="92" height="64" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>
<text class="body" x="112" y="430">A</text>
<rect x="216" y="206" width="210" height="92" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="284" y="206" width="72" height="92" fill="#bbf7d0" stroke="#16a34a" stroke-width="2"/>
<text class="body" x="315" y="320">B</text>
<rect x="226" y="330" width="150" height="86" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="274" y="356" width="70" height="44" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
<text class="body" x="296" y="430">C</text>
<text class="body" x="182" y="262">x</text>
<text class="body" x="186" y="378">=</text>
<line class="arrow" x1="170" y1="302" x2="262" y2="378"/>
<line class="arrow" x1="320" y1="298" x2="320" y2="352"/>
<text class="small" x="70" y="474">一个 block 先把 A/B 的 tile 搬进 shared，tile 内元素在 dot-product 中被反复复用。</text>

<rect class="panel" x="495" y="110" width="410" height="400"/>
<text class="label" x="525" y="150">C = AAT: strided load becomes tile transpose</text>
<rect x="540" y="214" width="110" height="188" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="540" y="260" width="110" height="52" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>
<text class="body" x="588" y="430">A</text>
<rect x="706" y="214" width="110" height="188" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="744" y="214" width="34" height="188" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
<text class="body" x="732" y="430">A^T view</text>
<line class="arrow" x1="652" y1="286" x2="744" y2="286"/>
<text class="small" x="528" y="474">baseline 里第二个操作数按列取，warp 同拍地址跨很大；shared tile 先按 coalesced 方式搬入，再在片上转置使用。</text>

<rect class="panel" x="930" y="110" width="430" height="400"/>
<text class="label" x="960" y="150">Padding removes column-wise bank conflict</text>
<rect x="972" y="212" width="128" height="128" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<line class="grid" x1="1004" y1="212" x2="1004" y2="340"/><line class="grid" x1="1036" y1="212" x2="1036" y2="340"/><line class="grid" x1="1068" y1="212" x2="1068" y2="340"/>
<line class="grid" x1="972" y1="244" x2="1100" y2="244"/><line class="grid" x1="972" y1="276" x2="1100" y2="276"/><line class="grid" x1="972" y1="308" x2="1100" y2="308"/>
<rect x="1010" y="212" width="18" height="128" fill="#fca5a5" opacity="0.75"/>
<text class="small" x="982" y="366">32 x 32</text>
<text class="small" x="978" y="388">同列访问更容易映射到重复 bank</text>

<rect x="1170" y="212" width="144" height="128" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
<line class="grid" x1="1202" y1="212" x2="1202" y2="340"/><line class="grid" x1="1234" y1="212" x2="1234" y2="340"/><line class="grid" x1="1266" y1="212" x2="1266" y2="340"/><line class="grid" x1="1298" y1="212" x2="1298" y2="340"/>
<line class="grid" x1="1170" y1="244" x2="1314" y2="244"/><line class="grid" x1="1170" y1="276" x2="1314" y2="276"/><line class="grid" x1="1170" y1="308" x2="1314" y2="308"/>
<rect x="1278" y="212" width="20" height="128" fill="#c4b5fd" opacity="0.85"/>
<text class="small" x="1200" y="366">32 x 33</text>
<text class="small" x="1168" y="388">+1 列把下一行起点错开，列访问不再整齐撞同一 bank</text>
"""
    write("shared-memory-gemm-principle.svg", body)


def l2_window() -> None:
    body = """
<text class="title" x="44" y="48">L2 Access Window: Sliding Window over Set-Aside L2</text>
<text class="body" x="44" y="78">按 Best Practices Guide 的 sliding-window 图重画：频繁访问的数据窗口映射到 set-aside L2；窗口超过可保留容量时，固定 hitRatio 会开始 thrash。</text>

<rect class="panel" x="40" y="110" width="620" height="400"/>
<text class="label" x="72" y="150">Fixed hitRatio = 1.0</text>
<rect x="84" y="214" width="530" height="46" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="110" y="214" width="330" height="46" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
<text class="body" x="250" y="204">freq window in global memory</text>
<text class="small" x="118" y="244">persistent region</text>
<rect x="84" y="332" width="530" height="46" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="146" y="332" width="220" height="46" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2"/>
<text class="body" x="260" y="322">set-aside L2</text>
<line class="arrow" x1="270" y1="260" x2="270" y2="332"/>
<line class="arrow" x1="406" y1="260" x2="366" y2="332"/>
<text class="small" x="90" y="414">窗口大于 set-aside 时，整段都想“持久化”，最终在保留区里互相驱逐。</text>

<rect class="panel" x="700" y="110" width="660" height="400"/>
<text class="label" x="732" y="150">Tuned hitRatio</text>
<rect x="744" y="214" width="572" height="46" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="790" y="214" width="360" height="46" fill="#bbf7d0" stroke="#16a34a" stroke-width="2"/>
<text class="body" x="932" y="204">same freq window</text>
<rect x="744" y="332" width="572" height="46" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
<rect x="872" y="332" width="188" height="46" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2"/>
<text class="body" x="958" y="322">set-aside L2</text>
<line class="arrow" x1="872" y1="260" x2="872" y2="332"/>
<line class="arrow" x1="1060" y1="260" x2="1060" y2="332"/>
<text class="small" x="782" y="414">把真正想保留的热数据规模固定在可承受范围，其余访问仍按 streaming 处理，所以大窗口时更稳。</text>
"""
    write("l2-access-window-principle.svg", body)


def async_copy() -> None:
    body = """
<text class="title" x="44" y="48">Async Copy: global -&gt; register -&gt; shared vs cp.async</text>
<text class="body" x="44" y="78">同步路径先进寄存器再写 shared；cp.async 让 global 到 shared 的搬运协议显式化，更容易减少中转与重叠等待。</text>

<rect class="panel" x="40" y="110" width="630" height="400"/>
<text class="label" x="72" y="150">sync</text>
<rect class="cpu" x="80" y="220" width="130" height="64" rx="12" ry="12"/><text class="body" x="120" y="258">global</text>
<rect class="warn" x="280" y="220" width="130" height="64" rx="12" ry="12"/><text class="body" x="315" y="258">register</text>
<rect class="gpu" x="480" y="220" width="130" height="64" rx="12" ry="12"/><text class="body" x="523" y="258">shared</text>
<line class="arrow" x1="210" y1="252" x2="280" y2="252"/><line class="arrow" x1="410" y1="252" x2="480" y2="252"/>
<text class="small" x="136" y="340">load + store，两条路径都要走</text>

<rect class="panel" x="710" y="110" width="650" height="400"/>
<text class="label" x="742" y="150">async</text>
<rect class="cpu" x="760" y="220" width="130" height="64" rx="12" ry="12"/><text class="body" x="800" y="258">global</text>
<rect class="accent" x="1090" y="220" width="130" height="64" rx="12" ry="12"/><text class="body" x="1133" y="258">shared</text>
<line class="arrow" x1="890" y1="252" x2="1090" y2="252"/>
<text class="body" x="943" y="238">cp.async</text>
<text class="small" x="798" y="340">显式 commit / wait；拷贝粒度越大，收益通常越明显。</text>
"""
    write("async-copy-principle.svg", body)


def occupancy() -> None:
    body = """
<text class="title" x="44" y="48">Occupancy Sweep: What Actually Changes</text>
<text class="body" x="44" y="78">occupancy 是单 SM 同时驻留的活跃 warp 比例；真正决定它的常见约束是 registers、shared memory、block size。</text>

<rect class="panel" x="40" y="110" width="620" height="400"/>
<text class="label" x="74" y="150">Resource limits</text>
<rect class="gpu" x="100" y="220" width="120" height="70" rx="12" ry="12"/><text class="body" x="128" y="260">registers</text>
<rect class="warn" x="280" y="220" width="120" height="70" rx="12" ry="12"/><text class="body" x="317" y="260">shared</text>
<rect class="accent" x="460" y="220" width="120" height="70" rx="12" ry="12"/><text class="body" x="495" y="260">block size</text>
<line class="arrow" x1="220" y1="255" x2="280" y2="255"/><line class="arrow" x1="400" y1="255" x2="460" y2="255"/>
<text class="small" x="128" y="352">任何一个资源先触顶，active blocks / SM 就先降。</text>

<rect class="panel" x="700" y="110" width="660" height="400"/>
<text class="label" x="734" y="150">Latency hiding</text>
<rect class="hot" x="760" y="220" width="160" height="70" rx="12" ry="12"/><text class="body" x="800" y="260">few warps</text>
<rect class="gpu" x="1040" y="220" width="220" height="70" rx="12" ry="12"/><text class="body" x="1093" y="260">many ready warps</text>
<line class="arrow" x1="920" y1="255" x2="1040" y2="255"/>
<text class="small" x="804" y="352">occupancy 高不等于一定更快，但太低时更难靠切换 warp 隐藏延迟。</text>
"""
    write("occupancy-principle.svg", body)


def main() -> None:
    timing_baseline()
    pinned_pageable()
    multi_stream_overlap()
    offset_stride()
    shared_gemm()
    l2_window()
    async_copy()
    occupancy()


if __name__ == "__main__":
    main()
