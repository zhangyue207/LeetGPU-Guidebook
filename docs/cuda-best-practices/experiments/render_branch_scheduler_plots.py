from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets"
OUT_PATH = ASSET_DIR / "branch-scheduler-performance.svg"

DIVERGENCE_ROWS = [
    ("uniform_true", 0.282176),
    ("warp_aligned_half", 0.547104),
    ("checkerboard", 0.547136),
    ("random_50", 0.541152),
    ("random_10", 0.546048),
]

BRANCH_ROWS = {
    "uniform": [
        (1, 4.621152, 4.601856),
        (2, 7.862464, 7.859360),
        (4, 14.746464, 14.771936),
        (8, 28.789888, 28.832064),
        (16, 57.150623, 57.223553),
        (32, 113.888641, 114.028481),
    ],
    "checkerboard": [
        (1, 7.079968, 9.205568),
        (2, 9.536512, 15.782144),
        (4, 15.610944, 30.043585),
        (8, 29.252993, 59.092449),
        (16, 57.501759, 117.425377),
        (32, 114.167969, 234.219620),
    ],
    "random_50": [
        (1, 7.076672, 8.879040),
        (2, 9.283616, 15.606432),
        (4, 15.418112, 29.904352),
        (8, 29.235647, 58.965633),
        (16, 57.389278, 117.315872),
        (32, 113.877663, 234.116455),
    ],
}

SCHEDULER_ROWS = {
    "dep_chain": [
        (0, 1.0, 0.996576),
        (8192, 1.0, 0.996448),
        (32768, 0.5, 1.129664),
    ],
    "ilp4": [
        (0, 1.0, 0.492160),
        (8192, 1.0, 0.492352),
        (32768, 0.5, 0.529792),
    ],
}


def fmt(value):
    return f"{value:.3f}"


def map_x(index, total, x0, width):
    if total == 1:
        return x0 + width / 2
    return x0 + width * index / (total - 1)


def map_y(value, vmin, vmax, y0, height):
    span = vmax - vmin
    if span <= 1e-9:
        span = 1.0
    return y0 + height - (value - vmin) / span * height


def svg_text(x, y, text, cls, anchor="start"):
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{safe}</text>'


def svg_multiline_text(x, y, lines, cls, anchor="start", line_height=18):
    parts = [f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">']
    for idx, line in enumerate(lines):
        safe = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        dy = "0" if idx == 0 else str(line_height)
        parts.append(f'<tspan x="{x}" dy="{dy}">{safe}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def draw_panel_frame(parts, x, y, w, h, title, subtitle):
    parts.append(f'<rect class="panel" x="{x}" y="{y}" width="{w}" height="{h}" />')
    parts.append(svg_text(x + 20, y + 34, title, "label"))
    parts.append(svg_text(x + 20, y + 60, subtitle, "small"))


def draw_divergence_panel(parts, x, y, w, h):
    draw_panel_frame(parts, x, y, w, h, "warp_divergence", "柱越高，说明分歧导致的时间代价越大")
    chart_x = x + 60
    chart_y = y + 100
    chart_w = w - 90
    chart_h = h - 150
    vmax = max(value for _, value in DIVERGENCE_ROWS) * 1.10

    for step in range(5):
        value = vmax * step / 4
        yy = map_y(value, 0.0, vmax, chart_y, chart_h)
        parts.append(f'<line class="grid" x1="{chart_x}" y1="{yy}" x2="{chart_x + chart_w}" y2="{yy}" />')
        parts.append(svg_text(chart_x - 10, yy + 5, fmt(value), "small", "end"))

    bar_w = chart_w / len(DIVERGENCE_ROWS) * 0.62
    for idx, (name, value) in enumerate(DIVERGENCE_ROWS):
        cx = map_x(idx, len(DIVERGENCE_ROWS), chart_x + 40, chart_w - 80)
        yy = map_y(value, 0.0, vmax, chart_y, chart_h)
        color = "#0f766e" if idx == 0 else "#dc2626"
        parts.append(
            f'<rect x="{cx - bar_w / 2}" y="{yy}" width="{bar_w}" height="{chart_y + chart_h - yy}" fill="{color}" rx="10" ry="10" />'
        )
        parts.append(svg_text(cx, yy - 10, fmt(value), "small", "middle"))
        label_lines = {
            "uniform_true": ["uniform", "true"],
            "warp_aligned_half": ["warp_aligned", "half"],
            "checkerboard": ["checkerboard"],
            "random_50": ["random", "50"],
            "random_10": ["random", "10"],
        }[name]
        parts.append(svg_multiline_text(cx, chart_y + chart_h + 20, label_lines, "small", "middle", 16))

    parts.append(
        svg_multiline_text(
            chart_x + chart_w / 2,
            y + h - 34,
            ["uniform_true 对比", "不同 lane pattern"],
            "body",
            "middle",
            18,
        )
    )


def draw_branch_panel(parts, x, y, w, h):
    draw_panel_frame(parts, x, y, w, h, "branch vs compute_both", "以 random_50 为主线，展示 body 变长时两者差距")
    chart_x = x + 64
    chart_y = y + 104
    chart_w = w - 94
    chart_h = h - 160
    ops = [row[0] for row in BRANCH_ROWS["random_50"]]
    series = {
        "branch": [row[1] for row in BRANCH_ROWS["random_50"]],
        "compute_both": [row[2] for row in BRANCH_ROWS["random_50"]],
    }
    colors = {"branch": "#2563eb", "compute_both": "#d97706"}
    vmax = max(max(values) for values in series.values()) * 1.08

    for step in range(5):
        value = vmax * step / 4
        yy = map_y(value, 0.0, vmax, chart_y, chart_h)
        parts.append(f'<line class="grid" x1="{chart_x}" y1="{yy}" x2="{chart_x + chart_w}" y2="{yy}" />')
        parts.append(svg_text(chart_x - 10, yy + 5, fmt(value), "small", "end"))

    xs = [map_x(i, len(ops), chart_x + 20, chart_w - 40) for i in range(len(ops))]
    for label, values in series.items():
        points = []
        for idx, value in enumerate(values):
            px = xs[idx]
            py = map_y(value, 0.0, vmax, chart_y, chart_h)
            points.append((px, py))
            parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{colors[label]}" />')
            parts.append(svg_text(px, py - 10, fmt(value), "small", "middle"))
        point_str = " ".join(f"{px},{py}" for px, py in points)
        parts.append(f'<polyline fill="none" stroke="{colors[label]}" stroke-width="3.5" points="{point_str}" />')

    for idx, op in enumerate(ops):
        parts.append(svg_text(xs[idx], chart_y + chart_h + 24, str(op), "small", "middle"))

    parts.append(f'<rect x="{chart_x + chart_w - 180}" y="{chart_y + 8}" width="14" height="14" fill="#2563eb" rx="3" ry="3" />')
    parts.append(svg_text(chart_x + chart_w - 158, chart_y + 20, "branch", "small"))
    parts.append(f'<rect x="{chart_x + chart_w - 180}" y="{chart_y + 34}" width="14" height="14" fill="#d97706" rx="3" ry="3" />')
    parts.append(svg_text(chart_x + chart_w - 158, chart_y + 46, "compute_both", "small"))
    parts.append(svg_text(chart_x + chart_w / 2, y + h - 18, "body_ops", "body", "middle"))


def draw_scheduler_panel(parts, x, y, w, h):
    draw_panel_frame(parts, x, y, w, h, "scheduler_latency_hiding", "对比 dep_chain 和 ilp4 在 occupancy 下降时的变化")
    chart_x = x + 64
    chart_y = y + 104
    chart_w = w - 94
    chart_h = h - 160
    labels = ["0KB / occ=1.0", "8KB / occ=1.0", "32KB / occ=0.5"]
    series = {
        "dep_chain": [row[2] for row in SCHEDULER_ROWS["dep_chain"]],
        "ilp4": [row[2] for row in SCHEDULER_ROWS["ilp4"]],
    }
    colors = {"dep_chain": "#dc2626", "ilp4": "#0f766e"}
    vmax = max(max(values) for values in series.values()) * 1.10

    for step in range(5):
        value = vmax * step / 4
        yy = map_y(value, 0.0, vmax, chart_y, chart_h)
        parts.append(f'<line class="grid" x1="{chart_x}" y1="{yy}" x2="{chart_x + chart_w}" y2="{yy}" />')
        parts.append(svg_text(chart_x - 10, yy + 5, fmt(value), "small", "end"))

    xs = [map_x(i, len(labels), chart_x + 20, chart_w - 40) for i in range(len(labels))]
    for label, values in series.items():
        points = []
        for idx, value in enumerate(values):
            px = xs[idx]
            py = map_y(value, 0.0, vmax, chart_y, chart_h)
            points.append((px, py))
            parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{colors[label]}" />')
            parts.append(svg_text(px, py - 10, fmt(value), "small", "middle"))
        point_str = " ".join(f"{px},{py}" for px, py in points)
        parts.append(f'<polyline fill="none" stroke="{colors[label]}" stroke-width="3.5" points="{point_str}" />')

    for idx, label in enumerate(labels):
        first, second = label.split(" / ")
        parts.append(svg_multiline_text(xs[idx], chart_y + chart_h + 20, [first, second], "small", "middle", 16))

    parts.append(f'<rect x="{chart_x + chart_w - 180}" y="{chart_y + 8}" width="14" height="14" fill="#dc2626" rx="3" ry="3" />')
    parts.append(svg_text(chart_x + chart_w - 158, chart_y + 20, "dep_chain", "small"))
    parts.append(f'<rect x="{chart_x + chart_w - 180}" y="{chart_y + 34}" width="14" height="14" fill="#0f766e" rx="3" ry="3" />')
    parts.append(svg_text(chart_x + chart_w - 158, chart_y + 46, "ilp4", "small"))
    parts.append(
        svg_multiline_text(
            chart_x + chart_w / 2,
            y + h - 34,
            ["extra shared memory", "/ occupancy"],
            "body",
            "middle",
            18,
        )
    )


def main():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="900" viewBox="0 0 1700 900">',
        "<defs><style>",
        ".bg { fill: #f8fafc; }",
        ".panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 2; rx: 20; ry: 20; }",
        '.title { font: 700 34px "DejaVu Sans", sans-serif; fill: #0f172a; }',
        '.subtitle { font: 400 19px "DejaVu Sans", sans-serif; fill: #475569; }',
        '.label { font: 700 20px "DejaVu Sans", sans-serif; fill: #1e293b; }',
        '.body { font: 400 18px "DejaVu Sans", sans-serif; fill: #334155; }',
        '.small { font: 400 15px "DejaVu Sans", sans-serif; fill: #64748b; }',
        ".grid { stroke: #e2e8f0; stroke-width: 1.5; }",
        "</style></defs>",
        '<rect class="bg" x="0" y="0" width="1700" height="900" />',
        '<text class="title" x="56" y="60">Branch / Scheduler 数据图</text>',
        '<text class="subtitle" x="56" y="94">基于 README 当前记录的 A100 测量值，补一张更直观的对比图。</text>',
    ]

    draw_divergence_panel(parts, 40, 130, 500, 700)
    draw_branch_panel(parts, 570, 130, 540, 700)
    draw_scheduler_panel(parts, 1140, 130, 520, 700)

    parts.append("</svg>")
    OUT_PATH.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
