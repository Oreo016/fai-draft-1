"""Dependency-free SVG charts for experiment outputs."""

from collections import defaultdict
from html import escape
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = ["#2563eb", "#0f766e", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]
WIDTH, HEIGHT = 1120, 680


def create_search_animation(frames: list[dict[str, Any]], output: Path) -> Path:
    """Render the first run episode as an animated, labeled SAR grid."""
    if not frames:
        raise ValueError("search animation needs at least one captured frame")
    images: list[Image.Image] = []
    font = ImageFont.load_default()
    for frame in frames:
        size = int(frame["size"])
        cell, margin, top = 64, 36, 130
        width, height = margin * 2 + size * cell, top + size * cell + 42
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)
        draw.text((margin, 20), f"SAR search | {frame['method']} | seed {frame['seed']}", fill="#0f172a", font=font)
        draw.text((margin, 42), "Blue circle: Scout   Purple circle: Carrier   Teal circle: Medic", fill="#334155", font=font)
        draw.text((margin, 60), "Yellow dot: victim   Green dot: rescued   Red triangle: active hazard   Gray triangle: inactive hazard", fill="#334155", font=font)
        draw.text((margin, 82), f"Step {frame['step']}   {frame['message']}", fill="#334155", font=font)
        origin_y = top
        for row in range(size):
            for col in range(size):
                x0, y0 = margin + col * cell, origin_y + row * cell
                draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill="#ffffff", outline="#cbd5e1", width=1)
        for position, status in frame["victims"]:
            row, col = position
            color = "#16a34a" if status == "rescued" else "#f59e0b"
            draw.ellipse((margin + col * cell + 20, origin_y + row * cell + 20,
                          margin + col * cell + 44, origin_y + row * cell + 44), fill=color)
        for position, status in frame["hazards"]:
            row, col = position
            x, y = margin + col * cell, origin_y + row * cell
            color = "#ef4444" if status == "active" else "#94a3b8"
            draw.polygon(((x + 32, y + 12), (x + 52, y + 50), (x + 12, y + 50)), fill=color)
        for agent_id, (position, role) in enumerate(zip(frame["positions"], frame["roles"])):
            row, col = position
            x, y = margin + col * cell, origin_y + row * cell
            color = {"scout": "#2563eb", "carrier": "#7c3aed", "medic": "#0891b2"}.get(role, "#475569")
            draw.ellipse((x + 15, y + 15, x + 49, y + 49), fill=color, outline="white", width=2)
            draw.text((x + 27, y + 26), str(agent_id), fill="white", font=font)
        action_text = " | ".join(f"{role}: {action}" for role, action in zip(frame["roles"], frame["actions"]))
        draw.text((margin, origin_y + size * cell + 12), action_text or "Episode start", fill="#334155", font=font)
        images.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=550, loop=0, optimize=False)
    return output


def create_visualizations(
    episodes: list[dict[str, Any]], summary: dict[str, Any], output: Path
) -> list[Path]:
    """Write SVG plots and an HTML gallery using only the Python standard library."""
    output.mkdir(parents=True, exist_ok=True)
    charts: list[tuple[str, str]] = []
    conditions = summary["conditions"]
    charts.append(("01-condition-returns.svg", _condition_bars(
        conditions, "mean_task_return", "task_return_confidence_interval",
        "Mean discounted task return by condition", "Return", decimals=2,
    )))
    charts.append(("02-message-volume.svg", _condition_bars(
        conditions, "mean_messages", None, "Mean messages sent by condition", "Messages", decimals=1,
    )))
    charts.append(("03-return-distributions.svg", _box_plot(
        episodes, "discounted_task_return", "Episode return distributions", "Discounted task return",
    )))
    charts.append(("04-return-heatmap.svg", _heatmap(
        episodes, "discounted_task_return", "Episode returns by method and seed", "Return",
    )))
    charts.append(("05-communication-return-tradeoff.svg", _tradeoff(
        conditions, "Communication and return trade-off",
    )))
    charts.append(("06-cost-sensitivity.svg", _cost_sensitivity(
        conditions, "Communication cost sensitivity",
    )))
    charts.append(("07-hypothesis-targets.svg", _hypotheses(
        summary["hypotheses"], "Observed hypothesis metrics and targets",
    )))
    charts.append(("08-diagnostics.svg", _diagnostics(
        summary["diagnostics"], "Held-out valuation and policy diagnostics",
    )))
    charts.append(("09-failure-outcomes.svg", _failure_chart(
        episodes, "Episode outcome counts by method",
    )))
    charts.append(("10-estimator-uncertainty.svg", _uncertainty(
        conditions, "Mean estimator uncertainty by condition",
    )))
    charts.append(("11-victims-rescued.svg", _condition_bars(
        conditions, "mean_rescued_victims", "rescued_victims_confidence_interval",
        "Mean victims rescued by condition", "Victims rescued", decimals=2,
    )))

    paths = []
    for name, svg in charts:
        path = output / name
        path.write_text(svg, encoding="utf-8")
        paths.append(path)
    cards = "\n".join(
        f'<article><h2>{escape(_title(name))}</h2><img src="{escape(name)}" alt="{escape(_title(name))}"></article>'
        for name, _ in charts
    )
    (output / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>SAR experiment graphs</title><style>body{font:16px system-ui;margin:2rem;background:#f5f7fb;color:#172033}"
        "h1{max-width:1100px;margin:0 auto 1.5rem}main{max-width:1160px;margin:auto;display:grid;gap:1.5rem}"
        "article{background:white;padding:1rem;border-radius:12px;box-shadow:0 2px 12px #14213d12}"
        "img{display:block;width:100%;height:auto}h2{font-size:1.1rem}</style><h1>SAR experiment graphical results</h1>"
        f"<main>{cards}</main></html>", encoding="utf-8",
    )
    (output / "README.txt").write_text(
        "Open index.html in a browser to view the chart gallery. Each chart is a standalone, scalable SVG.\n"
        "Charts cover condition comparisons, return distributions, per-seed patterns, trade-offs, cost sensitivity, "
        "hypothesis targets, held-out diagnostics, episode outcomes, and estimator uncertainty.\n",
        encoding="utf-8",
    )
    return paths


def _title(filename: str) -> str:
    return filename[3:-4].replace("-", " ").title()


def _svg(title: str, body: str, *, width: int = WIDTH, height: int = HEIGHT) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img"><rect width="100%" height="100%" fill="#fff"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#273449}.grid{stroke:#e4e9f0;stroke-width:1}'
        '.axis{stroke:#68778b;stroke-width:1.2}.muted{fill:#68778b;font-size:12px}</style>'
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-size="21" font-weight="700">{escape(title)}</text>'
        f'{body}</svg>'
    )


def _text(x: float, y: float, value: Any, *, size: int = 12, anchor: str = "middle", rotate: int = 0) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{transform}>{escape(str(value))}</text>'


def _condition_bars(rows: list[dict[str, Any]], metric: str, interval: str | None,
                    title: str, ylabel: str, decimals: int = 1) -> str:
    values = [float(row[metric] or 0) for row in rows]
    low = min([0.0, *values])
    high = max([0.0, *values])
    if interval:
        bounds = [row[interval] for row in rows if row.get(interval) is not None]
        low = min([low, *(float(pair[0]) for pair in bounds)])
        high = max([high, *(float(pair[1]) for pair in bounds)])
    spread = max(high - low, 1.0)
    low -= spread * .12
    high += spread * .15
    left, right, top, bottom = 90, 1090, 80, 500
    y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    body = []
    for tick in range(6):
        val = low + (high-low)*tick/5
        yy = y(val)
        body.extend([f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>',
                     _text(left-10, yy+4, f"{val:.{decimals}f}", anchor="end")])
    slot = (right-left)/max(len(rows), 1)
    bar_w = min(54, slot*.58)
    zero = y(0)
    for i, row in enumerate(rows):
        x = left + (i+.5)*slot
        yy = y(float(row[metric] or 0))
        top_y, h = min(yy, zero), max(abs(zero-yy), 1)
        body.append(f'<rect x="{x-bar_w/2:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{COLORS[i%len(COLORS)]}"/>')
        if interval and row.get(interval) is not None:
            lo, hi = map(float, row[interval]); cap = 8
            body.append(f'<line class="axis" x1="{x:.1f}" y1="{y(lo):.1f}" x2="{x:.1f}" y2="{y(hi):.1f}"/><line class="axis" x1="{x-cap/2:.1f}" y1="{y(lo):.1f}" x2="{x+cap/2:.1f}" y2="{y(lo):.1f}"/><line class="axis" x1="{x-cap/2:.1f}" y1="{y(hi):.1f}" x2="{x+cap/2:.1f}" y2="{y(hi):.1f}"/>')
        body.append(_text(x, 525, f"{row['method']} λ={row['cost_weight']:g}", size=10, rotate=-28))
        body.append(_text(x, max(top+12, top_y-8), f"{float(row[metric] or 0):.{decimals}f}", size=11))
    body.extend([f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                 f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
                 _text(22, 290, ylabel, rotate=-90)])
    return _svg(title, "".join(body))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered: return 0.0
    at = (len(ordered)-1)*q; lo = math.floor(at); hi = math.ceil(at)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)


def _box_plot(episodes: list[dict[str, Any]], key: str, title: str, ylabel: str) -> str:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in episodes: groups[(row["method"], float(row["cost_weight"]))].append(float(row[key]))
    items = sorted(groups.items()); all_values = [v for values in groups.values() for v in values]
    low, high = min(all_values, default=0), max(all_values, default=1)
    pad = max((high-low)*.1, .5); low -= pad; high += pad
    left,right,top,bottom=90,1090,80,500
    y=lambda v: bottom-(v-low)/(high-low)*(bottom-top)
    body=[]
    for t in range(6):
        v=low+(high-low)*t/5; yy=y(v)
        body += [f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>',_text(left-8,yy+4,f"{v:.1f}",anchor="end")]
    slot=(right-left)/max(len(items),1)
    for i,((method,cost),vals) in enumerate(items):
        x=left+(i+.5)*slot; q1,q2,q3=(_quantile(vals,q) for q in (.25,.5,.75)); lo,hi=min(vals),max(vals); bw=min(46,slot*.5)
        body += [f'<line class="axis" x1="{x}" y1="{y(lo):.1f}" x2="{x}" y2="{y(hi):.1f}"/>',f'<rect x="{x-bw/2:.1f}" y="{y(q3):.1f}" width="{bw:.1f}" height="{max(1,y(q1)-y(q3)):.1f}" fill="{COLORS[i%len(COLORS)]}" fill-opacity=".72" stroke="#263449"/>',f'<line stroke="#172033" stroke-width="3" x1="{x-bw/2:.1f}" y1="{y(q2):.1f}" x2="{x+bw/2:.1f}" y2="{y(q2):.1f}"/>',_text(x,525,f"{method} λ={cost:g}",size=10,rotate=-28)]
    body += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',_text(22,290,ylabel,rotate=-90)]
    return _svg(title,"".join(body))


def _heatmap(episodes: list[dict[str, Any]], key: str, title: str, legend: str) -> str:
    methods=sorted({r["method"] for r in episodes}); seeds=sorted({int(r["seed"]) for r in episodes})
    table={(r["method"],int(r["seed"])):float(r[key]) for r in episodes}
    values=list(table.values()); low,high=min(values,default=0),max(values,default=1); span=max(high-low,1e-9)
    x0,y0,cw,ch=150,105, min(78,850/max(len(seeds),1)), min(56,475/max(len(methods),1))
    body=[]
    for j,seed in enumerate(seeds): body.append(_text(x0+(j+.5)*cw,y0-12,seed))
    for i,method in enumerate(methods):
        body.append(_text(x0-12,y0+(i+.65)*ch,method,anchor="end"))
        for j,seed in enumerate(seeds):
            v=table.get((method,seed),0); f=(v-low)/span
            color=f"rgb({int(239-170*f)},{int(246-105*f)},{int(255-20*f)})"
            body.append(f'<rect x="{x0+j*cw}" y="{y0+i*ch}" width="{cw-2}" height="{ch-2}" rx="3" fill="{color}" stroke="#d8e0ea"><title>{escape(method)}, seed {seed}: {v:.3f}</title></rect>')
            body.append(_text(x0+(j+.5)*cw,y0+(i+.62)*ch,f"{v:.1f}",size=10))
    body.append(_text(600,600,f"{legend}: {low:.2f} to {high:.2f}",size=13))
    return _svg(title,"".join(body))


def _tradeoff(rows: list[dict[str, Any]], title: str) -> str:
    xs=[float(r["mean_messages"]) for r in rows]; ys=[float(r["mean_task_return"]) for r in rows]
    xmax=max(xs,default=1)*1.15 or 1; ymin=min(ys,default=0); ymax=max(ys,default=1); pad=max((ymax-ymin)*.12,.2); ymin-=pad; ymax+=pad
    l,r,t,b=100,1060,85,520; X=lambda v:l+v/xmax*(r-l); Y=lambda v:b-(v-ymin)/(ymax-ymin)*(b-t); body=[]
    for i in range(6):
        v=ymin+(ymax-ymin)*i/5; yy=Y(v); body += [f'<line class="grid" x1="{l}" y1="{yy:.1f}" x2="{r}" y2="{yy:.1f}"/>',_text(l-10,yy+4,f"{v:.1f}",anchor="end")]
        xv=xmax*i/5; xx=X(xv); body += [f'<line class="grid" x1="{xx:.1f}" y1="{t}" x2="{xx:.1f}" y2="{b}"/>',_text(xx,b+24,f"{xv:.0f}")]
    for i,row in enumerate(rows):
        x,y=X(float(row["mean_messages"])),Y(float(row["mean_task_return"]))
        body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{COLORS[i%len(COLORS)]}"><title>{escape(row["method"])} λ={row["cost_weight"]}: {row["mean_messages"]:.1f} messages, return {row["mean_task_return"]:.2f}</title></circle>',_text(x+8,y-10,f"{row['method']} λ={row['cost_weight']:g}",size=10,anchor="start")]
    body += [f'<line class="axis" x1="{l}" y1="{b}" x2="{r}" y2="{b}"/>',f'<line class="axis" x1="{l}" y1="{t}" x2="{l}" y2="{b}"/>',_text(580,615,"Mean messages per episode"),_text(22,300,"Mean discounted task return",rotate=-90)]
    return _svg(title,"".join(body))


def _cost_sensitivity(rows: list[dict[str, Any]], title: str) -> str:
    selected=sorted((r for r in rows if r["method"] in {"aware_nonmyopic","aware_myopic","agnostic_nonmyopic"}),key=lambda r:(r["method"],r["cost_weight"]))
    vals=[float(r["positive_voc_communication_rate"] or 0) for r in selected]; ymax=max(vals+[.1])*1.2
    l,r,t,b=100,1060,85,510; X=lambda v:l+v/.5*(r-l); Y=lambda v:b-v/ymax*(b-t); body=[]
    for i in range(6):
        v=ymax*i/5; yy=Y(v); body += [f'<line class="grid" x1="{l}" y1="{yy:.1f}" x2="{r}" y2="{yy:.1f}"/>',_text(l-10,yy+4,f"{v:.2f}",anchor="end")]
    groups:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in selected: groups[row["method"]].append(row)
    for i,(method,series) in enumerate(sorted(groups.items())):
        series.sort(key=lambda row:row["cost_weight"]); pts=[(X(float(row["cost_weight"])),Y(float(row["positive_voc_communication_rate"] or 0))) for row in series]
        body.append(f'<polyline fill="none" stroke="{COLORS[i]}" stroke-width="3" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}"/>')
        for x,y in pts: body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[i]}"/>')
        body.append(f'<line x1="{120+i*240}" y1="570" x2="{155+i*240}" y2="570" stroke="{COLORS[i]}" stroke-width="3"/>'+_text(164+i*240,574,method,anchor="start",size=11))
    body += [f'<line class="axis" x1="{l}" y1="{b}" x2="{r}" y2="{b}"/>',_text(X(0),b+24,"0.0"),_text(X(.5),b+24,"0.5"),_text(580,620,"Communication cost weight λ"),_text(22,300,"Positive VoC communication rate",rotate=-90)]
    return _svg(title,"".join(body))


def _hypotheses(hypotheses: dict[str, Any], title: str) -> str:
    metrics=[]
    for name,obj in hypotheses.items():
        keys=[k for k in obj if k.endswith("_target") or k in {"gain_target","reduction_target"}]
        targets=[obj[k] for k in keys if isinstance(obj[k],(int,float))]
        observed=next((obj[k] for k in ("observed_fraction","relative_rate_reduction","agreement_gain","communication_reduction_at_matched_return") if k in obj and isinstance(obj[k],(int,float))),None)
        if observed is not None and targets: metrics.append((name.split("_")[0],float(observed),float(targets[0])))
    l,r,t,b=260,1060,100,500; maxv=max([1.0]+[max(a,c) for _,a,c in metrics]); body=[]
    for i,(name,value,target) in enumerate(metrics):
        y=145+i*82; body.append(_text(l-15,y+6,name,anchor="end",size=14)); body.append(f'<rect x="{l}" y="{y-12}" width="{(r-l)*max(0,value)/maxv:.1f}" height="24" rx="4" fill="{COLORS[i]}"/>'); tx=l+(r-l)*target/maxv
        body.append(f'<line x1="{tx:.1f}" y1="{y-22}" x2="{tx:.1f}" y2="{y+22}" stroke="#172033" stroke-width="3"/>'); body.append(_text(min(r, l+(r-l)*max(0,value)/maxv+12),y+5,f"{value:.2f} (target {target:.2f})",anchor="start",size=12))
    body.append(_text((l+r)/2,570,"Observed value; vertical marker shows target",size=13))
    return _svg(title,"".join(body))


def _diagnostics(d: dict[str, Any], title: str) -> str:
    fields=[("Receiver regret", "receiver_regret_mean"),("Pair regret","pair_regret_mean"),("VoC MAE","voc_mae_nonmyopic"),("Filter recall","filter_recall"),("Useful-message rate","useful_message_rate"),("Myopic oracle agreement","agreement_myopic_to_oracle"),("Nonmyopic oracle agreement","agreement_nonmyopic_to_oracle")]
    values=[(label,float(d[key])) for label,key in fields if isinstance(d.get(key),(int,float))]
    maximum=max([1.0]+[v for _,v in values]); l,r=300,1040; body=[]
    for i,(label,value) in enumerate(values):
        y=112+i*60; body += [_text(l-16,y+5,label,anchor="end",size=13),f'<rect x="{l}" y="{y-14}" width="{(r-l)*max(0,value)/maximum:.1f}" height="28" rx="4" fill="{COLORS[i%len(COLORS)]}"/>',_text(min(r,l+(r-l)*max(0,value)/maximum+10),y+5,f"{value:.3f}",anchor="start",size=12)]
    return _svg(title,"".join(body))


def _failure_chart(episodes: list[dict[str, Any]], title: str) -> str:
    groups:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in episodes: groups[row["method"]].append(row)
    rows=sorted(groups.items()); maxn=max([1]+[len(v) for _,v in rows]); left,right,top,bottom=210,1060,90,520; body=[]
    for i,(method,items) in enumerate(rows):
        y=120+i*42; failures=sum(row.get("failure_case") is not None for row in items); success=len(items)-failures; scale=(right-left)/maxn
        body += [_text(left-12,y+5,method,anchor="end"),f'<rect x="{left}" y="{y-12}" width="{success*scale:.1f}" height="24" fill="#0f766e"><title>{success} without recorded failure</title></rect>',f'<rect x="{left+success*scale:.1f}" y="{y-12}" width="{failures*scale:.1f}" height="24" fill="#dc6b3f"><title>{failures} recorded failure cases</title></rect>',_text(right+20,y+5,f"{success} / {failures}",anchor="start",size=11)]
    body += ['<rect x="300" y="570" width="16" height="16" fill="#0f766e"/>',_text(324,583,"No failure case",anchor="start"),'<rect x="520" y="570" width="16" height="16" fill="#dc6b3f"/>',_text(544,583,"Recorded failure",anchor="start")]
    return _svg(title,"".join(body))


def _uncertainty(rows: list[dict[str, Any]], title: str) -> str:
    selected=[r for r in rows if r.get("mean_candidate_estimator_standard_error") is not None or r.get("mean_selected_estimator_standard_error") is not None]
    maximum=max([.01]+[float(r.get(key) or 0) for r in selected for key in ("mean_candidate_estimator_standard_error","mean_selected_estimator_standard_error")]); left,right,top,bottom=280,1040,90,520; body=[]
    for i,row in enumerate(selected):
        y=115+i*42; body.append(_text(left-12,y+5,f"{row['method']} λ={row['cost_weight']:g}",anchor="end",size=11))
        for j,key in enumerate(("mean_candidate_estimator_standard_error","mean_selected_estimator_standard_error")):
            value=float(row.get(key) or 0); yy=y-10+j*14; color=COLORS[j]
            body.append(f'<rect x="{left}" y="{yy}" width="{(right-left)*value/maximum:.1f}" height="11" fill="{color}"/>')
    body += ['<rect x="330" y="570" width="15" height="15" fill="#2563eb"/>',_text(353,583,"Candidate estimate SE",anchor="start"),'<rect x="560" y="570" width="15" height="15" fill="#0f766e"/>',_text(583,583,"Selected estimate SE",anchor="start")]
    if not selected: body.append(_text(560,300,"No estimator uncertainty values were recorded",size=15))
    return _svg(title,"".join(body))
