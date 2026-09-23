"""Draw the nominal leading channel from the saved run, preserving scale."""
from qdmnp.pipeline import ROOT

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

out = ROOT / "results/article_single_mnp_c12p5_a4p5/tip_side_audit"
manifest = json.loads((out.parent / "manifest.json").read_text(encoding="utf-8"))
geom = manifest["identity"]["config"]["geometry"]
selection = manifest["state"]["selection"]
assert selection["best_channel"] == "side_long"
c, a, r = geom["c_nm"], geom["a_nm"], geom["qd_radius_nm"]
g = selection["best_gap_nm"]
R = a + r + g

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(10.8, 7.8), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 900), ylim=(650, 0), aspect="equal")
ax.axis("off")
ink, muted, gold, goldedge = "#20354b", "#64748b", "#edbd49", "#a57816"
blue, cyan, pink, guide = "#236cbd", "#dceefb", "#b62c69", "#9caabd"
scale, cx, cy = 15., 406., 304.
qx = cx + scale * R


def number(value):
    return f"{value:g}".replace(".", "{,}")


def text(x, y, label, **kw):
    return ax.text(x, y, label, color=kw.pop("color", ink),
                   fontsize=kw.pop("fontsize", 13), va="center", **kw)


def arrow(start, end, color=ink, lw=1.3, style="<->", size=11, **kw):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=size,
                            color=color, lw=lw, **kw)
    ax.add_patch(patch)
    return patch


text(52, 42, "Ведущий канал: side_long", fontsize=22, fontweight="bold")
text(52, 75, "КТ у экватора; поле параллельно большой оси сфероида", color=muted)

ax.add_patch(Ellipse((cx, cy), 2*a*scale, 2*c*scale, facecolor=gold,
                     edgecolor=goldedge, lw=2, zorder=2))
ax.add_patch(Circle((qx, cy), r*scale, facecolor=cyan, edgecolor=blue, lw=2, zorder=3))
ax.plot([cx, cx], [cy+c*scale+12, cy-c*scale-14], color=guide, lw=.8, ls=(0, (4, 4)), zorder=3)
arrow((cx, cy-c*scale-7), (cx, cy-c*scale-25), color=muted, style="-|>", size=9, lw=1)
text(cx+10, cy-c*scale-21, "z", color=muted)
ax.plot([cx-18, qx+73], [cy, cy], color=guide, lw=.8, ls=(0, (4, 4)), zorder=3)
arrow((qx+64, cy), (qx+85, cy), color=muted, style="-|>", size=9, lw=1)
text(qx+95, cy, "x", color=muted)
ax.scatter([cx, qx], [cy, cy], s=17, color=[ink, blue], zorder=5)
text(cx, 220, "МНЧ", ha="center", fontsize=17, fontweight="bold")
text(cx, 249, "Au", ha="center", fontsize=16)

# The double arrow denotes the axis of field oscillation, not propagation.
arrow((178, 362), (178, 207), color=pink, lw=3.6, size=19)
text(126, 277, r"$\mathbf{E}_0$", ha="center", fontsize=25, color=pink)
text(178, 392, "Поляризация", ha="center", color=pink)
text(178, 418, r"$\mathbf{E}_0\parallel z$", ha="center", fontsize=15, color=pink)

xd = cx-a*scale-42
ax.plot([xd-5, cx], [cy-c*scale, cy-c*scale], color=guide, lw=.8, zorder=1)
ax.plot([xd-5, cx-4], [cy, cy], color=guide, lw=.8, zorder=1)
arrow((xd, cy-2), (xd, cy-c*scale+2), size=10)
text(xd-15, cy-c*scale/2, f"$c={number(c)}$ нм", rotation=90, ha="center")
arrow((cx+2, cy+25), (cx+a*scale-2, cy+25), size=9, lw=1.2, zorder=4)
ax.plot([cx+a*scale, cx+a*scale], [cy+4, cy+29], color=ink, lw=.65, zorder=4)
text(cx+a*scale/2, cy+48, f"$a={number(a)}$ нм", ha="center", fontsize=11, zorder=5)

text(610, 244, "КТ", fontsize=17, fontweight="bold", color=blue)
text(610, 272, r"$r_{\mathrm{QD}}=" + number(r) + "$ нм", color=blue)
ax.plot([603, qx+22], [256, cy-20], color=blue, lw=1.1)

xmetal, xqd = cx+a*scale, qx-r*scale
xmid = (xmetal+xqd)/2
ax.plot([xmetal, xqd], [cy, cy], color=pink, lw=2.5, zorder=6)
ax.plot([xmid, xmid, 550], [cy-4, 178, 178], color=pink, lw=1, zorder=4)
text(560, 175, f"$g={number(g)}$ нм", fontsize=14, color=pink)
text(560, 199, "Зазор между поверхностями", fontsize=11, color=muted)

ydim = 546
ax.plot([cx, cx], [cy+c*scale+8, ydim+8], color=guide, lw=.9, ls=(0, (4, 4)))
ax.plot([qx, qx], [cy+r*scale+8, ydim+8], color=guide, lw=.9, ls=(0, (4, 4)))
arrow((cx+2, ydim), (qx-2, ydim))
text((cx+qx)/2, ydim+29, r"$R=a+r_{\mathrm{QD}}+g=" + number(R) + "$ нм",
     ha="center", fontsize=14)

text(631, 394, r"$\mathbf{E}_0\perp\mathbf{R}$", fontsize=21, color=pink)
text(631, 430, "Поле касательно", fontsize=12, color=muted)
text(631, 451, "боковой поверхности МНЧ", fontsize=12, color=muted)
text(52, 622, "Сечение xz в масштабе.  Полуось b = a направлена перпендикулярно рисунку.",
     fontsize=11, color=muted)

for extension in ("png", "svg", "pdf"):
    target = out / f"side_long_geometry.{extension}"
    fig.savefig(target, dpi=170, facecolor="white")
    print(target)
plt.close(fig)
