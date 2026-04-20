"""Generate silu_plot.svg — a clean plot of the SiLU activation for the notes."""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 600)
silu = x / (1 + np.exp(-x))
relu = np.maximum(0, x)

# Analytical minimum of SiLU: solve σ(x) + x·σ(x)·(1−σ(x)) = 0, numerically at x ≈ −1.2785
x_min = -1.2785
y_min = x_min / (1 + np.exp(-x_min))

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)

# Reference curves first (so SiLU draws on top)
ax.plot(x, x, linewidth=1, color="#bbb", linestyle=":",
        label="y = x   (SiLU's right-tail asymptote)", alpha=0.9)
ax.plot(x, relu, linewidth=1.3, color="#888", linestyle="--",
        label="ReLU(x) = max(0, x)", alpha=0.85)

# Main SiLU curve
ax.plot(x, silu, linewidth=2.6, color="#1f77b4", label="SiLU(x) = x · σ(x)")

# Mark the global minimum
ax.plot([x_min], [y_min], "o", color="#e74c3c", markersize=8, zorder=5)
ax.annotate(f"global min\n≈ ({x_min:.2f}, {y_min:.3f})",
            xy=(x_min, y_min),
            xytext=(-4.7, -1.15),
            fontsize=10, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))

# Reference axes at 0
ax.axhline(0, color="#555", linewidth=0.8)
ax.axvline(0, color="#555", linewidth=0.8)

# Regime annotations
ax.text(3.2, 2.2, "linear regime\nSiLU(x) → x",
        fontsize=10, color="#2c3e50", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4f8fb", ec="#cfdbe6"))
ax.text(-5.8, 0.45, "gate-closed regime\nSiLU(x) → 0  (slowly,\nfrom slightly below)",
        fontsize=10, color="#2c3e50", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4f8fb", ec="#cfdbe6"))

# Styling
ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 6.2)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("activation output", fontsize=12)
ax.set_title("SiLU — smooth, unbounded above, dips slightly negative",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(True, alpha=0.28)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

plt.tight_layout()
out = "/Users/Chenjunyu/Desktop/Work/llama3/notes_assets/silu_plot.svg"
plt.savefig(out, format="svg", bbox_inches="tight")
plt.close()
print("wrote", out)
