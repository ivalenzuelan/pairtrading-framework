import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

fig, ax = plt.subplots(figsize=(10, 8))

# Components
components = {
    "Long-run Equilibrium": (0.5, 0.8),
    "Error Correction": (0.2, 0.4),
    "Short-term Dynamics": (0.8, 0.4),
    "Price Changes": (0.5, 0.1)
}

# Draw boxes
for text, (x,y) in components.items():
    ax.add_patch(Rectangle((x-0.15, y-0.05), 0.3, 0.1, fill=True, color='lightblue', alpha=0.7))
    plt.text(x, y, text, ha='center', va='center', fontsize=10)

# Draw arrows
arrows = [
    ((0.5, 0.75), (0.35, 0.45), "Adjustment"),
    ((0.5, 0.75), (0.65, 0.45), "Deviation"),
    ((0.2, 0.35), (0.5, 0.15), ""),
    ((0.8, 0.35), (0.5, 0.15), ""),
    ((0.5, 0.05), (0.5, -0.1), "Feedback"),
    ((0.5, -0.1), (0.1, -0.1), ""),
    ((0.1, -0.1), (0.1, 0.75), "")
]

for (x1,y1), (x2,y2), label in arrows:
    arrow = FancyArrowPatch((x1,y1), (x2,y2), 
                            arrowstyle='->', 
                            mutation_scale=20,
                            color='black')
    ax.add_patch(arrow)
    if label:
        plt.text((x1+x2)/2, (y1+y2)/2, label, 
                 ha='center', va='center', 
                 backgroundcolor='white',
                 fontsize=9)

# Draw equation
plt.text(0.5, -0.15, r"$\Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^4 \Gamma_i \Delta Y_{t-i} + \varepsilon_t$", 
         ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

# Formatting
plt.xlim(0, 1)
plt.ylim(-0.2, 1)
plt.axis('off')
plt.title("Vector Error Correction Model (VECM) Framework", fontsize=14)
plt.savefig('vecm_diagram.png', dpi=300, bbox_inches='tight')
plt.show()