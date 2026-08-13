import matplotlib.pyplot as plt

methods = ['ARF', 'LB', 'SRP', 'OB', 'StaticSPPFullSearch-Lite', 'StaticSPP-Lite',
           'DSPP-Lite', 'StaticSPPFullSearch-Adaptive', 'StaticSPP-Adaptive', 'DSPP-Adaptive']
accuracy = [0.916, 0.921, 0.909, 0.901, 0.895, 0.894, 0.895, 0.906, 0.905, 0.906]
pool_mem = [7748.5, 5111.9, 3149.6, 1120.1, 1456.4, 1456.3, 783.1, 996.5, 996.5, 970.6]
is_dspp = [False, False, False, False, False, False, True, False, False, True]

# manual offsets (dx, dy) in points, tuned to avoid overlap in the crowded cluster
offsets = {
    'ARF': (6, 4), 'LB': (6, 4), 'SRP': (6, 4), 'OB': (6, 4),
    'StaticSPPFullSearch-Lite': (6, 4), 'StaticSPP-Lite': (6, -12),
    'DSPP-Lite': (6, 4),
    'StaticSPPFullSearch-Adaptive': (8, 10), 'StaticSPP-Adaptive': (8, -14),
    'DSPP-Adaptive': (-95, 4),
}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for m, a, p, d in zip(methods, accuracy, pool_mem, is_dspp):
    marker = 'o' if d else ('s' if 'Static' in m else '^')
    color = 'crimson' if d else ('steelblue' if 'Static' in m else 'gray')
    ax.scatter(p, a, marker=marker, s=90, color=color, edgecolor='black', linewidth=0.5, zorder=3)
    ax.annotate(m, (p, a), textcoords="offset points", xytext=offsets[m], fontsize=7.5)

ax.set_xlabel('Pool memory (KB, macro-averaged across streams)')
ax.set_ylabel('Accuracy (macro-averaged across streams)')
ax.set_xscale('log')
ax.set_xlim(400, 12000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig('/home/claude/figs/accuracy_vs_memory.pdf')
plt.savefig('/home/claude/figs/accuracy_vs_memory.png', dpi=200)
print("saved fig2")
