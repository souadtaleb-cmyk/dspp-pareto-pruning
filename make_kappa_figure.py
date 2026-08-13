import matplotlib.pyplot as plt
import numpy as np

methods = ['ARF', 'DSPP-Adaptive', 'DSPP-Lite', 'LB', 'OB', 'SRP',
           'StaticSPP-Adaptive', 'StaticSPP-Lite', 'SFS-Adaptive', 'SFS-Lite']
creditcard = [0.600, 0.000, -0.000, -0.000, -0.000, 0.177, 0.000, -0.000, 0.000, -0.000]
http = [0.967, 0.959, 0.768, -0.000, 0.082, 0.974, 0.000, 0.026, 0.000, 0.027]

x = np.arange(len(methods))
width = 0.38

fig, ax = plt.subplots(figsize=(8, 4.2))
b1 = ax.bar(x - width/2, creditcard, width, label='CreditCard', color='#4472C4')
b2 = ax.bar(x + width/2, http, width, label='HTTP', color='#ED7D31')

ax.axhline(0, color='black', linewidth=0.6)
ax.set_ylabel("Cohen's Kappa")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=8)
ax.legend(frameon=False)
ax.set_ylim(-0.05, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/claude/figs/kappa_creditcard_http.pdf')
plt.savefig('/home/claude/figs/kappa_creditcard_http.png', dpi=200)
print("saved fig1")
