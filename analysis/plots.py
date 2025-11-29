import matplotlib.pyplot as plt
import numpy as np
import os

agents = ['Random', 'Rule-Based', 'DQN']

# TODO: Replace hard-coded values with actual results from evaluation
# This should be generated automatically from eval.py results
# using the results we had from the eval
avg_scores = [0.4, 0.4, 4.6]
max_scores = [1.3, 1.4, 12.7]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#E8505B', '#F9C74F', '#4A90E2']

bars1 = axes[0].bar(agents, avg_scores, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
axes[0].set_ylabel('Average Territory (%)', fontsize=13, fontweight='bold')
axes[0].set_title('Average Performance', fontsize=15, fontweight='bold')
axes[0].grid(True, alpha=0.25, axis='y', linestyle='--')
axes[0].set_ylim(0, max(avg_scores) * 1.3)

for bar, score in zip(bars1, avg_scores):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{score:.1f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

bars2 = axes[1].bar(agents, max_scores, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
axes[1].set_ylabel('Peak Territory (%)', fontsize=13, fontweight='bold')
axes[1].set_title('Peak Performance', fontsize=15, fontweight='bold')
axes[1].grid(True, alpha=0.25, axis='y', linestyle='--')
axes[1].set_ylim(0, max(max_scores) * 1.2)

for bar, score in zip(bars2, max_scores):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.1f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('results/agent_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
