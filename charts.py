import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data naturally (Pandas will read your headers automatically)
df = pd.read_csv("compression_results.csv")

# 2. Sort the ratios logically 
ratio_order = ['4:4:4', '4:2:2', '4:4:0', '4:2:0', '4:1:0']
df['Ratio'] = pd.Categorical(df['Ratio'], categories=ratio_order, ordered=True)
df = df.sort_values(['Image_Name', 'Ratio'])

# ==========================================
# GRAPH 1: Grouped Bar Chart (Compression Ratio)
# ==========================================
plt.figure(figsize=(10, 6))
# Notice we use 'Image_Name' for the hue here to match your CSV
sns.barplot(data=df, x='Ratio', y='Compression_Ratio', hue='Image_Name')

plt.title('Algorithm Efficiency: Compression Ratio vs. Chroma Subsampling', fontsize=14)
plt.xlabel('Chroma Subsampling Ratio', fontsize=12)
plt.ylabel('Compression Ratio', fontsize=12)
plt.legend(title='Original Image')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig('graph_compression_ratio.png', dpi=300)
plt.show()