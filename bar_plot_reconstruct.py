## Reconstruction loss with MOFA

# libraries
import numpy as np
import matplotlib.pyplot as plt

# Globals
BAR_WIDTH = 0.3
NUM_DECIMALS = 4

labels = ['RNA-seq', 'Gene CN', 'DNA Methylation', 'Average']
x = np.arange(len(labels) - 1)

moe_rna_rna = np.load("/Users/bram/Desktop/moe_three_modalities_first_results/Recon array rna_rna.npy")
moe_gcn_gcn = np.load("/Users/bram/Desktop/moe_three_modalities_first_results/Recon array gcn_gcn.npy")
moe_dna_dna = np.load("/Users/bram/Desktop/moe_three_modalities_first_results/Recon array dna_dna.npy")

poe_rna_rna = np.load("/Users/bram/Desktop/poe_three_modalities_first_results/Recon array rna_rna.npy")
poe_gcn_gcn = np.load("/Users/bram/Desktop/poe_three_modalities_first_results/Recon array gcn_gcn.npy")
poe_dna_dna = np.load("/Users/bram/Desktop/poe_three_modalities_first_results/Recon array dna_dna.npy")


# set heights of bars ( RNA - GCN - DNA)
mofa = [np.round(0.07250928867899345, NUM_DECIMALS), np.round(0.028519938851535167, NUM_DECIMALS), np.round(0.4035390146525409, NUM_DECIMALS)]
moe = [np.round(moe_rna_rna[-1], NUM_DECIMALS), np.round(moe_gcn_gcn[-1], NUM_DECIMALS), np.round(moe_dna_dna[-1], NUM_DECIMALS)]
poe = [np.round(poe_rna_rna[-1], NUM_DECIMALS), np.round(poe_gcn_gcn[-1], NUM_DECIMALS), np.round(poe_dna_dna[-1], NUM_DECIMALS)]

average_mofa = np.round(sum(mofa) / len(mofa), NUM_DECIMALS)
average_moe = np.round(sum(moe) / len(moe), NUM_DECIMALS)
average_poe = np.round(sum(poe) / len(poe), NUM_DECIMALS)

fig, ax = plt.subplots()
rects1 = ax.bar(x - BAR_WIDTH, mofa, BAR_WIDTH, label='MOFA+', color="#003f5c")
rects2 = ax.bar(x, moe, BAR_WIDTH, label='MMVAE: Mixture of Experts', color="#bc5090")
rects3 = ax.bar(x + BAR_WIDTH, poe, BAR_WIDTH, label='MMVAE: Product of Experts', color="#ffa600")

rects4 = ax.bar(3 - BAR_WIDTH, average_mofa, BAR_WIDTH, color="#003f5c", hatch="//")
rects5 = ax.bar(3, average_moe, BAR_WIDTH, color="#bc5090", hatch="//")
rects6 = ax.bar(3 + BAR_WIDTH, average_poe, BAR_WIDTH, color="#ffa600", hatch="//")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Reconstruction Loss per Modality', fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontweight="bold")
ax.set_title('Reconstruction Loss per Modality\n' +
             'MOFA+: (Factors={}, views={}, groups={})\n'.format(10, 3, 1) +
             'MMVAE: (latent_dim={}, batch_size={}, epochs={}, lr={})'.format(128, 256, 100, 0.001))

x_axis = np.arange(len(labels))
ax.set_xticks(x_axis)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
ax.bar_label(rects2, padding=3, fontsize=7)
ax.bar_label(rects3, padding=3, fontsize=7)
ax.bar_label(rects4, padding=3, fontsize=7)
ax.bar_label(rects5, padding=3, fontsize=7)
ax.bar_label(rects6, padding=3, fontsize=7)

fig.tight_layout()

save_dir = "/Users/bram/Desktop"
plt.savefig("{}/Validation Recon Loss 3 Modalities 31 May.png".format(save_dir), dpi=400)
plt.show()
