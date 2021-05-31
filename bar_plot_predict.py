# libraries
import numpy as np
import matplotlib.pyplot as plt

# Globals
BAR_WIDTH = 0.3
NUM_DECIMALS = 4

## Prediction loss with MMVAE's (CROSSMODAL)
# labels = ['RNA --> GCN', 'RNA --> DNA',
#           'GCN --> RNA', 'GCN --> DNA',
#           'DNA --> RNA', 'DNA --> GCN']
# x = np.arange(len(labels))

# # set heights of bars ( RNA_GCN - RNA_DNA - GCN_RNA - GCN_DNA - DNA_RNA - DNA_GCN)
# MOE_CROSS = [np.round(0.30725568532943726, NUM_DECIMALS), np.round(0.4897836148738861, NUM_DECIMALS),
#              np.round(0.3021275997161865, NUM_DECIMALS), np.round(0.48969361186027527, NUM_DECIMALS),
#              np.round(0.30212751030921936, NUM_DECIMALS), np.round(0.30704766511917114, NUM_DECIMALS)]

# POE_CROSS = [np.round(0.187183678150177, NUM_DECIMALS), np.round(0.10178180038928986, NUM_DECIMALS),
#              np.round(0.08658463507890701, NUM_DECIMALS), np.round(0.10178492963314056, NUM_DECIMALS),
#              np.round(0.08659809082746506, NUM_DECIMALS), np.round(0.13202133774757385, NUM_DECIMALS)]

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - BAR_WIDTH/2, MOE_CROSS, BAR_WIDTH, label='Mixture of Experts', color="#bc5090")
# rects2 = ax.bar(x + BAR_WIDTH/2, POE_CROSS, BAR_WIDTH, label='Product of Experts', color="#ffa600")

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Prediction Loss per Modality', fontweight='bold')
# ax.set_ylabel('Mean Squared Error', fontweight="bold")
# ax.set_title('Prediction Loss per Modality (Crossmodal)\n(latent_dim={}, batch_size={}, epochs={}, lr={})'
#              .format(128, 256, 100, 0.001))

# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=8)
# ax.legend()

# ax.bar_label(rects1, padding=3, fontsize=7)
# ax.bar_label(rects2, padding=3, fontsize=7)

# fig.tight_layout()

# save_dir = "/Users/bram/Desktop"
# plt.savefig("{}/Prediction Loss Crossmodal 3 Modalities 31 May.png".format(save_dir), dpi=600)
# plt.show()

## Prediction loss with MMVAE's (UNIMODAL)

labels = ['RNA --> RNA', 'GCN --> GCN', 'DNA --> DNA']
x = np.arange(len(labels))

# set heights of bars ( MoE - PoE)
MOE_UNIMODAL = [np.round(0.3020807206630707, NUM_DECIMALS), np.round(0.30718329548835754, 4), np.round(0.48947858810424805, 4)]
POE_UNIMODAL = [np.round(0.08659981191158295, NUM_DECIMALS), np.round(0.1871628761291504, 4), np.round(0.08422283083200455, 4)]

fig, ax = plt.subplots()
rects1 = ax.bar(x - BAR_WIDTH/2, MOE_UNIMODAL, BAR_WIDTH, label='Mixture of Experts', color="#bc5090")
rects2 = ax.bar(x + BAR_WIDTH/2, POE_UNIMODAL, BAR_WIDTH, label='Product of Experts', color="#ffa600")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Prediction Loss per Modality', fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontweight="bold")
ax.set_title('Prediction Loss per Modality (Unimodal)\n(latent_dim={}, batch_size={}, epochs={}, lr={})'
             .format(128, 256, 100, 0.001))

x_axis = np.arange(len(labels))
ax.set_xticks(x_axis)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
ax.bar_label(rects2, padding=3, fontsize=7)

fig.tight_layout()

save_dir = "/Users/bram/Desktop"
plt.savefig("{}/Prediction Loss Unimodal 3 Modalities 31 May.png".format(save_dir), dpi=600)
plt.show()
