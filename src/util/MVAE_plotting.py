import matplotlib.pyplot as plt


class LossPlotter(object):
    """
    Class that creates plots for either the training losses or validation losses.
    Losses are : total loss, reconstruction loss and KLD loss.
    Results are saved in the experiments' save dir.
    """

    def __init__(self, args, save_dir):
        print("Creating plots of the training and validation losses...")
        self.args = args
        self.x_axis = [*range(1, args.epochs + 1)]

        # For saving
        self.save_dir = save_dir

    def plot_training_losses(self, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter, total_batches):
        # Loss
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        plt.plot(self.x_axis, train_loss_meter.values[::total_batches], marker='.', color='tab:purple')
        for x, y in zip(self.x_axis, train_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax1.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training Loss")
        plt.savefig("{}/Loss.png".format(self.save_dir), dpi=400)

        # Reconstruction Loss
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        plt.plot(self.x_axis, train_recon_loss_meter.values[::total_batches], marker='.', color='tab:orange')
        for x, y in zip(self.x_axis, train_recon_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax2.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training Reconstruction Loss")
        plt.savefig("{}/Recon Loss.png".format(self.save_dir), dpi=400)

        # KLD Loss
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        plt.plot(self.x_axis, train_kld_loss_meter.values[::total_batches], marker='.', color='tab:red')
        for x, y in zip(self.x_axis, train_kld_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax3.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training KLD Loss")
        plt.savefig("{}/KLD Loss.png".format(self.save_dir), dpi=400)

    def plot_validation_loss(self, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter, total_val_batches):
        # Validation Loss
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        plt.plot(self.x_axis, val_loss_meter.values[::total_val_batches], marker='.', color='tab:purple')
        for x, y in zip(self.x_axis, val_loss_meter.values[::total_val_batches]):
            if x == 1 or x % 25 == 0:
                ax4.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation Loss")
        plt.savefig("{}/Validation Loss.png".format(self.save_dir), dpi=400)

        # Validation Reconstruction Loss
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        plt.plot(self.x_axis, val_recon_loss_meter.values[::total_val_batches], marker='.', color='tab:orange')
        for x, y in zip(self.x_axis, val_recon_loss_meter.values[::total_val_batches]):
            if x == 1 or x % 25 == 0:
                ax5.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation Reconstruction Loss")
        plt.savefig("{}/Validation Recon Loss.png".format(self.save_dir), dpi=400)

        # Validation KLD Loss
        fig6 = plt.figure(6)
        ax6 = fig6.add_subplot(111)
        plt.plot(self.x_axis, val_kld_loss_meter.values[::total_val_batches], marker='.', color='tab:red')
        for x, y in zip(self.x_axis, val_kld_loss_meter.values[::total_val_batches]):
            if x == 1 or x % 25 == 0:
                ax6.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(self.args.n_latents, self.args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation KLD Loss")
        plt.savefig("{}/Validation KLD Loss.png".format(self.save_dir), dpi=400)
