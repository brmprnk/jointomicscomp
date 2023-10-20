from src.totalVI.preprocessCITE import loadCITE, maskRNA, maskProteins
import torch
from src.util import logger
import os
from scvi.model import TOTALVI
import numpy as np
import pickle
from torch.distributions import Normal, Independent, Bernoulli
from src.baseline.baseline import classification, classificationMLP
from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation
import src.util.mydistributions as mydist

def run(args: dict) -> None:
    # Check cuda availability
    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('totalVI'))
    os.makedirs(save_dir)
    ckpt_dir = save_dir + '/checkpoint'


    n_modalities = args['nomics']
    assert n_modalities == 2

    # Load in data
    mudataTrain, mudataTest = loadCITE(save=False, dataPrefix=args['mudata_path'])

    params = dict(n_latent=args['zdim'], latent_distribution=args['enc_distribution'], n_hidden=args['n_neurons_per_layer'], n_layers_encoder=args['n_layers'], n_layers_decoder=args['n_layers'])

    net = TOTALVI(mudataTrain, **params)


    # net = net.double()

    if 'pre_trained' in args and args['pre_trained'] != '':

        if torch.cuda.is_available() and args['cuda']:
            net = TOTALVI.load(args['pre_trained'], adata=mudataTrain, accelerator='gpu', device='auto')
        else:
            net = TOTALVI.load(args['pre_trained'], adata=mudataTrain)
        logger.success("Loaded trained TOTALVI model.")

    else:

        logger.success("TOTALVI model.")

        # Create directories for checkpoint, sample and logs files

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logs_dir = save_dir + '/logs'

        # Data loadingx
        logger.info("Loading training and validation data into TOTALVI...")


        # Training and validation
        # shuffle split False and train size 0.87565 ensures that
        # cells from individual 2 are used on the validation set
        # (this is because loadCITE puts all cells of this donor at the end of the dataframes)
        trainSettings = {'max_epochs': args['epochs'], 'lr': args['lr'], 'accelerator': 'gpu', 'devices': 'auto', 'train_size': 0.87565, 'shuffle_set_split': False, 'batch_size': args['batch_size'], 'early_stopping': True, 'early_stopping_patience': 10, 'early_stopping_min_delta': 1., 'reduce_lr_on_plateau': False, 'n_steps_kl_warmup': 1, 'adversarial_classifier': False}
        net.train(**trainSettings)

        bestLoss = net.history['validation_loss'].min()[0]
        bestEpoch = np.where(net.history['validation_loss'] == bestLoss)[0][0]

        logger.info("Using model from epoch %d" % bestEpoch)


    if args['task'] == 0:
        lossDict = {'epoch': bestEpoch, 'val_loss': bestLoss}
        with open(save_dir + '/finalValidationLoss.pkl', 'wb') as f:
            pickle.dump(lossDict, f)

        net.save(ckpt_dir, overwrite=True)

    # Imputation
    if args['task'] > 0:

        mudataTrainRnaOnly = maskProteins(mudataTrain)
        mudataTrainProteinOnly = maskRNA(mudataTrain)

        mudataTestRnaOnly = maskProteins(mudataTest)
        mudataTestProteinOnly = maskRNA(mudataTest)


        Ntrain = np.where(mudataTrain.obs['rna:donor'] != 'P2')[0][-1] + 1

        net.module.eval()

        with torch.no_grad():
            ztrainval = net.get_latent_representation(mudataTrain, give_mean=True, return_dist=False)
            ztrain = ztrainval[:Ntrain]
            zvalidation = ztrainval[Ntrain:]
            ztest = net.get_latent_representation(mudataTest, give_mean=True, return_dist=False)
            del ztrainval

            z1trainval = net.get_latent_representation(mudataTrainRnaOnly, give_mean=True, return_dist=False)
            z1train = z1trainval[:Ntrain]
            z1validation = z1trainval[Ntrain:]
            z1test = net.get_latent_representation(mudataTestRnaOnly, give_mean=True, return_dist=False)
            del z1trainval

            z2trainval = net.get_latent_representation(mudataTrainProteinOnly, give_mean=True, return_dist=False)
            z2train = z2trainval[:Ntrain]
            z2validation = z2trainval[Ntrain:]
            z2test = net.get_latent_representation(mudataTestProteinOnly, give_mean=True, return_dist=False)
            del z2trainval

            #
            # # draw random samples from the prior and reconstruct them
            assert args['enc_distribution'] == 'normal'
            zrand = Independent(Normal(torch.zeros(net.module.n_latent), torch.ones(net.module.n_latent)), 1).sample([2000]).to(device)
            #zrand = zrand.double()
            # need to sample library size too
            # use empirical mean and variance of training set embeddings
            # we draw samples from distribution
            lls = np.array(np.sum(mudataTrain['rna_subset'].X,1)).reshape(-1,)
            m = np.median(lls)
            mad = np.median(np.abs(lls - m))

            libsizesrand = torch.tensor(np.random.randint(m-mad,m+mad,2000)).float().to(device).reshape(-1,1)

            pdict = net.module.generative(zrand, libsizesrand, torch.zeros(2000), torch.zeros(2000))

            pred = mydist.NegativeBinomialMixture(pdict['py_']['rate_back'], pdict['py_']['rate_fore'], pdict['py_']['r'], pdict['py_']['mixing'])
            selectedMixture1 = Bernoulli(pred.mixture_probs).sample()
            adtrand = pred.mu1 * selectedMixture1 + pred.mu2 * (1-selectedMixture1)
            adtrand = adtrand.cpu().detach()
            rnarand = pdict['px_']['rate'].cpu().detach()


            logger.info('Evaluating...')

            if args['nomics'] == 2:
                logger.info('Generation coherence')
                acc = evaluate_generation(rnarand, adtrand, args['data1'], args['data2'])
                logger.info('Concordance: %.4f: ' % acc)
                logger.info('\n\n')


            logger.info('Reconstruction metrics')

            logger.info('Validation set:')
            pdict = net.module.generative(torch.tensor(z1validation), torch.tensor(lls[Ntrain:]).reshape(-1,1), torch.zeros(z1validation.shape[0]), torch.zeros(z1validation.shape[0]))
            pdictProt = pdict['py_']
            pred = mydist.NegativeBinomialMixture(pdictProt['rate_back'], pdictProt['rate_fore'], pdictProt['r'], pdictProt['mixing'])

            ll2from1 = pred.log_prob(torch.tensor(mudataTrain['protein'].X[Ntrain:]).to(device))
            logger.info('Log likelihood ADT from RNA %.4f' % torch.mean(torch.sum(ll2from1, 1)))

            logger.info('\nTest set:')
            llstest = net.get_latent_library_size(mudataTest, give_mean=False)

            pdict = net.module.generative(torch.tensor(z1test).to(device), torch.tensor(llstest).to(device), torch.zeros(z1test.shape[0]), torch.zeros(z1test.shape[0]))
            pdictProt = pdict['py_']
            pred = mydist.NegativeBinomialMixture(pdictProt['rate_back'], pdictProt['rate_fore'], pdictProt['r'], pdictProt['mixing'])

            ll2from1 = pred.log_prob(torch.tensor(mudataTest['protein'].X).to(device))

            logger.info('Log likelihood ADT from RNA %.4f' % torch.mean(torch.sum(ll2from1, 1)))

            metricsTestIndividual = {'LL2/1': torch.sum(ll2from1, 1)}


        logger.info('Saving individual performances...')

        with open(save_dir + '/test_performance_per_datapoint.pkl', 'wb') as f:
            pickle.dump(metricsTestIndividual, f)

        logger.info('Saving embeddings...')

        with open(save_dir + '/embeddings.pkl', 'wb') as f:
            embDict = {'ztrain': ztrain, 'zvalidation': zvalidation, 'ztest': ztest}
            pickle.dump(embDict, f)

    # classification
    if args['task'] > 1:
        assert args['nomics'] == 2
        classLabels = np.load(args['labels'])
        labelNames = np.load(args['labelnames'], allow_pickle=True)

        train_ind = np.load(args['train_ind'])
        val_ind = np.load(args['val_ind'])
        test_ind = np.load(args['test_ind'])

        ytrain = classLabels[train_ind]
        yvalid = classLabels[val_ind]
        ytest = classLabels[test_ind]

        logger.info('Test performance, classification task, linear classifier, modality 1')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(z1train, ytrain, z1validation, yvalid, z1test, ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        logger.info('Test performance, classification task, linear classifier, modality 2')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(z2train, ytrain, z2validation, yvalid, z2test, ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        performance2 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

        pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

        logger.info('Test performance, classification task, linear classifier, both modalities')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), args['clf_criterion'])
        performance12 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

        pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}

        if 'level' in args:
            level = args['level']
            assert level == 'l3'
        else:
            level = 'l2'

        # -----------------------------------------------------------------
        logger.info('Test performance, classification task, non-linear classifier, modality 1')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(z1train, ytrain, z1validation, yvalid, z1test, ytest, 'type-classifier/eval/' + level + '/totalvi_' + args['data1'] + '/')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        logger.info('Test performance, classification task, non-linear classifier, modality 2')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(z2train, ytrain, z2validation, yvalid, z2test, ytest, 'type-classifier/eval/' + level + '/totalvi_' + args['data2'] + '/')
        performance2 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

        mlp_pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

        logger.info('Test performance, classification task, non-linear classifier, both modalities')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, 'type-classifier/eval/' + level + '/totalvi_' + args['data1'] + '_' + args['data2'] + '/')
        performance12 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

        mlp_pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}


        logger.info("Saving results")
        with open(save_dir + "/totalvi_task2_results.pkl", 'wb') as f:
            pickle.dump({'omic1': pr1, 'omic2': pr2, 'omic1+2': pr12, 'omic1-mlp': mlp_pr1, 'omic2-mlp': mlp_pr2, 'omic1+2-mlp': mlp_pr12}, f)
