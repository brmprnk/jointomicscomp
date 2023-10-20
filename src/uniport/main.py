import uniport as up
import anndata as ada
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from uniport.model.vae import VAE
from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation
from src.util.early_stopping import EarlyStopping
from src.baseline.baseline import trainRegressor
from src.nets import OmicRegressor, OmicRegressorSCVI
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset, evaluatePerDatapoint
import src.util.logger as logger
import pickle
from src.baseline.baseline import classification, classificationMLP


def run(args: dict) -> None:
    # Check cuda availability
    print(args['name'])
    print(args['data1'] + '-' + args['data2'])

    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('UniPort'))
    os.makedirs(save_dir)

    n_modalities = args['nomics']
    assert n_modalities == 2

    # Load in data
    unscaledomics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]
    omics = []
    scalers = []
    for i, omic1 in enumerate(unscaledomics):
        m = np.min(omic1)
        if m < 0:
            ss = MinMaxScaler().fit(omic1)
        else:
            ss = MaxAbsScaler().fit(omic1)

        omics.append(ss.transform(omic1))
        scalers.append(ss)

    labels = np.load(args['labels'])
    labeltypes = np.load(args['labelnames'], allow_pickle=True)

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    val_ind = np.load(args['val_ind'])
    test_ind = np.load(args['test_ind'])

    omics_train = [omic[train_ind] for omic in omics]
    adtrain = []
    for i, omic in enumerate(omics_train):
        ds = ada.AnnData(csr_matrix(omic))
        ds.obs['source'] = args['data%d' % (i+1)]
        ds.obs['domain_id'] = 0
        ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
        adtrain.append(ds)

    Niter = int(np.round(args['epochs'] * omics_train[0].shape[0] / args['batch_size']))

    omics_val = [omic[val_ind] for omic in omics]
    omics_test = [omic[test_ind] for omic in omics]

    input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

    #likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

    # assert input_dim1 == omic1.shape[1]
    # assert input_dim2 == omic2.shape[1]

    # encoder_layers = [int(kk) for kk in ]
    # decoder_layers = encoder_layers[::-1][1:]
    layers = args['latent_dim'].split('-')
    enc_arch = []
    for i in range(len(layers)-1):
        enc_arch.append(['fc', int(layers[i]), 1, 'relu'])

    enc_arch.append(['fc', int(layers[-1]), '', ''])


    if 'pre_trained' in args and args['pre_trained'] != '':

        state = torch.load(args['pre_trained'] + 'config.pt')
        enc, dec, n_domain, ref_id, num_gene = state['enc'], state['dec'], state['n_domain'], state['ref_id'], state['num_gene']
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain, mode='v')
        model.load_model(args['pre_trained'] + 'model.pt')
        model.to(device)
        model.eval()

    else:
        logger.success("Initialized UniPort model.")

        net = up.Run(adatas=adtrain, mode='v', iteration=Niter, save_OT=True, out='latent', batch_size=args['batch_size'],
        lr=args['lr'], enc=enc_arch,
        gpu=0,
        loss_type='BCE',
        outdir=save_dir,
        seed=124,
        num_workers=1,
        patience=args['early_stopping_patience'],
        batch_key='domain_id',
        source_name='source',
        model_info=False,
        verbose=False)


    if args['task'] > 0:
        # imputation
        adval = []
        for i, omic in enumerate(omics_val):
            ds = ada.AnnData(csr_matrix(omic))
            ds.obs['source'] = args['data%d' % (i+1)]
            ds.obs['domain_id'] = 0
            ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
            adval.append(ds)

        adtest = []
        for i, omic in enumerate(omics_test):
            ds = ada.AnnData(csr_matrix(omic))
            ds.obs['source'] = args['data%d' % (i+1)]
            ds.obs['domain_id'] = 0
            ds.obs['domain_id'] = ds.obs['domain_id'].astype('category')
            adtest.append(ds)

        with torch.no_grad():
            ad = up.Run(adatas=adtrain, mode='v', out='project', batch_size=args['batch_size'], gpu=0, outdir=args['pre_trained']+'../', seed=124, batch_key='domain_id', source_name='source')
            ztrain = ad.obsm['project']

            ad = up.Run(adatas=adval, mode='v', out='project', batch_size=args['batch_size'], gpu=0, outdir=args['pre_trained']+'../', seed=124, batch_key='domain_id', source_name='source')
            zvalidation = ad.obsm['project']

            ad = up.Run(adatas=adtest, mode='v', out='project', batch_size=args['batch_size'], gpu=0, outdir=args['pre_trained']+'../', seed=124, batch_key='domain_id', source_name='source')
            ztest = ad.obsm['project']

            zrand = Independent(Normal(torch.zeros(model.z_dim), torch.ones(model.z_dim)), 1).sample([2000]).to(device)
            # zrand = zrand.double()

            Xsample = [model.decoder(zrand, i).detach().cpu().numpy() for i in range(args['nomics'])]
            XsampleScale = [torch.tensor(ss.inverse_transform(x)) for ss, x in zip(scalers, Xsample)]


            logger.info('Evaluating...')
            #

            if args['nomics'] == 2:
                logger.info('Generation coherence')
                acc = evaluate_generation(XsampleScale[0], XsampleScale[1], args['data1'], args['data2'])
                logger.info('Concordance: %.4f: ' % acc)
                logger.info('\n\n')

        likelihoods = [args['likelihood1'], args['likelihood2']]
        categories = []
        for i in range(n_modalities):
            if 'n_categories'+ str(i+1) in args:
                categories.append(args['n_categories'+ str(i+1)])
            else:
                categories.append(-1)

        print(categories)

        logger.info('Likelihood for omic1: %s' % likelihoods[0])
        logger.info('Likelihood for omic2: %s' % likelihoods[1])

        # Initialize GLMs
        if likelihoods[1] not in set(['nb', 'zinb', 'nbm']):
            net2from1 = OmicRegressor(ztrain.shape[1], unscaledomics[1].shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, n_categories=categories[1])
        else:
            net2from1 = OmicRegressorSCVI(ztrain.shape[1], unscaledomics[1].shape[1], distribution=likelihoods[1], optimizer_name='Adam', lr=0.0001, log_input=False)

        net2from1 = net2from1.to(device).double()


        # modality 2 from 1
        dataTrain = [torch.tensor(ztrain, device=device), torch.tensor(unscaledomics[1][train_ind], device=device)]
        dataValidation = [torch.tensor(zvalidation, device=device), torch.tensor(unscaledomics[1][val_ind], device=device)]
        dataTest = [torch.tensor(ztest, device=device), torch.tensor(unscaledomics[1][test_ind], device=device)]

        datasetTrain = MultiOmicsDataset(dataTrain)
        datasetValidation = MultiOmicsDataset(dataValidation)
        datasetTest = MultiOmicsDataset(dataTest)

        validationEvalBatchSize = 64
        trainEvalBatchSize = 64


        train_loader = DataLoader(datasetTrain, batch_size=64, shuffle=True, num_workers=0,
                                  drop_last=False)

        train_loader_eval = DataLoader(datasetTrain, batch_size=trainEvalBatchSize, shuffle=False, num_workers=0,
                                       drop_last=False)

        valid_loader = DataLoader(datasetValidation, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader = DataLoader(datasetTest, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader_individual = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=0,
                                          drop_last=False)


        early_stopping = EarlyStopping(patience=10, verbose=True)
        ckpt_dir = save_dir + '/checkpoint'
        logs_dir = save_dir + '/logs'

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)


        bestLoss, bestEpoch = trainRegressor(device=device, net=net2from1, num_epochs=500, train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=valid_loader,
              ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping)


        logger.info("Using model from epoch %d" % bestEpoch)

        checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')

        net2from1.load_state_dict(checkpoint['state_dict'])


        metricsValidation = evaluateUsingBatches(net2from1, device, valid_loader, True)
        metricsTest = evaluateUsingBatches(net2from1, device, test_loader, True)

        metricsTestIndividual2from1 = evaluatePerDatapoint(net2from1, device, test_loader_individual, True)


        logger.info('Validation performance, imputation error modality 2 from 1')
        for m in metricsValidation:
            logger.info('%s\t%.4f' % (m, metricsValidation[m]))

        logger.info('Test performance, imputation error modality 2 from 1')
        for m in metricsTest:
            logger.info('%s\t%.4f' % (m, metricsTest[m]))

        logger.info('Saving individual performances...')

        with open(save_dir + '/test_performance_per_datapoint.pkl', 'wb') as f:
            pickle.dump(metricsTestIndividual2from1, f)

        logger.info('Saving embeddings...')

        with open(save_dir + '/embeddings.pkl', 'wb') as f:
            embDict = {'ztrain': ztrain, 'zvalidation': zvalidation, 'ztest': ztest}
            pickle.dump(embDict, f)




    if args['task'] > 1:
        assert args['nomics'] == 2
        classLabels = np.load(args['labels'])
        labelNames = np.load(args['labelnames'], allow_pickle=True)

        ytrain = classLabels[train_ind]
        yvalid = classLabels[val_ind]
        ytest = classLabels[test_ind]


        logger.info('Test performance, classification task, linear classifier')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.]), 'mcc')
        performance1 = [acc, pr, rc, f1, mcc, confMat, CIs]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        if 'level' in args:
            level = args['level']
            assert level == 'l3'
        else:
            level = 'l2'

        # -----------------------------------------------------------------
        logger.info('Test performance, classification task, non-linear classifier')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, 'type-classifier/eval/' + level + '/uniport_' + args['data1'] + '/')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}


        logger.info("Saving results")
        with open(save_dir + "/uniport_task2_results.pkl", 'wb') as f:
            pickle.dump({'omic1': pr1, 'omic1-mlp': mlp_pr1}, f)
