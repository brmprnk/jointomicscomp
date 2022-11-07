import os
import numpy as np
from torch.utils.data import DataLoader
from src.nets import *
from src.CGAE.model import train, impute, extract, MultiOmicsDataset, evaluateUsingBatches, evaluatePerDatapoint
from src.util import logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
from src.baseline.baseline import classification, classificationMLP
import pickle

def run(args: dict) -> None:
    # Check cuda availability
    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('CVAE'))
    os.makedirs(save_dir)

    n_modalities = args['nomics']

    # Load in data
    omics = [np.load(args['data_path%d' % (i+1)]) for i in range(n_modalities)]

    labels = np.load(args['labels'])
    labeltypes = np.load(args['labelnames'], allow_pickle=True)

    # Use predefined split
    train_ind = np.load(args['train_ind'])
    val_ind = np.load(args['val_ind'])
    test_ind = np.load(args['test_ind'])

    omics_train = [omic[train_ind] for omic in omics]
    omics_val = [omic[val_ind] for omic in omics]
    omics_test = [omic[test_ind] for omic in omics]

    ytrain = labels[train_ind]
    yvalid = labels[val_ind]
    ytest = labels[test_ind]

    # Number of features
    input_dims = [args['num_features%d' % (i+1)] for i in range(n_modalities)]

    likelihoods = [args['likelihood%d' % (i+1)] for i in range(n_modalities)]

    # assert input_dim1 == omic1.shape[1]
    # assert input_dim2 == omic2.shape[1]

    encoder_layers = [int(kk) for kk in args['latent_dim'].split('-')]
    decoder_layers = encoder_layers[::-1][1:]

    # Initialize network model
    if 'categorical' not in likelihoods:

        net = ConcatenatedVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                           args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                           args['enc_distribution'], args['beta_start_value'])

    else:
        categories = [args['n_categories%d' % (i + 1)] for i in range(n_modalities)]
        net = ConcatenatedVariationalAutoencoder(input_dims, encoder_layers, decoder_layers, likelihoods,
                           args['use_batch_norm'], args['dropout_probability'], args['optimizer'], args['lr'],  args['lr'],
                           args['enc_distribution'], args['beta_start_value'], n_categories=categories)



    net = net.double()

    if 'pre_trained' in args and args['pre_trained'] != '':
        checkpoint = torch.load(args['pre_trained'])

        net.encoders[0].load_state_dict(checkpoint['state_dict_enc'][0])
        for i in range(n_modalities):

            net.decoders[i].load_state_dict(checkpoint['state_dict_dec'][i])

        #net.load_state_dict(checkpoint['state_dict'])
        logger.success("Loaded trained ConcatenatedVariationalAutoencoder model.")

    else:

        logger.success("Initialized ConcatenatedVariationalAutoencoder model.")
        logger.info(str(net))
        logger.info("Number of model parameters: ")
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info("{}".format(num_params))

        # Create directories for checkpoint, sample and logs files
        ckpt_dir = save_dir + '/checkpoint'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logs_dir = save_dir + '/logs'

        # Data loading
        logger.info("Loading training and validation data into ConcatenatedVariationalAutoencoder...")

        dataTrain = [torch.tensor(omic, device=device) for omic in omics_train]
        dataValidation = [torch.tensor(omic, device=device) for omic in omics_val]
        dataTest = [torch.tensor(omic, device=device) for omic in omics_test]

        datasetTrain = MultiOmicsDataset(dataTrain)
        datasetValidation = MultiOmicsDataset(dataValidation)

        try:
            validationEvalBatchSize = args['train_loader_eval_batch_size']
            trainEvalBatchSize = args['train_loader_eval_batch_size']
        except KeyError:
            validationEvalBatchSize = dataValidation[0].shape[0]
            trainEvalBatchSize = dataTrain[0].shape[0]


        train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                                  drop_last=False)

        train_loader_eval = DataLoader(datasetTrain, batch_size=trainEvalBatchSize, shuffle=False, num_workers=0,
                                       drop_last=False)

        valid_loader = DataLoader(datasetValidation, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                                  drop_last=False)

        # Setup early stopping, terminates training when validation loss does not improve for early_stopping_patience epochs
        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

        # Training and validation

        bestLoss, bestEpoch = train(device=device, net=net, num_epochs=args['epochs'], train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=valid_loader,
              ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping, save_step=args['log_save_interval'], multimodal=True)


        logger.info("Using model from epoch %d" % bestEpoch)


    if args['task'] == 0:
        lossDict = {'epoch': bestEpoch, 'val_loss': bestLoss}
        with open(save_dir + '/finalValidationLoss.pkl', 'wb') as f:
            pickle.dump(lossDict, f)



    # Imputation
    if args['task'] > 0:

        dataTrain = [torch.tensor(omic, device=device) for omic in omics_train]
        dataValidation = [torch.tensor(omic, device=device) for omic in omics_val]
        dataTest = [torch.tensor(omic, device=device) for omic in omics_test]

        datasetTrain = MultiOmicsDataset(dataTrain)
        datasetValidation = MultiOmicsDataset(dataValidation)
        datasetTest = MultiOmicsDataset(dataTest)

        try:
            validationEvalBatchSize = args['train_loader_eval_batch_size']
            trainEvalBatchSize = args['train_loader_eval_batch_size']
        except KeyError:
            validationEvalBatchSize = dataValidation[0].shape[0]
            trainEvalBatchSize = dataTrain[0].shape[0]


        train_loader = DataLoader(datasetTrain, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                                  drop_last=False)

        train_loader_eval = DataLoader(datasetTrain, batch_size=trainEvalBatchSize, shuffle=False, num_workers=0,
                                       drop_last=False)

        valid_loader = DataLoader(datasetValidation, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader = DataLoader(datasetTest, batch_size=validationEvalBatchSize, shuffle=False, num_workers=0,
                                  drop_last=False)

        test_loader_individual = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=0,
                                  drop_last=False)


        ztrain = np.zeros((dataTrain[i].shape[0], net.z_dim))
        zvalidation = np.zeros((dataValidation[i].shape[0], net.z_dim))
        ztest = np.zeros((dataTest[i].shape[0], net.z_dim))

        z1train = np.zeros((dataTrain[i].shape[0], net.z_dim))
        z1validation = np.zeros((dataValidation[i].shape[0], net.z_dim))
        z1test = np.zeros((dataTest[i].shape[0], net.z_dim))

        z2train = np.zeros((dataTrain[i].shape[0], net.z_dim))
        z2validation = np.zeros((dataValidation[i].shape[0], net.z_dim))
        z2test = np.zeros((dataTest[i].shape[0], net.z_dim))


        net.eval()

        ind = 0
        b = args['batch_size']
        for data in train_loader_eval:
            batch = (data[0][0].double(), data[1][0].double())

            z_tmp, _ = net.embedAndReconstruct(batch)

            ztrain[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch1 = (data[0][0].double(), torch.zeros(data[1][0].shape).double().to(device))

            z_tmp, _ = net.embedAndReconstruct(batch1)

            z1train[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch2 = (torch.zeros(data[0][0].shape).double().to(device), data[1][0].double())

            z_tmp, _ = net.embedAndReconstruct(batch2)

            z2train[ind:ind+b] = z_tmp.cpu().detach().numpy()

            ind += b


        ind = 0
        for data in valid_loader:
            batch = (data[0][0].double(), data[1][0].double())
            z_tmp, _ = net.embedAndReconstruct(batch)

            zvalidation[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch1 = (data[0][0].double(), torch.zeros(data[1][0].shape).double().to(device))

            z_tmp, _ = net.embedAndReconstruct(batch1)

            z1validation[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch2 = (torch.zeros(data[0][0].shape).double().to(device), data[1][0].double())

            z_tmp, _ = net.embedAndReconstruct(batch2)

            z2validation[ind:ind+b] = z_tmp.cpu().detach().numpy()

            ind += b


        ind = 0
        for data in test_loader:
            batch = (data[0][0].double(), data[1][0].double())
            z_tmp, _ = net.embedAndReconstruct(batch)

            ztest[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch1 = (data[0][0].double(), torch.zeros(data[1][0].shape).double().to(device))

            z_tmp, _ = net.embedAndReconstruct(batch1)

            z1test[ind:ind+b] = z_tmp.cpu().detach().numpy()

            batch2 = (torch.zeros(data[0][0].shape).double().to(device), data[1][0].double())

            z_tmp, _ = net.embedAndReconstruct(batch2)

            z2test[ind:ind+b] = z_tmp.cpu().detach().numpy()


            ind += b
        #
        # # draw random samples from the prior and reconstruct them
        assert args['enc_distribution'] == 'normal'
        zrand = Independent(Normal(torch.zeros(net.z_dim), torch.ones(net.z_dim)), 1).sample([2000]).to(device)
        zrand = zrand.double()
        #
        #
        Xsample = [dec(zrand).mean.cpu().detach() for dec in net.decoders]

        from src.util.evaluate import evaluate_imputation, evaluate_classification, evaluate_generation
        logger.info('Evaluating...')

        if args['nomics'] == 2:
            logger.info('Generation coherence')
            acc = evaluate_generation(Xsample[0], Xsample[1], args['data1'], args['data2'])
            logger.info('Concordance: %.4f: ' % acc)
            logger.info('\n\n')

        logger.info('Reconstruction metrics')

        logger.info('Validation set:')
        metricsValidation = evaluateUsingBatches(net, device, valid_loader, True)
        for m in metricsValidation:
            logger.info('%s\t%.4f' % (m, metricsValidation[m]))

        logger.info('\nTest set:')
        metricsTest = evaluateUsingBatches(net, device, test_loader, True)
        for m in metricsTest:
            logger.info('%s\t%.4f' % (m, metricsTest[m]))

        metricsTestIndividual = evaluatePerDatapoint(net, device, test_loader_individual, True)
        # for m in metricsTest:
        #     logger.info('%s\t%.4f' % (m, torch.mean(metricsTestIndividual[m])))

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
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(z1train, ytrain, z1validation, yvalid, z1test, ytest, 'type-classifier/eval/' + level + '/cvae_' + args['data1'] + '/')
        performance1 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance1[0], np.mean(performance1[1]), np.mean(performance1[2]), np.mean(performance1[3]), performance1[4]))


        mlp_pr1 = {'acc': performance1[0], 'pr': performance1[1], 'rc': performance1[2], 'f1': performance1[3], 'mcc': performance1[4], 'confmat': performance1[5], 'CIs': CIs}

        logger.info('Test performance, classification task, non-linear classifier, modality 2')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(z2train, ytrain, z2validation, yvalid, z2test, ytest, 'type-classifier/eval/' + level + '/cvae_' + args['data2'] + '/')
        performance2 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance2[0], np.mean(performance2[1]), np.mean(performance2[2]), np.mean(performance2[3]), performance2[4]))

        mlp_pr2 = {'acc': performance2[0], 'pr': performance2[1], 'rc': performance2[2], 'f1': performance2[3], 'mcc': performance2[4], 'confmat': performance2[5], 'CIs': CIs}

        logger.info('Test performance, classification task, non-linear classifier, both modalities')
        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(ztrain, ytrain, zvalidation, yvalid, ztest, ytest, 'type-classifier/eval/' + level + '/cvae_' + args['data1'] + '_' + args['data2'] + '/')
        performance12 = [acc, pr, rc, f1, mcc, confMat]
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (performance12[0], np.mean(performance12[1]), np.mean(performance12[2]), np.mean(performance12[3]), performance12[4]))

        mlp_pr12 = {'acc': performance12[0], 'pr': performance12[1], 'rc': performance12[2], 'f1': performance12[3], 'mcc': performance12[4], 'confmat': performance12[5], 'CIs': CIs}


        logger.info("Saving results")
        with open(save_dir + "/CVAE_task2_results.pkl", 'wb') as f:
            pickle.dump({'omic1': pr1, 'omic2': pr2, 'omic1+2': pr12, 'omic1-mlp': mlp_pr1, 'omic2-mlp': mlp_pr2, 'omic1+2-mlp': mlp_pr12}, f)
