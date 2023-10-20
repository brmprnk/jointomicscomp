import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.decomposition import PCA
from src.util.evaluate import *
import src.util.logger as logger
from src.util.early_stopping import EarlyStopping
from src.util.umapplotter import UMAPPlotter
import matplotlib.pyplot as plt
import seaborn as sns
from src.nets import OmicRegressor, OmicRegressorSCVI, MLP
from tensorboardX import SummaryWriter
from src.CGAE.model import evaluateUsingBatches, MultiOmicsDataset, evaluatePerDatapoint
from torch.utils.data import DataLoader
from src.util.trainTypeClassifier import CustomDataset

def trainRegressor(device, net, num_epochs, train_loader, train_loader_eval, valid_loader, ckpt_dir, logs_dir, early_stopping):
    # Define logger
    tf_logger = SummaryWriter(logs_dir)

    # Evaluate validation set before start training
    print("[*] Evaluating epoch 0...")
    metrics = evaluateUsingBatches(net, device, train_loader_eval, True)

    assert 'loss' in metrics
    print("--- Training loss:\t%.4f" % metrics['loss'])

    metrics = evaluateUsingBatches(net, device, valid_loader, True)
    bestValLoss = metrics['loss']
    bestValEpoch  = 0

    print("--- Validation loss:\t%.4f" % metrics['loss'])
    print(num_epochs)

    # Start training phase
    print("[*] Start training...")
    # Training epochs
    for epoch in range(num_epochs):
        net.train()

        print("[*] Epoch %d..." % (epoch + 1))
        # for param_group in optimizer.param_groups:
        #	print('--- Current learning rate: ', param_group['lr'])

        for data in train_loader:
            # Get current batch and transfer to device
            # data = data.to(device)

            with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
                # Set the parameter gradients to zero
                net.opt.zero_grad()

                current_loss = net.compute_loss([data[0][0].to(device).double(), data[1][0].to(device).double()])

                # Backward pass and optimize
                current_loss.backward()
                net.opt.step()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict()}
        torch.save(state, ckpt_dir + '/model_last.pth.tar')

        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))

        metricsTrain = evaluateUsingBatches(net, device, train_loader_eval, True)
        print("--- Training loss:\t%.4f" % metricsTrain['loss'])


        metricsValidation = evaluateUsingBatches(net, device, valid_loader, True)
        print("--- Validation loss:\t%.4f" % metricsValidation['loss'])

        if metricsValidation['loss'] < bestValLoss:
            bestValLoss = metricsValidation['loss']
            bestValEpoch = epoch + 1
            torch.save(state, ckpt_dir + '/model_best.pth.tar')


        early_stopping(metricsValidation['loss'])

        for m in metricsValidation:
            tf_logger.add_scalar(m + '/train', metricsTrain[m], epoch + 1)
            tf_logger.add_scalar(m + '/validation', metricsValidation[m], epoch + 1)


        # Stop training when not improving
        if early_stopping.early_stop:
            logger.info('Early stopping training since loss did not improve for {} epochs.'
                        .format(early_stopping.patience))
            break

    print("[*] Finish training.")
    return bestValLoss, bestValEpoch



def impute(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, num_features, criterion='mse'):
    # returns imputed values as well as evaluation of the imputation based on mse and rsquared

    criteria = ['mse', 'spearman', 'r2']
    index = criteria.index(criterion)

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = Ridge(alpha=a, fit_intercept=True, random_state=1)
        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_imputation(y_valid, model.predict(x_valid), num_features, criterion)[index]

    if criterion == 'mse':
        bestModel = models[np.argmin(validationPerformance)]
    else:
        bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test)

    mse, corr, rsq = evaluate_imputation(y_test, predictions, num_features, 'mse')


    return predictions, mse, corr, rsq



def classification(x_train, y_train, x_valid, y_valid, x_test, y_test, alphas, criterion='mcc', baseline=False):
    # returns the predicted class labels, as well as accuracy measures

    if criterion == 'acc':
        ind = 0
    elif criterion == 'pr':
        ind = 1
    elif criterion == 'rc':
        ind = 2
    elif criterion == 'f1':
        ind = 3
    else:
        assert criterion == 'mcc'
        ind = 4

    validationPerformance = np.zeros(alphas.shape[0])
    models = []
    for i, a in enumerate(alphas):
        model = SGDClassifier(penalty='l2', loss='hinge', alpha=a, fit_intercept=True, random_state=1)

        # train
        model.fit(x_train, y_train)

        # save so that we don't have to re-train
        models.append(model)

        # evaluate using user-specified criterion
        validationPerformance[i] = evaluate_classification(y_valid, model.predict(x_valid))[ind]

    bestModel = models[np.argmax(validationPerformance)]

    predictions = bestModel.predict(x_test).astype(int)
    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(y_test, predictions)
    return predictions, acc, pr, rc, f1, mcc, confMat, CIs




def classificationMLP(x_train, y_train, x_valid, y_valid, x_test, y_test, savedir='./'):
    # returns the predicted class labels, as well as accuracy measures


    trainDataset = CustomDataset(x_train, y_train)
    validationDataset = CustomDataset(x_valid, y_valid)
    testDataset = CustomDataset(x_test, y_test)


    train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    valid_loader = DataLoader(validationDataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    test_loader = DataLoader(testDataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    logger.info('defining model with %d classes' % np.unique(y_train).shape[0])
    model = MLP(x_train.shape[1], 64, np.unique(y_train).shape[0])
    model = model.double()

    device = torch.device('cuda:0')
    model = model.to(device)

    # logger.success('Model ok')

    early_stopping = EarlyStopping(patience=10, verbose=True)

    checkpoint_dir = savedir + 'checkpoint/'
    log_dir = savedir + 'logs/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    model.optimize(150, 0.0001, train_loader, valid_loader, checkpoint_dir, log_dir, early_stopping)

    checkpoint = torch.load(savedir + 'checkpoint/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    ypred_test = np.zeros((y_test.shape[0],), int)

    model.eval()
    with torch.no_grad():
        testLoss = 0.

        loss_fun = torch.nn.CrossEntropyLoss()

        i = 0
        b = 64
        for x, y in test_loader:
            y_pred = model.forward(x[0].double().to(device))

            ypred_test[i:i+b] = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
            testLoss += loss_fun(y_pred, y[0].to(device)).item() * x[0].shape[0]

            i += b

        testLoss /= len(train_loader.dataset)


    acc, pr, rc, f1, mcc, confMat, CIs = evaluate_classification(y_test, ypred_test)
    return ypred_test, acc, pr, rc, f1, mcc, confMat, CIs



def run_baseline(args: dict) -> None:
    """

    @param args: Dictionary containing input parameters
    @return:
    """
    logger.info("Running Baseline - Task {}".format(args['task']))
    save_dir = os.path.join(args['save_dir'], '{}'.format('baseline'))
    os.makedirs(save_dir)

    device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
    logger.info("Selected device: {}".format(device))
    torch.manual_seed(args['random_seed'])

    ckpt_dir = save_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logs_dir = save_dir + '/logs'



    n_modalities = args['nomics']
    assert n_modalities == 2

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


    # Initialize network model
    if 'categorical' not in likelihoods:
        if likelihoods[1] not in {'nb', 'zinb', 'nbm'}:
            net2from1 = OmicRegressor(input_dims[0], input_dims[1], distribution=likelihoods[1], optimizer_name=args['optimizer'], lr=args['lr'])
        else:
            if args['data1'] == 'ADT' and args['use_batch_norm']:
                ubn = True
            else:
                ubn = False
            net2from1 = OmicRegressorSCVI(input_dims[0], input_dims[1], distribution=likelihoods[1], optimizer_name=args['optimizer'], lr=args['lr'], use_batch_norm=ubn, log_input=args['log_transform1'])

        net2from1 = net2from1.to(device).double()

        if likelihoods[0] not in {'nb', 'zinb', 'nbm'}:
            net1from2 = OmicRegressor(input_dims[1], input_dims[0], distribution=likelihoods[0], optimizer_name=args['optimizer'], lr=args['lr'])
        else:
            if args['data2'] == 'ADT' and args['use_batch_norm']:
                ubn = True
            else:
                ubn = False

            net1from2 = OmicRegressorSCVI(input_dims[1], input_dims[0], distribution=likelihoods[0], optimizer_name=args['optimizer'], lr=args['lr'], use_batch_norm=ubn, log_input=args['log_transform2'])

        net1from2 = net1from2.to(device).double()


    else:
        n_categories = [args['n_categories%d' % (i+1)] for i in range(n_modalities)]

        net2from1 = OmicRegressor(input_dims[0], input_dims[1], distribution=likelihoods[1], optimizer_name=args['optimizer'], lr=args['lr'], n_categories=n_categories[1])
        net2from1 = net2from1.to(device).double()

        net1from2 = OmicRegressor(input_dims[1], input_dims[0], distribution=likelihoods[0], optimizer_name=args['optimizer'], lr=args['lr'], n_categories=n_categories[0])
        net1from2 = net1from2.to(device).double()


    logger.info("Succesfully loaded in all data")
    if args['task'] > 0:
        logger.success("Running baseline for task {} with data from 1: {} and 2: {}".format(args['task'], args['data1'],
                                                                                            args['data2']))

        # modality 2 from 1
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


        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

        bestLoss, bestEpoch = trainRegressor(device=device, net=net2from1, num_epochs=args['epochs'], train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=valid_loader,
              ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping)


        logger.info("Using model from epoch %d" % bestEpoch)

        checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')

        net2from1.load_state_dict(checkpoint['state_dict'])
        net2from1.eval()

        metricsValidation = evaluateUsingBatches(net2from1, device, valid_loader, True)
        metricsTest = evaluateUsingBatches(net2from1, device, test_loader, True)

        logger.info('Validation performance, imputation error modality 2 from 1')
        for m in metricsValidation:
            logger.info('%s\t%.4f' % (m, metricsValidation[m]))

        logger.info('Test performance, imputation error modality 2 from 1')
        for m in metricsTest:
            logger.info('%s\t%.4f' % (m, metricsTest[m]))

        metricsTestIndividual2from1 = evaluatePerDatapoint(net2from1, device, test_loader_individual, True)


        # modality 1 from 2
        dataTrain = [torch.tensor(omic, device=device) for omic in omics_train[::-1]]
        dataValidation = [torch.tensor(omic, device=device) for omic in omics_val[::-1]]
        dataTest = [torch.tensor(omic, device=device) for omic in omics_test[::-1]]

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


        early_stopping = EarlyStopping(patience=args['early_stopping_patience'], verbose=True)

        bestLoss, bestEpoch = trainRegressor(device=device, net=net1from2, num_epochs=args['epochs'], train_loader=train_loader,
              train_loader_eval=train_loader_eval, valid_loader=valid_loader,
              ckpt_dir=ckpt_dir, logs_dir=logs_dir, early_stopping=early_stopping)


        logger.info("Using model from epoch %d" % bestEpoch)

        checkpoint = torch.load(ckpt_dir + '/model_best.pth.tar')

        net1from2.load_state_dict(checkpoint['state_dict'])
        net1from2.eval()

        metricsValidation = evaluateUsingBatches(net1from2, device, valid_loader, True)
        metricsTest = evaluateUsingBatches(net1from2, device, test_loader, True)

        logger.info('Validation performance, imputation error modality 1 from 2')
        for m in metricsValidation:
            logger.info('%s\t%.4f' % (m, metricsValidation[m]))

        logger.info('Test performance, imputation error modality 1 from 2')
        for m in metricsTest:
            logger.info('%s\t%.4f' % (m, metricsTest[m]))

        metricsTestIndividual1from2 = evaluatePerDatapoint(net1from2, device, test_loader_individual, True)

        with open(os.path.join(save_dir, args['name'] + 'test_performance_per_datapoint.pkl'), 'wb') as f:
            pickle.dump({'1from2': metricsTestIndividual1from2, '2from1': metricsTestIndividual2from1}, f)


        performance = {'imputation_val': metricsValidation, 'imputation_test': metricsTest}



    # also classify
    if args['task'] > 1:
        alphas = np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20.])
        Nclasses = np.unique(ytrain).shape[0]
        print(Nclasses)

        if 'log_transform1' in args and args['log_transform1']:
            omics_train[0] = np.log(omics_train[0] + 1)
            omics_val[0] = np.log(omics_val[0] + 1)
            omics_test[0] = np.log(omics_test[0] + 1)

        if 'log_transform2' in args and args['log_transform2']:
            omics_train[1] = np.log(omics_train[1] + 1)
            omics_val[1] = np.log(omics_val[1] + 1)
            omics_test[1] = np.log(omics_test[1] + 1)



        pca1 = PCA(n_components=32, whiten=False, svd_solver='full')
        pca1.fit(omics_train[0])

        X1train = pca1.transform(omics_train[0])
        X1valid = pca1.transform(omics_val[0])
        X1test = pca1.transform(omics_test[0])

        pca2 = PCA(n_components=32, whiten=False, svd_solver='full')
        pca2.fit(omics_train[1])

        X2train = pca2.transform(omics_train[1])
        X2valid = pca2.transform(omics_val[1])
        X2test = pca2.transform(omics_test[1])


        Xtrain = np.hstack((X1train, X2train))
        Xvalid = np.hstack((X1valid, X2valid))
        Xtest = np.hstack((X1test, X2test))

        assert Xtrain.shape[1] == Xtest.shape[1]
        # assert x_train.shape[1] == 50 * len(omic1_train_file)
        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))

        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(X1train, ytrain, X1valid, yvalid, X1test, ytest, alphas, args['clf_criterion'])

        logger.info('Test performance, classification task, modality 1')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        performanceClf1 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}

        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')

        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))

        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(X2train, ytrain, X2valid, yvalid, X2test, ytest, alphas, args['clf_criterion'])

        logger.info('Test performance, classification task, modality 2')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        performanceClf2 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}

        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')


        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))

        _, acc, pr, rc, f1, mcc, confMat, CIs = classification(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, alphas, args['clf_criterion'])

        performanceClf12 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}

        logger.info('Test performance, classification task, both modalities')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')


        performance['omic1'] = performanceClf1
        performance['omic2'] = performanceClf2
        performance['omic1+2'] = performanceClf12


        # MLP

        if 'level' in args:
            level = args['level']
            assert level == 'l3'
        else:
            level = 'l2'

        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))


        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(X1train, ytrain, X1valid, yvalid, X1test, ytest, 'type-classifier/eval/' + level + '/baseline_' + args['data1'] + '/')

        logger.info('Test performance, classification task, modality 1')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        performanceClf1 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}
        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')


        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))

        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(X2train, ytrain, X2valid, yvalid, X2test, ytest, 'type-classifier/eval/' + level + '/baseline_' + args['data2'] + '/')

        logger.info('Test performance, classification task, modality 2')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        performanceClf2 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}
        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')


        print('Train: %d samples, %d labels' % (ytrain.shape[0], np.unique(ytrain).shape[0]))
        print('Valid: %d samples, %d labels' % (yvalid.shape[0], np.unique(yvalid).shape[0]))
        print('Test: %d samples, %d labels' % (ytest.shape[0], np.unique(ytest).shape[0]))

        _, acc, pr, rc, f1, mcc, confMat, CIs = classificationMLP(Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest, 'type-classifier/eval/' + level + '/baseline_' + args['data1'] + '_' + args['data2'] + '/')

        performanceClf12 = {'acc': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'mcc': mcc, 'confMat': confMat, 'CIs': CIs}

        logger.info('Test performance, classification task, both modalities')
        logger.info('ACC: %.4f\tPR: %.4f\tRC: %.4f\tF1: %.4f\tMCC: %.4f' % (acc, np.mean(pr), np.mean(rc), np.mean(f1), mcc))
        print('Precision: %d' % pr.shape[0])
        print('Recall: %d' % rc.shape[0])
        print('F1: %d' % f1.shape[0])
        print('')


        performance['mlp_omic1'] = performanceClf1
        performance['mlp_omic2'] = performanceClf2
        performance['mlp_omic1+2'] = performanceClf12



    with open(os.path.join(save_dir, args['name'] + 'results_pickle'), 'wb') as f:
        pickle.dump(performance, f)
