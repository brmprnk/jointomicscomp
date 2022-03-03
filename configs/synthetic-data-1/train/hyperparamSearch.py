import yaml

# models = ['cgae', 'mvib', 'poe', 'moe']
models = ['cgae', 'mvib']


lrs = [1e-4, 1e-3]
enc_archs = ['1', '2', '2-1', '2-2-1', '3-2-1']
mi_archs = ['100-1', '500-1', '500-100-1']
dropouts = [0.0, 0.1]
batchnorm = [False, True]


for model in models:
    with open(model + '_default.yaml') as file:
        config = yaml.safe_load(file)

    counter = -1

    baseName = config['GLOBAL_PARAMS']['name']

    if model == 'mvib':

        for bn in batchnorm:
            config['MVIB']['use_batch_norm'] = bn

            for dropoutP in dropouts:
                config['MVIB']['dropout_probability'] = dropoutP



                for mi_arch in mi_archs:
                    config['MVIB']['mi_net_arch'] = mi_arch

                    for learning_rate in lrs:
                        config['MVIB']['enc1_lr'] = learning_rate
                        config['MVIB']['enc2_lr'] = learning_rate
                        config['MVIB']['dec1_lr'] = learning_rate
                        config['MVIB']['dec2_lr'] = learning_rate

                        for arch in enc_archs:
                            config['MVIB']['latent_dim'] = arch

                            counter += 1

                            config['GLOBAL_PARAMS']['name'] = baseName + '-' + str(counter)

                            with open(model + '_' + str(counter) + '.yaml', 'w') as writeFile:
                                yaml.safe_dump(config, writeFile)

    elif model == 'cgae':

        for bn in batchnorm:
            config['CGAE']['use_batch_norm'] = bn

            for dropoutP in dropouts:
                config['CGAE']['dropout_probability'] = dropoutP

                for learning_rate in lrs:
                    config['CGAE']['enc1_lr'] = learning_rate
                    config['CGAE']['enc2_lr'] = learning_rate
                    config['CGAE']['dec1_lr'] = learning_rate
                    config['CGAE']['dec2_lr'] = learning_rate

                    for arch in enc_archs:
                        config['CGAE']['latent_dim'] = arch
                        counter += 1

                        config['GLOBAL_PARAMS']['name'] = baseName + '-' + str(counter)

                        with open(model + '_' + str(counter) + '.yaml', 'w') as writeFile:
                            yaml.safe_dump(config, writeFile)

    else:
        pass