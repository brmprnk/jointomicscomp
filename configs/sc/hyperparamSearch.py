import yaml

models = ['cgae', 'mvib', 'poe', 'moe']
#models = ['cgae', 'cvae', 'moe']

lrs = [1e-4, 1e-3]
enc_archs = ['32', '64', '128-32', '128-64', '256-32', '256-64', '256-256-32', '256-256-64', '256-128-32', '256-128-64']
mi_archs = ['100-1', '500-1', '500-100-1']
dropouts = [0.0, 0.1]
batchnorm = [False, True]
Ks = [10, 20]

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
                        config['MVIB']['enc1_lr'] = learning_rate * 0.1
                        config['MVIB']['enc2_lr'] = learning_rate * 0.1
                        config['MVIB']['dec1_lr'] = learning_rate * 0.1
                        config['MVIB']['dec2_lr'] = learning_rate * 0.1
                        config['MVIB']['mi_net_lr'] = learning_rate * 0.1

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


    elif model == 'cvae':

        for bn in batchnorm:
            config['CVAE']['use_batch_norm'] = bn

            for dropoutP in dropouts:
                config['CVAE']['dropout_probability'] = dropoutP

                for learning_rate in lrs:
                    config['CVAE']['lr'] = learning_rate

                    for arch in enc_archs:
                        config['CVAE']['latent_dim'] = arch
                        counter += 1

                        config['GLOBAL_PARAMS']['name'] = baseName + '-' + str(counter)

                        with open(model + '_' + str(counter) + '.yaml', 'w') as writeFile:
                            yaml.safe_dump(config, writeFile)


    elif model == 'poe':

        for bn in batchnorm:
            config['PoE']['use_batch_norm'] = bn

            for dropoutP in dropouts:
                config['PoE']['dropout_probability'] = dropoutP

                for learning_rate in lrs:
                    config['PoE']['lr'] = learning_rate

                    for arch in enc_archs:
                        config['PoE']['latent_dim'] = arch
                        counter += 1

                        config['GLOBAL_PARAMS']['name'] = baseName + '-' + str(counter)

                        with open(model + '_' + str(counter) + '.yaml', 'w') as writeFile:
                            yaml.safe_dump(config, writeFile)


    else:
        assert model == 'moe'
        for bn in batchnorm:
            config['MoE']['use_batch_norm'] = bn

            for dropoutP in dropouts:
                config['MoE']['dropout_probability'] = dropoutP

                for kk in Ks:
                    config['MoE']['K'] = kk

                    for learning_rate in lrs:
                        config['MoE']['enc1_lr'] = learning_rate
                        config['MoE']['enc2_lr'] = learning_rate
                        config['MoE']['dec1_lr'] = learning_rate
                        config['MoE']['dec2_lr'] = learning_rate

                        for arch in enc_archs:
                            config['MoE']['latent_dim'] = arch

                            counter += 1

                            config['GLOBAL_PARAMS']['name'] = baseName + '-' + str(counter)

                            with open(model + '_' + str(counter) + '.yaml', 'w') as writeFile:
                                yaml.safe_dump(config, writeFile)
