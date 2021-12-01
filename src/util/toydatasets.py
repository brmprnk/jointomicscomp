import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons

def saveDataset(V1, V2, y, saveDir):
    sampleNames = np.array(['sample_%d' % i for i in range(V1.shape[0])])
    featNames1 = np.array(['feat1_%d' % i for i in range(V1.shape[1])])
    featNames2 = np.array(['feat2_%d' % i for i in range(V2.shape[1])])

    np.save(saveDir + 'view1.npy', V1)
    np.save(saveDir + 'view2.npy', V2)
    np.save(saveDir + 'labels.npy', y)
    np.save(saveDir + 'view1_features.npy', featNames1)
    np.save(saveDir + 'view2_features.npy', featNames2)
    np.save(saveDir + 'samples.npy', sampleNames)



def easy(N1, N2, N3, save=False, saveDir=None, plot=False, rnd_seed=42):
    # 3 classes, 2 modalities
    # both modalities are required to separate all 3 classes
    # separation is linear
    # returns the 2 matrices (1 for each modality) and a vector of class labels
    np.random.seed(rnd_seed)

    # points per class
    Ns = [N1, N2, N3]


    y = np.zeros(np.sum(Ns))
    y[:Ns[0]] = 1
    y[Ns[0]:Ns[0] + Ns[1]] = 2
    y[-Ns[2]:] = 3

    # dimensionality of feature space
    d = 3


    # modality 1, classes 2 and 3 overlap both are linearly separable from class 1
    m1 = np.zeros(d)
    m2 = 5 * np.ones(d)
    m3 = m2

    mm = [m1, m2, m3]

    C1 = np.eye(d)
    C2 = C1
    C3 = C1

    CC = [C1, C2, C3]

    V = []
    for m, C, N in zip(mm, CC, Ns):
        x = np.random.multivariate_normal(m, C, N)
        V.append(x)

    V1 = np.vstack(tuple(V))


    # modality 2, classes 1 and 3 overlap both are linearly separable from class 2
    m2 = np.zeros(d)
    m1 = 5 * np.ones(d)
    m3 = m1

    mm = [m1, m2, m3]

    C1 = np.eye(d)
    C2 = C1
    C3 = C1

    CC = [C1, C2, C3]

    V = []
    for m, C, N in zip(mm, CC, Ns):
        x = np.random.multivariate_normal(m, C, N)
        V.append(x)

    V2 = np.vstack(tuple(V))

    if plot:

        colors = ['C0', 'C2', 'C5']
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)

        pca = PCA(n_components=2)
        xx1 = pca.fit_transform(V1)

        for yy in range(1,4):
            ax.scatter(xx1[y==yy, 0], xx1[y==yy, 1], color=colors[yy-1], s=10, label='Class %d' % yy)

        ax.set_aspect('equal')
        ax.set_title('Modality 1')
        plt.legend()


        ax = fig.add_subplot(2,1,2)

        pca = PCA(n_components=2)
        xx2 = pca.fit_transform(V2)

        for yy in range(1,4):
            ax.scatter(xx2[y==yy, 0], xx2[y==yy, 1], color=colors[yy-1], s=10, label='Class %d' % yy)

        ax.set_aspect('equal')
        ax.set_title('Modality 2')

        plt.legend()

        plt.tight_layout()



        plt.show()

    # shuffle order
    rndInd = np.random.permutation(V1.shape[0])
    V1 = V1[rndInd]
    V2 = V2[rndInd]
    y = y[rndInd]


    if save:
        assert saveDir is not None, 'If save is True, provide path to save folder'
        saveDataset(V1, V2, y, saveDir)


    return V1, V2, y



def hard(N1, N2, N3, save=False, saveDir=None, plot=False, rnd_seed=42):
    # 3 classes, 2 modalities
    # both modalities are required to separate all 3 classes
    # separation is linear
    # returns the 2 matrices (1 for each modality) and a vector of class labels
    np.random.seed(rnd_seed)

    # points per class
    Ns = [N1, N2, N3]


    y = np.zeros(np.sum(Ns))
    y[:Ns[0]] = 1
    y[Ns[0]:Ns[0] + Ns[1]] = 2
    y[-Ns[2]:] = 3

    # dimensionality of feature space
    d = 2


    # modality 1, classes 1 and 3 overlap. class 2 is split into 2 linearly separable clusters

    # class 1, uniform in circle of radius 1
    r1 = np.random.rand(Ns[0])
    theta1 = np.random.rand(Ns[0]) * 2 * np.pi - np.pi

    x1_1 = r1 * np.cos(theta1)
    x1_2 = r1 * np.sin(theta1)

    V = [np.array([x1_1, x1_2])]

    # class 2, two gaussians with means at +/- 5
    m1 = [3, 0]
    m2 = [-3, 0]

    C = np.array([[0.5, 0], [0, 1]])

    x3_1 = np.random.multivariate_normal(m1, C, Ns[-1]//2)
    x3_2 = np.random.multivariate_normal(m2, C, Ns[-1]//2)

    V.append(x3_1.T)
    V.append(x3_2.T)

    # class 3, uniform in circle of radius 0.5
    r2 = 0.5 * np.random.rand(Ns[0])
    theta2 = np.random.rand(Ns[0]) * 2 * np.pi - np.pi

    x2_1 = r2 * np.cos(theta2)
    x2_2 = r2 * np.sin(theta2)

    V.append(np.array([x2_1, x2_2]))



    V1 = np.hstack(tuple(V)).T


    # modality 2, each class on a 1D manifold, classes 1 and 2 are close to each other and linearly separable, class 3 is non-linearly separable
    V2, yy = make_moons(n_samples=tuple([Ns[0]+Ns[1], Ns[2]]), shuffle=False)
    print(yy)

    if plot:

        colors = ['C0', 'C2', 'C5']
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)

        pca = PCA(n_components=2)
        xx1 = V1
        for yy in range(1,4):
            ax.scatter(xx1[y==yy, 0], xx1[y==yy, 1], color=colors[yy-1], s=10, label='Class %d' % yy)

        ax.set_aspect('equal')
        ax.set_title('Modality 1')
        plt.legend()


        ax = fig.add_subplot(2,1,2)

        xx2 = V2

        for yy in range(1,4):
            ax.scatter(xx2[y==yy, 0], xx2[y==yy, 1], color=colors[yy-1], s=10, label='Class %d' % yy)

        ax.set_aspect('equal')
        ax.set_title('Modality 2')

        plt.legend()

        plt.tight_layout()



        plt.show()

    if save:
        assert saveDir is not None, 'If save is True, provide path to save folder'
        saveDataset(V1, V2, y, saveDir)



    return V1, V2, y



# V1, V2, y = hard(True)
