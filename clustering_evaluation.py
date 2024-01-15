def clustering_evaluation(data, true_label, metrics='ACC'):
    """
    :param data: (N, C)
    :param true_label: (N, )
    :param metrics: ACC or NMI
    :return:
    """

    # data = normalize(data, axis=1)
    kmeans = KMeans(n_clusters=len(np.unique(true_label))).fit(data)
    pred_label = kmeans.labels_

    # Munkres方法重新分配簇号
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]

            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
        new_predict1 = np.zeros(len(pred_label))
        for i in range(len(new_predict1)):
            if new_predict1[i] == true_label[i]:
                new_predict1[i] = 1

    if 'ACC' in metrics:
        acc = np.round(accuracy_score(true_label, new_predict), 5)
        print('ACC: ', acc)
    if 'NMI' in metrics:
        nmi = np.round(normalized_mutual_info_score(true_label, new_predict), 5)
        print('NMI: ', nmi)
    if 'ARI' in metrics:
        ari = np.round(adjusted_rand_score(true_label, new_predict), 5)
        print('ARI: ', ari)

    return new_predict

def plot_result(data, true_label, pred_label, save_file='result.png'):
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  tsne = TSNE(n_components=2, verbose=1)
  data = tsne.fit_transform(data)
  plt.figure(figsize=(16, 8))
  plt.subplot(121)
  plt.scatter(data[:, 0], data[:, 1], c=true_label, s=0.8, cmap='rainbow')
  plt.title('true label')
  plt.subplot(122)
  plt.scatter(data[:, 0], data[:, 1], c=pred_label, s=0.8, cmap='rainbow')
  plt.title('pred label')
  plt.savefig(save_file)
