#!/usr/bin/env python
# coding: utf-8

def plotROC(model, X, y, cv, atlas, kind, classifier):
    
    # #############################################################################
    # ROC analysis
    
    path = 'C:\\Users\\xyang\\Desktop\\research\\PytorchStudy\\results\\'

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import auc
    from sklearn.metrics import plot_roc_curve

    # Run classifier with cross-validation and plot ROC curves
    #cv = StratifiedKFold(n_splits = 5)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        viz = plot_roc_curve(model, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title= "Model: " + str(classifier) + "   Atlas: "+ str(atlas) + "   Kind: " + str(kind))
    ax.legend(loc="lower right")
    plt.savefig(str(path) + str(classifier) + '_' + str(atlas) + '_' + str(kind) + '.png')
    plt.show()

