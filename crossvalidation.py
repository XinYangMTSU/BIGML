#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def CrossValidation(model,measure,y,skf,atalas):
 
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score, recall_score
    import numpy as np
    
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0)
    }
    
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.connectome import sym_matrix_to_vec

    conn_est = ConnectivityMeasure(kind = measure)
    conn_matrices = conn_est.fit_transform(atalas)
    X = sym_matrix_to_vec(conn_matrices)
    
    from sklearn.model_selection import cross_validate

    scores = cross_validate(model, X, y, cv = skf,
                             scoring = scoring,
                             return_train_score=True)
    return [X, scores]
 
    
    #return [X, acc_test, sen_test, spe_test, acc_train, sen_train, spe_train]
