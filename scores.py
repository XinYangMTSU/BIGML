#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def getScores(scores):
    
    import numpy as np
    
    acc_test = "{:.2%}".format(round(np.mean(scores['test_accuracy']),4))
    sen_test = "{:.2%}".format(round(np.mean(scores['test_sensitivity']),4))
    spe_test = "{:.2%}".format(round(np.mean(scores['test_specificity']),4))

    acc_train = "{:.2%}".format(round(np.mean(scores['train_accuracy']),4))
    sen_train = "{:.2%}".format(round(np.mean(scores['train_sensitivity']),4))
    spe_train = "{:.2%}".format(round(np.mean(scores['train_specificity']),4))

#     print("accuracy_test:",    acc_test)
#     print("sensitivity_test:", sen_test) 
#     print("specificity_test:", spe_test) 

#     print("accuracy_train:",    acc_train)
#     print("sensitivity_train:", sen_train) 
#     print("specificity_train:", spe_train)
    
    return [acc_test, sen_test, spe_test]
    
    
