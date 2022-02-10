from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def plot_myroc(sk_probs, cv_probs, test_y):
    
    # keep probabilities for the positive outcome only
    sk_probs1 = sk_probs[:, 1]
    cv_probs1 = cv_probs[:, 1]

    # calculate scores
    cv_auc = roc_auc_score(test_y, cv_probs1)
    lr_auc = roc_auc_score(test_y, sk_probs1)

    # summarize scores
    #print('Cross-Validated: ROC AUC=%.3f' % (cv_auc))
    #print('Basic Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    cv_fpr, cv_tpr, _ = roc_curve(test_y, cv_probs1)
    lr_fpr, lr_tpr, _ = roc_curve(test_y, sk_probs1)

    # plot the roc curve for the model
    pyplot.plot(cv_fpr, cv_tpr, linestyle='--', label='Cross-Validated ROC AUC=%.3f' % (cv_auc))
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Basic Logistic ROC AUC=%.3f' % (lr_auc))
    
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    return(pyplot)

def plot_mynewroc(sk_probs, cv_probs, test_y):
    
    # keep probabilities for the positive outcome only
    sk_probs1 = sk_probs[:, 1]
    cv_probs1 = cv_probs[:, 1]

    # calculate scores
    cv_auc = roc_auc_score(test_y, cv_probs1)
    lr_auc = roc_auc_score(test_y, sk_probs1)

    # summarize scores
    #print('Cross-Validated: ROC AUC=%.3f' % (cv_auc))
    #print('Basic Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    cv_fpr, cv_tpr, _ = roc_curve(test_y, cv_probs1)
    lr_fpr, lr_tpr, _ = roc_curve(test_y, sk_probs1)

    # plot the roc curve for the model
    pyplot.plot(cv_fpr, cv_tpr, linestyle='--', label='Cross-Validated ROC AUC=%.3f' % (cv_auc))
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Basic Logistic ROC AUC=%.3f' % (lr_auc))
    
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    return(pyplot)