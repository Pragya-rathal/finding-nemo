from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np

def topk_acc(logits,targets,k=1):
    pred=np.argsort(-logits,axis=1)[:,:k]
    return float(np.mean([t in p for t,p in zip(targets,pred)]))

def cls_metrics(logits,targets):
    pred=np.argmax(logits,1)
    return {'top1':topk_acc(logits,targets,1),'top5':topk_acc(logits,targets,5),'macro_f1':f1_score(targets,pred,average='macro'),'balanced_acc':balanced_accuracy_score(targets,pred)}
