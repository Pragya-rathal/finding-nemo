import torch

def hierarchical_taxonomy(order,family,genus,species,th_species=0.6,th_genus=0.5,th_family=0.4):
    s_prob,s_id=species.softmax(-1).max(-1)
    g_prob,g_id=genus.softmax(-1).max(-1)
    f_prob,f_id=family.softmax(-1).max(-1)
    o_prob,o_id=order.softmax(-1).max(-1)
    resolved=[]
    for i in range(species.shape[0]):
        if s_prob[i]>=th_species:
            level='species'; idx=int(s_id[i]); conf=float(s_prob[i])
        elif g_prob[i]>=th_genus:
            level='genus'; idx=int(g_id[i]); conf=float(g_prob[i])
        elif f_prob[i]>=th_family:
            level='family'; idx=int(f_id[i]); conf=float(f_prob[i])
        else:
            level='order'; idx=int(o_id[i]); conf=float(o_prob[i])
        resolved.append({'level':level,'id':idx,'confidence':conf})
    return resolved
