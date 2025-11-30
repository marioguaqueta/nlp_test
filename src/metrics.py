import json
from collections import Counter

def flatten_json(y, delimiter='.'):
    """
    Flattens a JSON object into a dictionary of paths and values.
    Lists are handled by including the index in the path, BUT to handle 
    "order doesn't matter", we should ideally sort lists or use a different strategy.
    
    Given the competition rules "Order of elements in lists ([]) does not affect the score",
    we will attempt to sort lists of dictionaries by their content before flattening,
    or treat them as a multiset of paths if possible.
    
    However, a simple robust way is to sort lists of primitives, and for lists of objects,
    sort them by a deterministic string representation.
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + delimiter)
        elif type(x) is list:
            try:
                sorted_x = sorted(x, key=lambda k: json.dumps(k, sort_keys=True))
            except Exception:
                sorted_x = x
            
            for i, a in enumerate(sorted_x):
                flatten(a, name + str(i) + delimiter)
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def calculate_f1(pred_json_str, true_json_str, key_weight=1, field_weight=9):
    """
    Calculates the weighted F1 score.
    """
    try:
        pred = json.loads(pred_json_str)
    except:
        return 0.0
    
    try:
        true = json.loads(true_json_str)
    except:
        return 0.0

    pred_flat = flatten_json(pred)
    true_flat = flatten_json(true)

    pred_paths = set(pred_flat.keys())
    true_paths = set(true_flat.keys())
    
    tp_score = 0
    fp_score = 0
    fn_score = 0
    
    all_paths = pred_paths.union(true_paths)
    
    for path in all_paths:
        in_pred = path in pred_paths
        in_true = path in true_paths
        
        if in_pred and in_true:
            tp_score += key_weight
        elif in_pred and not in_true:
            fp_score += key_weight
        elif not in_pred and in_true:
            fn_score += key_weight
            
        if in_pred and in_true:
            if str(pred_flat[path]) == str(true_flat[path]):
                tp_score += field_weight
            else:
                fp_score += field_weight
                fn_score += field_weight
        elif in_pred:
            fp_score += field_weight
        elif in_true:
            fn_score += field_weight

    epsilon = 1e-9
    precision = tp_score / (tp_score + fp_score + epsilon)
    recall = tp_score / (tp_score + fn_score + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1

