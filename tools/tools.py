def inferType(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    
    try:
        return float(value)
    except ValueError:
        pass
   
    return value



def evaluateRule(rule, variables)->bool:
    return eval(rule,{}, variables)


def flatten(lst):
    flatten_list = []
    for item in lst:
        if isinstance(item, list):
            flatten_list.extend(flatten(item))
        else:
            flatten_list.append(item)
    return flatten_list