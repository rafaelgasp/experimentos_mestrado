import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

def get_traces_as_tokens(traces_df, col_ref="Activity"):
    """
        Groups activities executions into traces
        as a string of tokens
        
        Ex:
             Activity |  Timestamp
            -----------------------
            START     | 2019-09-01
            A         | 2019-09-01
            B         | 2019-09-01
            C         | 2019-09-01
            END-A     | 2019-09-01
            
       into: "START A B C END-A"         
    """
    return traces_df.groupby("Trace_order")[col_ref].apply(
        lambda x: " ".join(x.values)
    )

def get_count_representation(tokens, binary=True, tfidf=False, ngram_range=(1, 1)):
    """
        Generic method to represent traces as vectors by counting
        activities or transitions
    """
    
    if tfidf:
        cv = TfidfVectorizer(
            norm = None,
            smooth_idf = False,
            tokenizer=str.split, 
            lowercase=False,
            use_idf=True,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0
        )
    else:
        cv = CountVectorizer(
            tokenizer=str.split, 
            lowercase=False,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0,
            binary=binary
        )
    
    cv_result = cv.fit_transform(tokens)
    
    return pd.DataFrame(
        cv_result.todense(), 
        columns=cv.get_feature_names()
    )

# # # # # # # #
# Transitions #
# # # # # # # #
def get_binary_transitions_representation(tokens):    
    """
        Binary Transistions representation of traces
        (1 or 0 if the transition occur in traces)
    """
    return get_count_representation(tokens, True, False, (2,2))

def get_frequency_transitions_representation(tokens):    
    """
        Frequency Transistions representation of traces
        (# of occurences of a transition on the trace)
    """
    return get_count_representation(tokens, False, False, (2,2))

def get_tfidf_transitions_representation(tokens):    
    """
        TF-IDF Transistions representation of traces
        (frequency of the transition occur in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (2,2))


# # # # # # 
# Activity #
# # # # # # 
def get_binary_representation(tokens):    
    """
        Binary representation of traces
        (1 or 0 if an activity occur in traces)
    """
    return get_count_representation(tokens, True, False, (1,1))

def get_frequency_representation(tokens):    
    """
        Frequency representation of traces
        (# of times an activity occur in traces)
    """
    return get_count_representation(tokens, False, False, (1,1))

def get_tfidf_representation(tokens):    
    """
        TF-IDF representation of traces
        (frequency of the occurence of activity in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (1,1))


# # # # # #
# Position #
# # # # # # 
def trace_to_positions(trace, activities_list):
    """
        Represents trace by the order/position in
        which the activities occur
    """
    r = {}
    
    for activity in activities_list:
        found_at = []
        last_found = 0
        
        while True:
            try:
                idx = trace.index(activity, last_found)
                found_at.append(idx)                   
                last_found = idx + 1
            except ValueError:
                break
        
        if len(found_at) == 0:
            r[activity + "_min"] = -1
            r[activity + "_max"] = -1
            r[activity + "_avg"] = -1
            r[activity + "_middle"] = -1
        else:
            r[activity + "_min"] = min(found_at)
            r[activity + "_max"] = max(found_at)
            r[activity + "_avg"] = np.mean(found_at)
            r[activity + "_middle"] = np.mean([min(found_at), max(found_at)])
    
    return r


def get_positions_representation(tokens, include_cols=["_min", "_max", "_middle", "_avg"]):
    """
        Position/order representation of traces
    """
    lista_atividades = tokens.apply(lambda x: x.split()).explode().unique()
    positions = tokens.apply(
        lambda x: trace_to_positions(x.split(), lista_atividades)
    )
    resp = pd.DataFrame(positions.values.tolist())
    
    to_include = []
    for sufix_col in include_cols:
        for col in resp.columns:
            if sufix_col in col:
                to_include.append(col)

    return resp[to_include]

def get_min_max_positions_representation(tokens):
    return get_positions_representation(tokens, include_cols=["_min", "_max"])

def get_avg_positions_representation(tokens):
    return get_positions_representation(tokens, include_cols=["_avg", "_middle"])


# # # # # # # # # #
# Extra Functions #
# # # # # # # # # #
def reinverse_tokens(tokens, inv_aliases, ret_string=True):
    """
        Invert aliases back to full activities names
    """
    r = []
    
    if isinstance(tokens, str):
        t = tokens.split()
    else:
        t = tokens
    
    for token in t:
        if token in inv_aliases:
            r.append(inv_aliases[token])
        else:
            r.append(token)
    
    if ret_string:
        return " ".join(r)
    
    return r

def get_alpha_concurrency_pairs(tokens, expand=True):    
    cv_pairs = CountVectorizer(
        tokenizer=str.split, 
        lowercase=False,
        ngram_range=(2, 2)
    )
    
    cv_pairs.fit(tokens.values)
    
    sorted_pairs = []
    for pair in cv_pairs.get_feature_names():
        sorted_pairs.append(sorted(pair.replace(" ", "")))
        
    values, counts = np.unique(sorted_pairs, return_counts=True)
    
    if expand:
        def expand_pairs(concurrent_pairs):
            resp = []

            for pair in concurrent_pairs:
                resp.append([pair[0], pair[1]])
                resp.append([pair[1], pair[0]])

            return resp
        
        return [tuple(x) for x in expand_pairs(values[counts > 1].tolist())]
    else:
        return [tuple(x) for x in values[counts > 1].tolist()]

def trace_to_run(trace, pair):
    t = trace
    
    ordered = " ".join(pair[1] + pair[0])
    unordered = " ".join(pair[0] + pair[1])

    t = t.replace(unordered, ordered)
    
    #print(t, pair, t.replace(unordered, ordered))
    
    return t

def traces_to_runs(tokens, concurrent_pairs):
    resp = []
    
    for trace in tokens:
        t = trace
        for pair in concurrent_pairs:
            t = trace_to_run(t, pair)
            
        resp.append(t)
    
    resp = pd.Series(resp)
    resp.index = tokens.index
    
    return resp