import json
import re

import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats

def get_performance_trials(runs):
    perf_dict = {'solved': 0,
                 '25%': 0, 
                 '50%': 0, 
                 '75%':0,
                 '0%': 0, 
                 'failure': 0}
    for run in runs:
        run = run.strip()
                        
        # 1. Find all punctuation indices (. ! ?)
        # We use finditer to get the exact positions of punctuation
        punct_matches = list(re.finditer(r'[.!?]', run))

        last_sentence = ""
        # Logic: If we have at least 2 punctuation marks, grab everything 
        # after the second-to-last one.
        if len(punct_matches) >= 2:
            # Get the end index of the second-to-last punctuation mark
            split_index = punct_matches[-2].end()
            last_sentence = run[split_index:].strip()
        elif len(punct_matches) == 1:
            # If there is only one sentence, take the whole thing
            last_sentence = run.strip()
        else:
            # Fallback if no punctuation is found
            last_sentence = run.strip()[-100:]
        #print(last_sentence)
        if 'solution' in last_sentence or 'solved' in last_sentence:
            perf_dict['solved'] += 1
        elif ' 0%' in last_sentence:
            perf_dict['0%'] += 1
        elif '25%' in last_sentence:
            perf_dict['25%'] += 1
        elif '50%' in last_sentence:
            perf_dict['50%'] += 1
        elif '75%' in last_sentence:
            perf_dict['75%'] += 1
        else: 
            perf_dict['failure'] += 1
        #print(perf_dict)

    return perf_dict

def make_perf_list(dict):
    perf_list = []
    for k in dict:
        if k == "25%":
            val = 0.25
        elif k == "0%":
            val = 0.0
        elif k == "50%":
            val = 0.5
        elif k =="75%":
            val = 0.75
        elif k == "solved":
            val = 1.0
        else:
            continue
        perf_list = perf_list + dict[k]*[val]
    return perf_list

def get_significance_test(LLMscores, nobuff_score, random_scores):
    comparisons = [
    ("Random", random_scores, "None", nobuff_score),
    ("Selected", LLMscores, "None", nobuff_score),
    ("Selected", LLMscores, "Random", random_scores)
    ]

    print(f"{'Comparison':<30} | {'U-Stat':<10} | {'P-Value':<10} | {'Significant?'}")
    print("-" * 65)

    alpha = 0.05  # Standard threshold
    # Bonferroni correction: dividing alpha by number of tests (3) to avoid false positives
    corrected_alpha = alpha / 3 

    for name_a, data_a, name_b, data_b in comparisons:
        # 'alternative="greater"' tests if Group A is GREATER than Group B
        u_stat, p_val = stats.mannwhitneyu(data_a, data_b, alternative='greater')
        
        is_sig = "YES" if p_val < corrected_alpha else "NO"
        
        print(f"{name_a} > {name_b:<17} | {u_stat:<10.1f} | {p_val:<10.4f} | {is_sig}")

def get_buff_perf(dir, file):
    with open(dir + file, 'r') as fp:
        buff_runs = json.load(fp)
    #print(len(nobuff_runs))
    buff = get_performance_trials(buff_runs)
    return buff
if __name__ == "__main__":
    dir = "hanoi_caches/"
    # with open(dir + 'hanoi_4disk_empty_cache.json', 'r') as fp:
    #     nobuff_runs = json.load(fp)
    # print(len(nobuff_runs))
    # nobuff = get_performance_trials(nobuff_runs)

    # print(nobuff)

    all_LLM_sep_runs = []
    for i in range(1, 7):
        with open(dir + f'hanoi_4disk_LLM_replay_sep_cache_{i}.json', 'r') as fp:
            tmp = json.load(fp)
        all_LLM_sep_runs = all_LLM_sep_runs + tmp

    print(len(all_LLM_sep_runs))
    LLMbuff = get_performance_trials(all_LLM_sep_runs)
    print(LLMbuff)

    # all_random_runs = []
    # for i in range(1, 7):
    #     with open(dir + f'hanoi_4disk_random_cache_{i}.json', 'r') as fp:
    #         tmp = json.load(fp)
    #     all_random_runs = all_random_runs + tmp

    # print(len(all_random_runs))
    # random_buff = get_performance_trials(all_random_runs)
    # print(random_buff)

    categories = ['solved', '25%', '50%', '0%', 'failure']

    # table = np.array([
    #     [nobuff[c] for c in categories],
    #     [LLMbuff[c] for c in categories]
    # ])

    # chi2, p, dof, expected = chi2_contingency(table)

    # print(f"Chi-square statistic: {chi2:.3f}")
    # print(f"Degrees of freedom: {dof}")
    # print(f"p-value: {p:.4f}")
        
    # table2 = np.array([
    #     [nobuff[c] for c in categories],
    #     [random_buff[c] for c in categories]
    # ])

    # chi2, p, dof, expected = chi2_contingency(table2)

    # print(f"Chi-square statistic: {chi2:.3f}")
    # print(f"Degrees of freedom: {dof}")
    # print(f"p-value: {p:.4f}")

    # empty_scores = make_perf_list(nobuff)
    LLM_scores = make_perf_list(LLMbuff)
    # random_scores = make_perf_list(random_buff)

    # get_significance_test(LLM_scores, empty_scores, random_scores)


    # with open("hanoi_caches/"+ "hanoi_experience_5disk_cache_1.json", 'r') as fp:
    #     only3success = json.load(fp)
    
    # print(len(only3success))
    # only3 = get_performance_trials(only3success[1:])
    # print(only3)

    dir = "hanoi_caches/"
    opt3 = get_buff_perf(dir, "hanoi_4disk_onlysuccess3_cache_1.json")
    opt4 = get_buff_perf(dir, "hanoi_4disk_onlysucess_cache_1.json")
    #opt4 = opt4[:10]
    sub4 = get_buff_perf(dir, "hanoi_4disk_onlysucessSuboptimal_cache_1.json")
    opt5 = get_buff_perf(dir, "hanoi_5disk_optimal_cache_1.json")
    print(opt3)
    print(opt4)
    print(sub4)
    print(opt5)
    opt3['solved'] -=1
    sub4['solved'] -= 1
    opt5['solved'] -= 1
    opt3_scores = make_perf_list(opt3)
    sub4_scores = make_perf_list(sub4)
    opt5_scores = make_perf_list(opt5)

    get_significance_test(LLM_scores, opt3_scores, opt5_scores)

