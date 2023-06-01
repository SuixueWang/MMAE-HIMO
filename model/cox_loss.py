import torch, random
from lifelines.utils import concordance_index
import numpy as np
import math
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt

def PartialLogLikelihood2(logits, fail_indicator, time, ties=None):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model_backup
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # pre-calculate cumsum

    # 先根据生存时间排序，最终目的是从小到大排序。但是对logit的指数累积求和得从后往前计算，计算完再变更回来从前往后
    # fail_indicator = fail_indicator[torch.argsort(time, descending=False)]
    # logits = logits[torch.argsort(time, descending=False)]

    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    # cumsum_hazard_ratio = torch.flip(cumsum_hazard_ratio, dims=[0])

    if ties == 'noties':
        likelihood = logits - torch.log(cumsum_hazard_ratio)
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()

    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix

def PartialLogLikelihood(hazard_pred, censor, survtime, ties=None):

    n_observed = censor.sum(0)+1
    ytime_indicator = R_set(survtime)
    ytime_indicator = torch.FloatTensor(ytime_indicator).to(survtime.device)
    risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred).float())
    diff = hazard_pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1).float())
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost

def calc_concordance_index(logits, fail_indicator, fail_time):
    """
    Compute the concordance-index value.
    Parameters:
        label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        y_pred: np.array, predictive proportional risk of network.
    Returns:
        concordance index.
    """

    logits = logits.cpu().detach().numpy()
    fail_indicator = fail_indicator.cpu().detach().numpy()
    fail_time = fail_time.cpu().detach().numpy()

    hr_pred = -logits
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)

    # ci = concordance_index_censored(fail_indicator.astype(np.bool_), fail_time, logits)

    return ci

def cox_log_rank(hazardsdata, labels, survtime_all):

    hazardsdata = hazardsdata.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    survtime_all = survtime_all.cpu().detach().numpy()

    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def test_lifetime():
    from lifelines import KaplanMeierFitter
    from lifelines.datasets import load_waltons
    waltons = load_waltons()

    kmf = KaplanMeierFitter(label="waltons_data")
    kmf.fit(waltons['T'], waltons['E'])
    kmf.plot()

    plt.show()

if __name__ == "__main__":

    test_lifetime()