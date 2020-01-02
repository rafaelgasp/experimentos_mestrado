# wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=14UBHX6GTW_4YiyjNJB3EEq7Xb83AjuaK' -O process_mining_datasets.zip

import pandas as pd
# pd.set_option("max_columns", 200)
import numpy as np
from tqdm import tqdm

%load_ext autoreload
%autoreload 2

# Meus pacotes
import parse_mxml as pm
import log_representation as lr
import plots as plts
import model_utils as mu
import drift_detection as dd
import offline_streaming_clustering as off_sc

from scipy.spatial import distance
from sklearn.base import clone as sk_clone 

from copy import deepcopy
import random
random.seed(42)
import os
import warnings
warnings.filterwarnings("ignore")

import glob

from sklearn.cluster import KMeans, AgglomerativeClustering

import gc
gc.enable()

# # # # # # # # # # #
# LOAN APPLICATIONS #
# # # # # # # # # # #
aliases = {
    'Loan__application_received': 'START',
    'Appraise_property': 'A',
    'Approve_application': 'B',
    'Assess_eligibility': 'C',
    'Assess_loan_risk': 'D',
    'Cancel_application': 'E',
    'Check__application__form_completeness': 'F',
    'Check_credit_history': 'G',
    'Check_if_home_insurance_quote_is_requested': 'H',
    'Prepare_acceptance_pack': 'I',
    'Receive_updated_application': 'J',
    'Reject_application': 'K',
    'Return_application_back_to_applicant': 'L',
    'Send_acceptance_pack': 'M',
    'Send_home_insurance_quote': 'N',
    'Verify_repayment_agreement': 'O',
    'Loan__application_approved': 'END_A',
    'Loan_application_rejected': 'END_R',
    'Loan__application_canceled': 'END_C',
}

inv_aliases = {v: k for k, v in aliases.items()}

logs = glob.glob("process_mining_datasets/*/*k.MXML")