from MedRoBERTaNL_wrapper import RobertaForMaskedLM_embedding_model
import hdbscan
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from try_utils import parse_commandline_args, check_if_debugging
from itertools import repeat
from pathlib import Path
from boruta import BorutaPy
import tensorflow_text
from xgboost import XGBRegressor
from lifelines.utils import concordance_index
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import (check_random_state, check_is_fitted, _check_sample_weight)
from datetime import timedelta, date
from kneed import KneeLocator
from scipy.stats import sem
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
from gensim.models.phrases import Phrases
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
import io
from multiprocessing import Pool as ProcessPool
from transformers import AutoTokenizer, AutoModelForMaskedLM
from TargetHF_ty.targethf.definitions.cohort_def import COHORT_TY_START
import umap
from collections import Counter
from GaussMMStepMix import run_it
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from gensim.utils import simple_preprocess
from stepmix.stepmix import StepMix, StepMixClassifier
from matplotlib import pyplot as plt
from sklearn import set_config
from time import sleep as wait
import re
import tempfile
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.decomposition import TruncatedSVD
from TargetHF_ty.targethf.definitions import icpc_def
from wordcloud import WordCloud
from itertools import product
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
import pyarrow as pa
from sklearn.metrics import roc_auc_score
from stepmix.utils import get_mixed_descriptor
from gensim.parsing.preprocessing import strip_tags
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sys import stdout
from TargetHF_ty.targethf.data.tagging import multi_boolex, icpc_match
from sklearn.linear_model import LogisticRegression
import debugpy
from constants import (T, F)
from sklearn.preprocessing import normalize
from sklearn.cluster import dbscan
from stepmix.stepmix import StepMix
import tensorflow as tf
from sys import exit
from scipy.sparse import hstack
from tensorflow.keras.models import Model
from flask import render_template, make_response
from scipy.spatial.distance import pdist
from Top2Vec import Top2Vec
from sklearn.feature_extraction.text import CountVectorizer
from TargetHF_ty.targethf.data.distillation import groupby_notna
from html import escape as html_escape
import xgboost as xgb
from sklearn.metrics import (make_scorer, roc_auc_score, accuracy_score, confusion_matrix, classification_report, rand_score, silhouette_score)
import seaborn as sns
import sys
from TargetHF_ty.targethf.data.tagging import year_diff
from datetime import datetime as dt
from functools import reduce
from itertools import product 
import hnswlib
# from RegExSystemLukas import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import rand_score
import statsmodels.api as sm
from TargetHF_ty.targethf.data.tagging import year_delta, year_diff
from multiprocessing import Pool
from symspellpy.symspellpy import SymSpell, Verbosity
from TargetHF_ty.targethf.definitions import icpc_def, text_def
from try_utils import (get_default_logger_fn, is_cluster_interesting)
from scipy.special import softmax
import gc 
import scipy.stats as st
from itertools import chain
from GaussMMStepMix import run_it as run_LCA 
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from try_utils import parse_commandline_args, check_if_debugging, get_default_logger_fn
import gc
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score)
import simplejson as json
import functools as ft
import random
import seaborn as sns 
from constants import T, F
from os import environ
from textEmbeddingT2V import run_it
from VariableSelection import run_it
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, Response
from try_utils import *
import numpy as np
import pickle
from lifelines.calibration import survival_probability_calibration
from try_utils import try_save_pickle, try_read_pickle
from try_logger import set_logfilename, get_logfilename, set_logger, get_logger
from collections import OrderedDict
from uuid import uuid4
from TargetHF_ty.targethf.definitions import icpc_def, cohort_def
from os import path
from sklearn.metrics import average_precision_score, roc_auc_score
import time
from try_utils import (try_table, try_expand, try_reduce, tty, try_multiindex, nrow)
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn import model_selection
from constants import T, F, THRESHOLD_MIN_AGE_DAYS
import html
from try_stepmix import StepMixBICScore
from constants import *
from pyarrow.parquet import ParquetFile
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored
import statsmodels.api as sm
from sklearn.pipeline import Pipeline




import tensorflow_hub as hub

use_model_urls = {
    "universal-sentence-encoder-multilingual": "https://kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2?tfhub-redirect=true",
    "universal-sentence-encoder": "https://kaggle.com/models/google/universal-sentence-encoder/4?tfhub-redirect=true",
    "universal-sentence-encoder-large": "https://kaggle.com/models/google/universal-sentence-encoder-large/5?tfhub-redirect=true",
    "universal-sentence-encoder-multilingual-large": "https://kaggle.com/models/google/universal-sentence-encoder-multilingual-large/3?tfhub-redirect=true"
}

for model,url in use_model_urls.items():
	print(f"loading model {model} from {url}...")
	_ = hub.load(url)