from os import major
from carla import MLModelCatalog
from carla.data.catalog.online_catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres, Wachter, ActionableRecourse, Face, Revise, Clue, CRUD 
from sklearn.linear_model import LogisticRegression
from carla.recourse_methods import CCHVAE, balanceCCHVAE
from carla import Benchmark
from carla.models.negative_instances import predict_negative_instances
import numpy as np
import argparse 
import logging 
from utils_recourse import *
from carla.data.catalog import CsvCatalog
import pandas as pd 
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--ar_model', type=str, default='wachter', help='recourse model to use')
parser.add_argument('--ml_model', type=str, default='lr', help='ml model to use')
parser.add_argument('--dataset', type=str, default='compas', help='dataset to use')
parser.add_argument('--sens', type=str, default='race', help='sens to use')
parser.add_argument('--seed', type=int, default=0, help='seed for the random number generator')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--bq', type=float, default=0.6, help='balance quantile')
parser.add_argument('--num_neg', type=int, default=0, help='number of ingative instance')
parser.add_argument('--balancedata', type=int, default=1, help='whether to prepare data or train')
parser.add_argument('--balancedatastage', type=int, default=1, help='whether to prepare data or train')
parser.add_argument('--forcetrain', type=int, default=1, help='whether to train the model')
parser.add_argument('--nrep', type=int, default=5, help='number of repititions')
parser.add_argument('--nrep0', type=int, default=0, help='number of repititions')
parser.add_argument('--gamma', type = int, default = 1)
args = parser.parse_args()

nrep = args.nrep 
if args.sens == 'sex':
    sen_feature = 'sex_Male_1.0'
if args.sens == 'age':
    sen_feature = 'age_1'

result = pd.DataFrame(columns = ['nrep','metric','value','ml_model','dataset','method'])
for it in np.arange(args.nrep0, args.nrep):
    seed_everything(it)
    if args.dataset == 'give_me_some_credit':
        dataset = OnlineCatalog('give_me_some_credit')
        df = dataset.df
        df.age = (df.age > np.quantile(df.age, 0.5)) * 1
        df = df.sample(10000)
        df.to_csv('./data/gmc/gmc.csv', index=False)
        categorical = ['age']
        immutable = ['age']
        immutable_features_col = ["age_1"]
        continuous = ['RevolvingUtilizationOfUnsecuredLines',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
        dataset = CsvCatalog(file_path="./data/gmc/gmc.csv",
                            continuous=continuous,
                            categorical=categorical,
                            immutables=immutable,
                            target='SeriousDlqin2yrs')
        mutable_feature = ['RevolvingUtilizationOfUnsecuredLines',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
        features = mutable_feature        
    elif args.dataset == 'law':
        data = pd.read_csv("./data/law/law_data.csv", index_col = 0)
        dataset_name = args.dataset
        categorical = ['race','sex','region_first']

        default = data['region_first'][0]
        data['region_first'][data['region_first'] == default] = 0
        data['region_first'][data['region_first'] != default] = 1        
        data['race'][data['race'] != 'White'] = 'Other'
        data = data.sample(10000)
        data.to_csv('./data/law/law.csv', index = False)
        
        #immutable = ["age", "status"]
        immutable = ["sex_2", "race_White", 'region_first_1']
        immutable_features_col = ['sex_2', 'race_White', 'region_first_1']
        continuous = ['LSAT','UGPA','ZFYA','sander_index']
        dataset = CsvCatalog(file_path="./data/law/law.csv",
                            continuous=continuous,
                            categorical=categorical,
                            immutables=immutable,
                            target='first_pf')
        mutable_feature = ['LSAT','UGPA','ZFYA','sander_index']
        if args.sens == 'sex':
            sen_feature = "sex_2"
        elif args.sens == 'race':
            sen_feature = "race_White"
        features = mutable_feature        
    print(dataset.df.columns)
    dataset._immutables = immutable_features_col


    if args.ml_model == 'ann':
        ml_model = MLModelCatalog(
            dataset,
            model_type="ann",
            load_online=False,
            backend="pytorch"
        )
        training_params = {"lr": 0.002, "epochs": 99, "batch_size": 1024, "hidden_size": [32]}
        ml_model.train(
            learning_rate=training_params["lr"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            hidden_size=training_params["hidden_size"], 
            force_train = args.forcetrain
        )
    elif args.ml_model == 'lr':
        ml_model = MLModelCatalog(
            dataset,
            model_type="linear",
            load_online=False,
            backend="pytorch"
        )
        training_params = {"lr": 0.01, "epochs": 199, "batch_size": 32, "hidden_size": [64]}
        ml_model.train(
            learning_rate=training_params["lr"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            hidden_size=training_params["hidden_size"],
            force_train = args.forcetrain
        )


    # check accuracy 
    labels = ml_model.predict(dataset.df_test)
    labels = (labels > 0.5 ) * 1
    print(f'accuracy of trained model on train is {np.mean([i==j for i,j in zip(labels, dataset.df_test[dataset.target].values)])}')

    labels = ml_model.predict(dataset.df_train)
    labels = (labels > 0.5 ) * 1
    print(f'accuracy of trained model on test is {np.mean([i==j for i,j in zip(labels, dataset.df_train[dataset.target].values)])}')

    features = mutable_feature

    if args.ar_model == 'balanced':
        import statsmodels.api as sm 
        positive_data = dataset.df[ml_model.predict(dataset.df) > 0.5]
        positive_data.to_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_positive.csv', index=False)
        negative_data = dataset.df[ml_model.predict(dataset.df) <= 0.5]
        negative_data.to_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_negative.csv', index=False)
        if args.balancedata == 1 and args.balancedatastage == 0 and it == args.nrep - 1:        
            import sys 
            sys.exit()
        elif args.balancedata == 1 and args.balancedatastage == 0:
            continue

    print(mutable_feature)
    if args.num_neg == 0:
        negative_instances = dataset.df_train[ml_model.predict(dataset.df_train)<0.5]
    else:
        negative_instances = dataset.df_train[ml_model.predict(dataset.df_train)<0.5]
        negative_instances = negative_instances.sample(args.num_neg)

    positive_data = dataset.df[ml_model.predict(dataset.df) > 0.5]

    print(f'positive sensitive attribute distribution is {positive_data[sen_feature].mean()}')
    print(f'negative sensitive attribute distribution is {negative_instances[sen_feature].mean()}')
    negative_data = negative_instances
    if args.ar_model == 'wachter':
        hyperparams = {"loss_type": "BCE", "binary_cat_features": True}
        ar_model = Wachter(ml_model, hyperparams)
    elif args.ar_model == 'CCHVAE':
        hyperparams = {
            "data_name": dataset.name,
            "n_search_samples": 100,
            "p_norm": 2,
            "step": 1e-2,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 256,2],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 500,
                "lr": 1e-3,
                "batch_size": 32,
            },  
        }
        ar_model = CCHVAE(ml_model, hyperparams)
    elif args.ar_model == 'balanceCCHVAE':
        sen_mask = np.ones(len(ml_model.feature_input_order), dtype=bool)
        print(ml_model.feature_input_order)
        sen_mask[ml_model.feature_input_order.index(sen_feature)] = False
        sen_mask = 1 - sen_mask
        hyperparams = {
            "data_name": dataset.name,
            "n_search_samples": 100,
            "p_norm": 2,
            "step": 1e-2,
            "max_iter": 2000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 256, 2],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 100,
                "lr": 1e-3,
                "batch_size": 32,
                "kl_weight": 1
            },  
            "sen_mask": sen_mask, 
            "gamma": args.gamma
        }
        ar_model = balanceCCHVAE(ml_model, hyperparams)
    elif args.ar_model == 'growing_spheres':
        ar_model = GrowingSpheres(ml_model)
    elif args.ar_model == "face":
        hyperparams = {"mode": 'knn', "fraction": 0.1}
        ar_model = Face(ml_model, hyperparams)
    elif args.ar_model == 'crud':
        hyperparams ={
        "data_name": "crud", 
        "target_class": [0, 1],
        "lambda_param": 0.001,
        "optimizer": "RMSprop",
        "lr": 1e-2,
        "max_iter": 2000,

        "binary_cat_features": True,
        "vae_params": {
            "layers":  [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 256,2],
            "train": True,
            "epochs": 100,
            "lr": 1e-2,
            "batch_size": 32,
        },
        }
        ar_model = CRUD(ml_model, hyperparams)  
    elif args.ar_model == 'balanced':
        if args.dataset == 'syn':
            import pandas as pd 
            # get density estimation for each sensitive attribute 
            from sklearn.neighbors import KernelDensity
            from sklearn.mixture import GaussianMixture 
            kde0 = GaussianMixture(20)
            kde1 = GaussianMixture(20)
            mindata = ml_model.data.df_train[ml_model.data.df_train[sen_feature]==1]
            majdata = ml_model.data.df_train[ml_model.data.df_train[sen_feature]==0]
            mindatapos = mindata[ml_model.predict(mindata) > 0.5][features]
            majdatapos = majdata[ml_model.predict(majdata) > 0.5][features]
            kde0.fit(mindatapos)
            kde1.fit(majdatapos)
            sampled_points = np.empty(shape = (0, len(mutable_feature)))
            sample_size = 100000
            thissample, _ = kde0.sample(sample_size)
            uniform = np.random.uniform(size = sample_size)
            density = np.exp(kde1.score_samples(thissample))
            accepted_samples = thissample[uniform < density]
            sampled_points = np.concatenate((sampled_points, accepted_samples))
            sampled_data = pd.DataFrame(sampled_points)
            sampled_data.columns = mutable_feature
            density = kde1.score_samples(sampled_points) + kde0.score_samples(sampled_points)
            thres = 0.8
            sampled_data = sampled_data[density > np.quantile(density,thres)]
        elif args.balancedata == 0: 
            # fit a density model 
            dep_type = ''.join(['u' if '_1' in i else 'c' for i in mutable_feature])
            #immutables = [i for i in dataset.df.columns if 'race' in i or 'age' in i or 'status' in i or 'sex' in i]
            immutables = [i for i in dataset.df.columns if 'race' in i or 'status' in i or 'sex' in i]
            imu_type = ''.join(['c' if 'age' in i else 'u' for i in immutables])
            density_model = sm.nonparametric.KDEMultivariateConditional(endog=positive_data[features].values, \
                exog=positive_data[immutables].values, dep_type=dep_type, indep_type=imu_type, bw='normal_reference')
            sen_index = np.where([i == sen_feature for i in immutables])[0][0]
            prop_density = 0.5 * 1
            sample_size = 10000
            sampled_points = [np.random.choice(2, size = sample_size) if i == 'u' else np.random.uniform(size = sample_size) for i in dep_type]
            sampled_points = np.array(sampled_points).T.reshape(-1,len(dep_type))
            cf = []
            q = args.bq 
            for i in tqdm(negative_data.index):
                this_data = negative_data.loc[i]
                this_data[mutable_feature]

                import copy 
                origin_immutable = this_data[immutables].values.reshape(1,-1)
                cf_immutable = copy.deepcopy(origin_immutable)
                cf_immutable[0,sen_index] = 1 - cf_immutable[0,sen_index]
                est_density1 = density_model.pdf(endog_predict=sampled_points, \
                    exog_predict = np.repeat(origin_immutable, sample_size,0))
                est_density0 = density_model.pdf(endog_predict=sampled_points, \
                    exog_predict = np.repeat(cf_immutable, sample_size,0))
                rand_sample = np.random.uniform(size=sample_size)
                M = 2
                this_sampled_data = sampled_points[rand_sample < est_density1 * est_density0 / prop_density / M]
                this_density = est_density0 * est_density1
                this_density = this_density[rand_sample < est_density1 * est_density0 / prop_density / M]
                sampled_data_filtered = this_sampled_data[this_density > np.quantile(this_density,q)]
                distance = pairwise_distances(this_data[mutable_feature].values.reshape(1,-1), sampled_data_filtered, metric='euclidean')
                thissample = sampled_data_filtered.copy()
                thissample = pd.DataFrame(thissample, columns=mutable_feature)
                for j in immutables:
                    thissample[j] = this_data[j]
                thissample[dataset.target] = this_data[dataset.target]
                result = ml_model.predict(thissample).reshape(-1)
                # pick the one with lowest distance 
                record_columns = thissample.columns
                for j in np.argsort(distance[0]):
                    if result[j] > 0.5:
                        cf.append(thissample.iloc[j])
                        break
            cf = pd.DataFrame(cf)
            cf.columns = record_columns
            cf.index = negative_data.index
            cf = cf[negative_data.columns]
            cf = cf.loc[negative_instances.index]
        elif args.balancedata == 1 and args.balancedatastage == 1:
            print('cf selection:')
            file_name = f'./balance_data/sampled_data/{args.dataset}_{sen_feature}_rep{it}_sampling.csv'
            sampled_data = pd.read_csv(file_name, index_col=0)  
            cf = []
            sampled_data = sampled_data.round(4)

            if args.dataset == 'german':
                immutable_features = [i + '_1' if i not in continuous else i for i in immutable ]
            else:
                immutable_features = [i + '_1.0' if 'age' not in i and i not in continuous else i for i in immutable ]
            for i in negative_instances.index:
                thisdata = negative_instances.loc[i]
                thisdata = thisdata.round(4)
                this_sampled_data = sampled_data[(sampled_data[immutable_features_col]==thisdata[immutable_features_col]).sum(1)==len(immutable_features)]
                print(this_sampled_data)
                sampled_data_filtered = this_sampled_data[this_sampled_data.density >= np.quantile(this_sampled_data.density,args.bq)]
                distance = pairwise_distances(thisdata[mutable_feature].values.reshape(1,-1), sampled_data_filtered[mutable_feature], metric='euclidean')
                thissample = sampled_data_filtered.copy()[mutable_feature]
                for j in immutable_features_col:
                    thissample[j] = thisdata[j]
                thissample[dataset.target] = thisdata[dataset.target]
                result1 = ml_model.predict(thissample).reshape(-1)
                # pick the one with lowest distance 
                record_columns = thissample.columns
                if_find = 0
                for j in np.argsort(distance[0]):
                    if result1[j] > 0.5:
                        if_find = 1
                        cf.append(thissample.iloc[j])
                        break
                if if_find == 0:
                    cf.append(thissample.iloc[j])
            cf = pd.DataFrame(cf)
            cf.columns = record_columns

            cf.index = negative_instances.index
            cf = cf[negative_instances.columns]        

    if args.ar_model == 'balanced':
        1
    else:
        print('generating counterfactuals...')
        cf = ar_model.get_counterfactuals(negative_instances)
        cf.index = negative_instances.index
        print('generated.')

    cf = cf.fillna(0)
    import copy 
    current_population = copy.deepcopy(dataset.df_train)

    pred = (ml_model.predict(current_population) > 0.5).reshape(-1)
    pred_origin = (ml_model.predict(current_population) > 0.5).reshape(-1)
    predprob = (ml_model.predict(current_population)).reshape(-1)

    p_central, radius  = centralization(dataset.df_train[sen_feature].values, \
        dataset.df_train[mutable_feature].values, 
        )
    print(f'Previous Centralization is {p_central}')
    p_central_pred = centralization_pred(dataset.df_train[sen_feature].values, \
        dataset.df_train[mutable_feature].values, predprob
        )
    print(f'Previous Centralization with Pred is {p_central_pred}')    
    p_atkinson = atkinson(dataset.df_train[sen_feature].values,\
        dataset.df_train[mutable_feature].values, \
        origin_features = dataset.df_train[mutable_feature].values)
    print(f'Previous Atkinson is {p_atkinson}')
    a = dataset.df_train

    p_ap = avg_proximity2(a[sen_feature].values,\
        a[mutable_feature].values, \
            dataset.df_train[mutable_feature].values[pred == 1])
    print(f'Previous Avg Proximity is {p_ap}')

    result = result.append({'nrep':it, 'metric':'centralization_pre', \
        'value': p_central, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)

    result = result.append({'nrep':it, 'metric':'atkinson_pre', \
        'value': p_atkinson, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)

    result = result.append({'nrep':it, 'metric':'avgproxi_pre', \
        'value': p_ap, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)


    cf = encode_constraint(cf, negative_instances, immutable_features_col)
    current_population = copy.deepcopy(dataset.df_train)
    current_population.loc[negative_instances.index,features] = cf[features].values
    current_population = current_population.dropna()

    pred = (ml_model.predict(current_population) > 0.5).reshape(-1)
    predprob = (ml_model.predict(current_population)).reshape(-1)

    c_central, radius = centralization(current_population[sen_feature].values, \
        current_population[features].values, 
        radius
        )
    print(f'Current Centralization is {c_central}')
 
    c_atkinson = atkinson(current_population[sen_feature].values, \
        current_population[features].values, \
        origin_features = dataset.df_train[mutable_feature].values)
    print(f'Current Atkinson is {c_atkinson}')
    a = current_population
    b = dataset.df_train
    c_ap = avg_proximity2(a[sen_feature].values, \
        a[features].values, \
            origin_features =dataset.df_train[mutable_feature].values)
    print(f'Current Avg Proxi Index is {c_ap}')

    inv_rate = invalidation(cf, ml_model)
    print(f'invalidation is {inv_rate}')
    rec_cost = recourse_cost(negative_instances, cf)

    negative_sens = negative_instances[sen_feature]
    rec_fair = fairness_cost(negative_instances, cf, negative_sens)

    vynn = ynn(dataset.df[cf.columns], cf, ml_model, 5)
    close = closeness(cf[mutable_feature], dataset.df[dataset.df[dataset.target]==1][mutable_feature])

    result = result.append({'nrep':it, 'metric':'centralization_after', \
        'value': c_central, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)
    result = result.append({'nrep':it, 'metric':'atkinson_after', \
        'value': c_atkinson, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)
    result = result.append({'nrep':it, 'metric':'avgproxi_after', \
        'value': c_ap, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)

    result = result.append({'nrep':it, 'metric':'inv_rate', \
        'value': inv_rate, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)
    result = result.append({'nrep':it, 'metric':'recourse_cost', \
        'value': rec_cost, 'ml_model': args.ml_model, 'method': args.ar_model, 'dataset': args.dataset}, ignore_index = True)
    result.value = result.value.astype(float)

    result = result.append({'nrep':it, 'metric':'recourse_faircost', \
        'value': rec_fair, 'ml_model': args.ml_model, 'method': args.ar_model}, ignore_index = True) 
    result = result.append({'nrep':it, 'metric':'ynn', \
        'value': vynn, 'ml_model': args.ml_model, 'method': args.ar_model}, ignore_index = True)        
    result = result.append({'nrep':it, 'metric':'close', \
        'value': close, 'ml_model': args.ml_model, 'method': args.ar_model}, ignore_index = True)    
    print(result)
    summean = result.groupby(['metric','ml_model','method']).mean()
    sumse = result.groupby(['metric','ml_model','method']).std()/np.sqrt(nrep)
    resultsum = summean.append(sumse)
    result.to_csv(f'./log/realresult1/{args.dataset}_{args.sens}_{args.ar_model}_{args.ml_model}_{args.nrep0}_{args.nrep}_{args.gamma}.csv')
    resultsum.to_csv(f'./log/realresult1/sum_{args.dataset}_{args.sens}_{args.ar_model}_{args.ml_model}_{args.nrep0}_{args.nrep}_{args.gamma}.csv')