from cde.density_estimator import KernelMixtureNetwork, MixtureDensityNetwork#, NormalizingFlowEstimator
import pandas as pd
import numpy as np 
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='compas', help='dataset to use')
parser.add_argument('--sens', type=str, default='race', help='sen feature')
parser.add_argument('--nrep', type=int, default=5, help='number of rep')
parser.add_argument('--dtype', type=str, default='gmix', help='density models')
parser.add_argument('--nepochs', type=int, default=5, help='number of epochs')

args = parser.parse_args()
if args.sens == 'sex':
    sen_feature = 'sex_Male_1.0'
if args.sens == 'age':
    sen_feature = 'age_1'
sens_feature = sen_feature
dataset = args.dataset

for it in range(args.nrep):
    if args.dataset == 'give_me_some_credit':
        categorical = ['age']
        sen_feature = "age_1"
        sens_feature = "age_1"
        immutable = ['age_1']
        immutable_features_col = ["age_1"]
        continuous = ['RevolvingUtilizationOfUnsecuredLines',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
        positive_data = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_positive.csv')
        negative_data = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_negative.csv')
        mutable = mutable_feature = ['RevolvingUtilizationOfUnsecuredLines',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
        sens_id = immutable.index(sens_feature)       
    elif args.dataset == 'law':
        categorical = ['race','sex','region_first']
        
        immutable = ["sex_2", "race_White", 'region_first_1']
        immutable_features_col = ['sex_2', 'race_White', 'region_first_1']
        continuous = ['LSAT','UGPA','ZFYA','sander_index']
        mutable = mutable_feature = ['LSAT','UGPA','ZFYA','sander_index']        
        positive_data = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_positive.csv')
        negative_data = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_{args.sens}_rep{it}_negative.csv')
        if args.sens == 'sex':
            sen_feature = "sex_2"
            sens_feature = "sex_2"
        elif args.sens == 'race':
            sen_feature = "race_White"        
            sens_feature = "race_White"        
        sens_id = immutable.index(sens_feature)  
    elif dataset == 'syn':
        immutable = ['sex_1']
        mutable = ['x1','x2']
        all_data = pd.read_csv('./balance_data/syn_train.csv', index_col = 0)
        negative_data = pd.read_csv('./balance_data/syn_negative_instances.csv', index_col = 0)
        positive_data = all_data[~all_data.index.isin(negative_data.index)]

        import matplotlib.pyplot as plt 
        _, a = density_model.sample(np.ones(1000))
        _, b = density_model.sample(np.zeros(1000))
        plt.scatter(a[:,0],a[:,1])
        plt.scatter(b[:,0],b[:,1])
        plt.scatter(np.array(sampled_points)[:,0], np.array(sampled_points)[:,1])
        plt.show()

    density_model = MixtureDensityNetwork(f'mixd{it}', len(immutable), len(mutable), n_centers = 50, hidden_sizes = (16,16), n_training_epochs = args.nepochs)
    
    all_data = positive_data.append(negative_data)
    density_model.fit(positive_data[immutable].values, positive_data[mutable].values)

    file_name = f'./balance_data/sampled_data/{args.dataset}_{sen_feature}_rep{it}_sampling.csv'

    def rej_sampling_syn(negative_data):
        sampled_points = np.empty(shape = (0, len(mutable)))
        density = []
        while sampled_points.shape[0] < 1000:
            print(sampled_points.shape)
            sample_size = 200000
            _, thissample = density_model.sample(np.ones(sample_size))
            uniform = np.random.uniform(size = sample_size)
            accepted_samples = thissample[uniform < density_model.pdf(np.zeros(sample_size), thissample)]
            sampled_points = np.concatenate((sampled_points, accepted_samples))
        samples_density = density_model.pdf(np.zeros(sampled_points.shape[0]),sampled_points) * \
            density_model.pdf(np.ones(sampled_points.shape[0]),sampled_points)
        sampled_data = pd.DataFrame(sampled_points)
        sampled_data.columns = mutable 
        sampled_data['density'] = samples_density 
        sampled_data.to_csv()
        return sampled_data

    def proper_round(a):
        '''
        given any real number 'a' returns an integer closest to 'a'
        '''
        a_ceil = np.ceil(a)
        a_floor = np.floor(a)
        if np.abs(a_ceil - a) < np.abs(a_floor - a):
            return int(a_ceil)
        else:
            return int(a_floor)

    import copy 
    def rej_sampling_real(negative_data, density_model):
        # sampling high density samples for each immutable_features
        sampled_points = np.empty(shape = (0, len(mutable)+len(immutable)))
        density = []
        num_sampled = 1000 
        sampled_density = np.array([])
        print(f'number of negative data is {negative_data.shape[0]}')
        for i in tqdm(range(negative_data.shape[0])):
            accepted_samples = np.empty((0, len(mutable_feature)))
            samples_density = np.array([])
            while len(accepted_samples) < num_sampled:
                thissample_neg = negative_data.iloc[i,:]
                sample_size = 50000
                immu_feature = thissample_neg[immutable].values.reshape(1,-1).repeat(sample_size,0)
                immu_feature_ori = copy.deepcopy(immu_feature)
                _, thissample = density_model.sample(immu_feature)
                uniform = np.random.uniform(size = sample_size)
                density1 = density_model.pdf(immu_feature, thissample)
                immu_feature[:,sens_id] = 1 - immu_feature[:,sens_id]
                density2 = density_model.pdf(immu_feature, thissample)
                M = 20
                this_accepted_samples = thissample[np.logical_and(uniform < density2 / M, density1 > 0, density2 > 0)]
                accepted_samples = np.concatenate((accepted_samples, this_accepted_samples), 0)
                accepted_samples = np.array(accepted_samples)
                this_samples_density = (density2 * density1)[uniform < density2 / M]
                samples_density = np.append(samples_density, this_samples_density)
            accepted_samples = np.concatenate((accepted_samples, immu_feature_ori[:accepted_samples.shape[0],:]),1)
            sampled_points = np.concatenate((sampled_points, accepted_samples[:num_sampled,]))
            samples_density = samples_density[:num_sampled]
            sampled_density = np.append(sampled_density, samples_density)
        sampled_data = pd.DataFrame(sampled_points)
        sampled_data.columns = mutable + immutable
        sampled_data['density'] = sampled_density 
        sampled_data.to_csv(file_name)
        return sampled_data

    print(f'number of negative data is {negative_data.shape[0]}')
    rej_sampling_real(negative_data, density_model)