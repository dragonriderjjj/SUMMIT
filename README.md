# Learning from Fragmentary Multivariate Time Series Data with Scalable Numerical Embedding

This is the guideline for reproducing experiment results in the paper. You have to follow the procedure below before starting the experiment.

- Preparing data for experiment
    -- We only provide preprocessed (summarized) PhysioNet2012 (P12) dataset in the `./data/P12`. (We also provide the full pipeline for preprocessing data.)
    -- Anonymous Hospital Hepatocellular Carcinoma (HCC) dataset is protected by the IRB.
    -- MIMIC-III (MI3) can be accessed before getting the license. You can learn more from detail <https://physionet.org/content/mimiciii/>.
- Environment setup
- Starting experiment

## Environment Setup
All experiments in the paper are conducted on the platform with:
 - GPU: RTX 3060 with 12GB VRAM
 - CUDA Version: 11.4
 - gcc Version: 7.5.0
 - python Version: 3.8.8
 - pytorch Version: 1.13.1
 - scikit-learn Version: 1.1.1
 - xgboost Version: 1.7.5
 - (for data preparing) tensorflow Version: 2.13.0
 - (for data preparing) tensorflow_datasets Version: 4.9.2

You can follow the commands below to construct the appropriate environment with ***conda***.
```bash
# construct a virtual environment
conda create --name TranSCANE python=3.8.8
conda activate TranSCANE
# install related package
pip install -r requirements.txt
```


## Preparing for Data
There are three datasets in this experiment:
 - PhysioNet2012 (P12) Dataset: This is an open dataset from the keggle competition. We collect these dataset by [Horn's work](https://github.com/ExpectationMax/medical_ts_datasets).
 - MIMIC-III (MI3) Dataset: This is also an open dataset. However, you have to apply for a license before accessing these data.
 - Anonymous Hospital Hepatocellular Carcinoma (HCC) Dataset: This dataset is private, we can't disclose any information about it.

##### PhysioNet2012 (P12) Dataset
Although we have provide the P12 data in the attached files, you can access the data by yourself by following the steps below.
※ Due to the round-off difference when summarizing, the preprocessed data may be different with the data we provided.
```bash
# The commanc is optional. If you have cloned medical_ts_datasets, you do not need to clone it again.
git clone https://github.com/ExpectationMax/medical_ts_datasets.git
# The command below will collect the training, validation, and testing data from medical_ts_datasets (https://github.com/ExpectationMax/medical_ts_datasets). Then, it will apply the summarization strategy on these data.
bash preparing_summarizing_P12.sh
```

##### MIMIC-III (MI3) Dataset
You need to get access from <https://physionet.org/content/mimiciii/>. The command below assume you can get the MIMIC-III data from the `medical_ts_dataset` package.
※ More detail for `medical_ts_dataset`, check <https://github.com/ExpectationMax/medical_ts_datasets>.
```bash
# The commanc is optional. If you have cloned medical_ts_datasets, you do not need to clone it again.
git clone https://github.com/ExpectationMax/medical_ts_datasets.git
# The command below will collect the training, validation, and testing data from medical_ts_datasets. Then, it will apply the summarization strategy on these data.
bash preparing_summarizing_MI3.sh
```
## Starting Experiment

Before starting the following experiments, make sure your platform is consistent with us. For different GPUs, the training result may vary due to round-off difference. We also provide our trained model's best checkpoint.

#### Overall Result
```bash
# Use this command to reproduce the overall result
bash overall_result.sh $mode $dataset

# This is an example command for testing on P12 dataset.
bash overall_result.sh test P12
```
`$dataset` can be `P12`, `MI3`, or `HCC`.

- `P12` is available.
- `MI3` can be used after accessing the license.
- `HCC` is protected by the IRB, so it is not available.

There are two `$mode` for this script: `train_test` and `test`. 

- `train_test` mode will train all models and use the trained models to test on the target dataset's test set.
- `test` mode will use our provided model's checkpoint to test on the target dataset's test set. The checkpoints from the `train` mode or `train_test` mode are identical to them used in this mode.


#### Predicted Probability with Different Imputations
Because the original sample is from HCC dataset which is unavailable, we randomly select a sample from P12 to show the imputation robustness of TranSCANE.
This sample's "ALP" is unmissing at the first, the 20-th, and the 22-th summarization times. We will try three different value (1, 100, 10000) to impute missing "ALP". You can use the following command to get the results.
```bash
# Use this command to do the imputation robustness experiment with all moels
bash impute_robust.sh $impute_value_1 $impute_value_2 $impute_value_3

# This is an example command for impute missing AFP with 1, 100, and 10000 on the target sample.
bash impute_robust.sh 1 100 10000
```
You can try different number to impute the missing AFP. The experiment result will show that out model, TranSCANE, remains immune to noisy imputations and faithfully predicts the probability based on the observed values.
#### Performance of Models on The Samples with Shuffled Timestamp
```bash
# Use this command to reproduce the result with timestamp-shuffled samples
bash shuffled_timestamp.sh $dataset

# This is an example command for testing on P12 dataset.
bash shuffled_timestamp.sh P12
```
`$dataset`  here can be `P12`, `MI3`, or `HCC`.

We provide our trained models in the *best model folder* (`./exp_output/model_ckt_${dataset}/${model}/`). You can use the model you trained (in folder
`./exp_output/overall_result/${experiment_date}_${dataset}_${model}/testing_result/ckt.pth`)
by moving them to the *best model folder*.
#### Attention Weight Visualization
We provide an simulated sample to visualize the attention map. You can plot the attention map by following the command below.
```bash
# You can use this command to plot the simulate sample's attention map.
python3 viz_attn.py -c "./config/attn_viz/tesne.json" -r "exp_output/model_ckt_HCC/tesne/ckt.pth" -p ${path_to_save_the_visualization_attention_map}
```
# SUMMIT
