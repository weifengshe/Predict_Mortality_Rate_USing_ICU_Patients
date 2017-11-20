# Project: Mortality Prediction of ICU patients from MIMICIII Dataset using Latent Dirichlet Allocation Model


## Introduction

This if the final project of CSE8803 Big Data For Health Informatics. This folder contains the scripts only and the scripts were ran in spark-shell environment on the [Elastic Map-Reduce(EMR) cluster](https://aws.amazon.com/elasticmapreduce/) of [Amazon Web Services (AWS)](http://aws.amazon.com/). The freely avaible [Medical Information Mart for Intensive Care (MIMIC) III data](http://mimic.physionet.org/) is uploaded and stored in the AWS S3 Bucket. 

## load data to s3 bucket

all the data should be loaded to s3 bucket. The following scripts assume a uri set ("s3://mimiciii/") to the S3 bucket location of of your MIMIC III files (uncompresssed) and the folder ("outputs/") you want the generated files to be written to.  

## AWS Elastic Map-Reduce set up

Cluster was set by using the web-dashboard. the EMR cluster has setting at: emr-4.5.0, Spark 1.6.1 on Hadoop 2.7.2, Hive 1.0.0, Pig 0.14.0, Hue 3.7.1,  YARN with Ganglia 3.7.2 and Zeppelin-Sandbox 0.5.6. The cluster contains 1 r3.xlarge master node and 3 r3.xlarge worker nodes. After the cluster started. 
Setting the Security groups for EMR-master cluster: choose cluster -> actions -> Edit inbound rules ->  
Add rule : Type/Custom TCP Rule, Port Range/8890, Source/My IP
Type/SSH, Port Rangle/22, Source/My IP and save.


## Instruction for running the code

Download MIMICIII data and uncompress them to .csv file. 

Process the NOTEEVENTS.csv file with the 
```bash
python processnotes.py
```
Upload all the .CSV file to amazon S3 bucket. 

Upload the scripts file into your cluster: 

```bash
scp -i <-your-keypair.pem-> location/to/repo/*.scala hadoop@ec2-xx-xx-xx-xx.computer-x.amazonaws.com:~/
```

login your cluster:

```bash
ssh -i <-your-keypair.pem-> hadoop@ec2-xx-xx-xx-xx.computer-x.amazonaws.com
```

Once you successfully login, you should run the command as suggest to update the system. 
```bash
sudo yum updata
```
after sucessfully updated, start the spark shell environment

```bash
SPARK_REPL_OPTS="-XX:MaxPermSize=10g" spark-shell --packages com.databricks:spark-csv_2.10:1.4.0 --conf spark.driver.maxResultSize=10g --conf spark.driver.memory=10g --conf spark.executor.memory=15g
```

once the spark-shell started, the script is loaded by the :load command:
this script is for serverity scores calculation and save the corresponding file in s3 bucket. 
```scala
:load CalculateServerityScores.scala

```
This script is for building the TF-IDF model, LDA model and save them in s3 bucket. 
```scala
:load CalculateServerityScores.scala

```

This script is for building the svm model and evaluate its performance on test dataset. 
```scala
:load BuildEvaluateSVMModel.scala

```

This script is for building the linear regression model and evaluate its performance on test dataset and kaggle dataset, also save the result kaggle file in s3 bucket. 
```scala
:load BuildEvaluateLRModel.scala

```
