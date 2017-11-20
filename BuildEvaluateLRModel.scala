
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, IDF,IDFModel, ElementwiseProduct}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.clustering.{LDA, LDAModel, LocalLDAModel}

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.databricks.spark.csv.CsvContext

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.classification.SVMWithSGD

//import org.apache.spark.sql.types._
import org.apache.spark.sql.DataFrame

import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("akka").setLevel(Level.WARN)

val uri:String = "s3://mimiciii/"
val folder:String = "outputs/"

def registerSchema(filename:String, tableName:String,
    tableSchema:StructType,uri:String,sqlContext:SQLContext){

    val table = sqlContext.read.
      format("com.databricks.spark.csv").
      option("header", "true").
      schema(tableSchema).load(uri+filename).cache()
    table.registerTempTable(tableName.toUpperCase)
  }

def registerOutputSchema(filename:String, tableName:String,
    tableSchema:StructType,uri:String,sqlContext:SQLContext){

    val table = sqlContext.read.
      format("com.databricks.spark.csv").
      option("header", "true").
      option("nullValue", "null").
      schema(tableSchema).load(uri+filename).cache()
    table.registerTempTable(tableName.toUpperCase)
  }

val admissionsSchema = StructType(Array(
    StructField("ROW_ID", IntegerType, true),
    StructField("SUBJECT_ID", IntegerType, true),
    StructField("HADM_ID", IntegerType, true),
    StructField("ADMITTIME", TimestampType, true),
    StructField("DISCHTIME", TimestampType, true),
    StructField("DEATHTIME", TimestampType, true),
    StructField("ADMISSION_TYPE", StringType, true),
    StructField("ADMISSION_LOCATION", StringType, true),
    StructField("DISCHARGE_LOCATION", StringType, true),
    StructField("INSURANCE", StringType, true),
    StructField("LANGUAGE", StringType, true),
    StructField("RELIGION", StringType, true),
    StructField("MARITAL_STATUS", StringType, true),
    StructField("ETHNICITY", StringType, true),
    StructField("EDREGTIME", StringType, true),
    StructField("EDOUTTIME", StringType, true),
    StructField("DIAGNOSIS", StringType, true),
    StructField("HOSPITAL_EXPIRE_FLAG", IntegerType, true),
    StructField("HAS_IOEVENTS_DATA", IntegerType, true),
    StructField("HAS_CHARTEVENTS_DATA", IntegerType, true)))
    
registerSchema("ADMISSIONS.csv","admissions",admissionsSchema,uri,sqlContext)

val patientsSchema = StructType(Array(StructField("ROW_ID", IntegerType, true),
    StructField("SUBJECT_ID", IntegerType, true),
    StructField("GENDER", StringType, true),
    StructField("DOB", TimestampType, true),
    StructField("DOD", TimestampType, true),
    StructField("DOD_HOSP", TimestampType, true),
    StructField("DOD_SSN", TimestampType, true),
    StructField("EXPIRE_FLAG", IntegerType, true)))
    
registerSchema("PATIENTS.csv","patients",patientsSchema,uri,sqlContext)

val noteeventsSchema = StructType(Array(StructField("ROW_ID", IntegerType, true),
    StructField("SUBJECT_ID", IntegerType, true),
    StructField("HADM_ID", IntegerType, true),
    StructField("CHARTDATE", DateType, true),
    StructField("CHARTTIME", StringType, true),
    StructField("STORETIME", TimestampType, true),
    StructField("CATEGORY", StringType, true),
    StructField("DESCRIPTION", StringType, true),
    StructField("CGID", IntegerType, true),
    StructField("ISERROR", StringType, true),
    StructField("TEXT", StringType, true)))
    
registerSchema("NOTEEVENTS_PROCESSED.csv","noteevents",noteeventsSchema,uri,sqlContext)

val icustaysSchema = StructType(Array(StructField("ROW_ID", IntegerType, true),
    StructField("SUBJECT_ID", IntegerType, true),
    StructField("HADM_ID", IntegerType, true),
    StructField("ICUSTAY_ID", IntegerType, true),
    StructField("DBSOURCE", StringType, true),
    StructField("FIRST_CAREUNIT", StringType, true),
    StructField("LAST_CAREUNIT", StringType, true),
    StructField("FIRST_WARDID", IntegerType, true),
    StructField("LAST_WARDID", IntegerType, true),
    StructField("INTIME", TimestampType, true),
    StructField("OUTTIME", TimestampType, true),
    StructField("LOS", DoubleType, true)))
    

registerSchema("ICUSTAYS.csv","icustays",icustaysSchema,uri,sqlContext)

val oasisSchema = StructType(Array(StructField("subject_id", IntegerType, true),
    StructField("hadm_id", IntegerType, true),
    StructField("icustay_id", IntegerType, true),
    StructField("ICUSTAY_AGE_GROUP", StringType, true),
    StructField("hospital_expire_flag", IntegerType, true),
    StructField("icustay_expire_flag", IntegerType, true),
    StructField("OASIS", IntegerType, true),
    StructField("OASIS_PROB", DoubleType, true),
    StructField("age", IntegerType, true),
    StructField("age_score", IntegerType, true),
    StructField("preiculos", IntegerType, true),
    StructField("preiculos_score", IntegerType, true),
    StructField("gcs", DoubleType, true),
    StructField("gcs_score", IntegerType, true),
    StructField("heartrate", DoubleType, true),
    StructField("heartrate_score", IntegerType, true),
    StructField("meanbp", DoubleType, true),
    StructField("meanbp_score", IntegerType, true),
    StructField("resprate", DoubleType, true),
    StructField("resprate_score", IntegerType, true),
    StructField("temp", DoubleType, true),
    StructField("temp_score", IntegerType, true),
    StructField("urineoutput", DoubleType, true),
    StructField("UrineOutput_score", IntegerType, true),
    StructField("mechvent", IntegerType, true),
    StructField("mechvent_score", IntegerType, true),
    StructField("electivesurgery", IntegerType, true),
    StructField("electivesurgery_score", IntegerType, true)))
    
registerOutputSchema("OASIS.csv","oasis",oasisSchema,uri+folder,sqlContext) 

val sapsiiSchema = StructType(Array(StructField("subject_id", IntegerType, true),
    StructField("hadm_id", IntegerType, true),
    StructField("icustay_id", IntegerType, true),
    StructField("SAPSII", IntegerType, true),
    StructField("SAPSII_PROB", DoubleType, true),
    StructField("age_score", IntegerType, true),
    StructField("hr_score", IntegerType, true),
    StructField("sysbp_score", IntegerType, true),
    StructField("temp_score", IntegerType, true),
    StructField("PaO2FiO2_score", IntegerType, true),
    StructField("uo_score", IntegerType, true),
    StructField("bun_score", IntegerType, true),
    StructField("wbc_score", IntegerType, true),
    StructField("potassium_score", IntegerType, true),
    StructField("sodium_score", IntegerType, true),
    StructField("bicarbonate_score", IntegerType, true),
    StructField("bilirubin_score", IntegerType, true),
    StructField("gcs_score", IntegerType, true),
    StructField("comorbidity_score", IntegerType, true),
    StructField("UrineOutput_score", IntegerType, true),
    StructField("admissiontype_score", IntegerType, true)))    

registerOutputSchema("SAPSII.csv","sapsii",sapsiiSchema,uri+folder,sqlContext) 

val sofaSchema = StructType(Array(StructField("subject_id", IntegerType, true),
    StructField("hadm_id", IntegerType, true),
    StructField("icustay_id", IntegerType, true),
    StructField("SOFA", IntegerType, true),
    StructField("respiration", IntegerType, true),
    StructField("coagulation", IntegerType, true),
    StructField("liver", IntegerType, true),
    StructField("cardiovascular", IntegerType, true),
    StructField("cns", IntegerType, true),
    StructField("renal", IntegerType, true)))

registerOutputSchema("SOFA.csv","sofa",sofaSchema,uri+folder,sqlContext) 

val kaggleSchema = StructType(Array(StructField("subject_id", IntegerType, true)))

registerSchema("icu_mortality_test_patients.csv", "kaggle_patients", kaggleSchema, uri, sqlContext)

val kaggle = sqlContext.sql("""select * from kaggle_patients""")


// filter out kaggle patient id 
val filteredPatients = sqlContext.sql("""
    SELECT pat.subject_id
    FROM patients pat  
    LEFT JOIN kaggle_patients kag 
    ON  pat.subject_id = kag.subject_id
    WHERE kag.subject_id IS NULL 
 """)
// create labels for each train, test and kaggle dataset
val trainer: (Int => Int) = (arg:Int) => 0
val sqlTrainer = udf(trainer)
val tester: (Int => Int) = (arg:Int) => 1
val sqlTester = udf(tester)
val kaggler: (Int => Int) = (arg:Int) => 2
val sqlKaggler = udf(kaggler)


val splits = filteredPatients.randomSplit(Array(0.7, 0.3), seed = 8803L)  //Seed Set for Reproducibility
val train_patients = splits(0).select("subject_id").withColumn("test",sqlTrainer(col("subject_id")))
val test_patients = splits(1).select("subject_id").withColumn("test",sqlTester(col("subject_id")))
val kaggle_pat = kaggle.select("subject_id").withColumn("test", sqlKaggler(col("subject_id")))
val patient_groups = train_patients.unionAll(kaggle_pat).unionAll(test_patients).sort("subject_id").cache()

//patient_groups.count() // 46520

patient_groups.registerTempTable("patient_groups")


val firstDayScores = sqlContext.sql("""
with subset as (
    SELECT subject_id, hadm_id, min(intime) as firstin FROM icustays GROUP BY subject_id, hadm_id)

SELECT ad.HOSPITAL_EXPIRE_FLAG, ad.subject_id,ad.hadm_id, oasis.OASIS,sapsii.SAPSII,sofa.SOFA,pat.GENDER, 
    ((unix_timestamp(ad.admittime)-unix_timestamp(pat.DOB))/(86400*365)) as age,
     ((unix_timestamp(ad.dischtime)-unix_timestamp(ad.admittime))/(86400)) as los 
FROM admissions ad
LEFT JOIN subset s
ON s.subject_id = ad.subject_id
AND s.hadm_id = ad.hadm_id
AND ad.has_chartevents_data = 1
--AND lower(ad.diagnosis) NOT LIKE '%organ donor%'
LEFT JOIN icustays icu
ON icu.subject_id = s.subject_id
AND icu.hadm_id = s.hadm_id
AND icu.intime = s.firstin
AND icu.intime < ad.admittime + interval '1' day
LEFT JOIN oasis
ON icu.subject_id = oasis.subject_id
AND icu.hadm_id = oasis.hadm_id
AND icu.icustay_id = oasis.icustay_id
LEFT JOIN sapsii
ON icu.subject_id = sapsii.subject_id
AND icu.hadm_id = sapsii.hadm_id
AND icu.icustay_id = sapsii.icustay_id
LEFT JOIN sofa
ON icu.subject_id = sofa.subject_id
AND icu.hadm_id = sofa.hadm_id
AND icu.icustay_id = sofa.icustay_id
JOIN patients pat
ON ad.subject_id = pat.subject_id
--AND ((unix_timestamp(ad.admittime)-unix_timestamp(pat.DOB))/(86400*365)) > 15
""").cache()

firstDayScores.registerTempTable("firstDayScore")

// contain subject_id, hadm_id, SEX, oasis_num, sapsii_num, sofa_num, age, hosp_death
val dataDf = sqlContext.sql("""
SELECT firstDayScore.subject_id,
       firstDayScore.hadm_id,
        case WHEN GENDER = 'M' then 1 else 0 end as SEX,
        coalesce(OASIS,0) as oasis_num,
        coalesce(SAPSII,0) as sapsii_num,
        coalesce(SOFA,0) as sofa_num,
        age,
        HOSPITAL_EXPIRE_FLAG as hosp_death
        --test
FROM firstDayScore 
JOIN patient_groups
ON patient_groups.subject_id = firstDayScore.subject_id 
""")

//dataDf.count() // 58976
//dataDf.show()
dataDf.registerTempTable("data")

//read in and extract StopWords and add lab unit need to remove also
var stopwords1 = sc.textFile(uri+"onixstopwords").collect.toArray
stopwords1 = stopwords1 ++ Array("ml","dl","mg","kg","pm")
val stopwords = stopwords1.toSet
val broadcaststopwords = sc.broadcast(stopwords)

// extract the note events, remove stopword, and also the words with length less than 2
sqlContext.udf.register("cleanString", (s: String) => {
    val words = s.split(" ")
    val filtered_words = words.map(_.toLowerCase()).filter(w => !broadcaststopwords.value.contains(w))
    var result = ""
    if (filtered_words.size > 2){
        result = filtered_words.reduceLeft((x,y)=>x+" "+y)
    }
    result
})
sqlContext.udf.register("combindString",(arrayCol: Seq[String]) => arrayCol.mkString(" "))


//build tf-idf, lda model only using the notes for train dataset
// remove the error notes, notes without strorage time, discharge 
val trainDocuments = sqlContext.sql("""
SELECT noteevents.subject_id,noteevents.hadm_id,combindString(collect_list(cleanString(regexp_replace(regexp_replace(text,"[^a-zA-Z\\s]",' '),"NEWLINE",' ')))) as Status
FROM noteevents 
JOIN patient_groups
ON patient_groups.subject_id = noteevents.subject_id
-- only select the train dataset
AND patient_groups.test = 0
WHERE storetime is not null
AND iserror = '' 
AND category != 'Discharge summary'
GROUP BY noteevents.subject_id, noteevents.hadm_id
""")

val regexTokenizer = new RegexTokenizer().
    setInputCol("Status").
    setOutputCol("Words").
    setPattern("\\W")

val trainRegexTokenized = regexTokenizer.transform(trainDocuments)

val remover = new StopWordsRemover().
    setInputCol("Words").
    setOutputCol("Filtered")

val trainCleanedTokenized = remover.transform(trainRegexTokenized)
trainCleanedTokenized.cache()

val cvModel = CountVectorizerModel.load(uri+"models/cvModel")
val idfModel = IDFModel.load(uri+"models/idfModel")
val ldaModel = LocalLDAModel.load(uri+"models/ldaModel")


val trainCountTokenized = cvModel.transform(trainCleanedTokenized)

val trainIdfTokenized = idfModel.transform(trainCountTokenized)
trainIdfTokenized.cache()
// trainIdfTokenized.show()

 val trainLdaTokenized = ldaModel.transform(trainIdfTokenized)
trainLdaTokenized.registerTempTable("trainLdaTokenized")
// trainLdaTokenized.show()

val trainAllData = sqlContext.sql("""
SELECT trainLdaTokenized.TopicVector, data.* 
FROM trainLdaTokenized 
JOIN data
ON data.subject_id = trainLdaTokenized.subject_id
AND data.hadm_id = trainLdaTokenized.hadm_id
""")

val assemblerAllFeatures = new VectorAssembler().
setInputCols(Array("TopicVector", 
    "SEX",
    "oasis_num",
    "sapsii_num",
    "sofa_num",
    "age")).
setOutputCol("ModelFeatures")

val assemblerStructuredFeatures = new VectorAssembler().
setInputCols(Array(
    "SEX",
    "oasis_num",
    "sapsii_num",
    "sofa_num",
    "age")).
setOutputCol("ModelFeatures")

val assemblerNotes = new VectorAssembler().
setInputCols(Array("TopicVector")).
setOutputCol("ModelFeatures")

val trainAllFeatures = assemblerAllFeatures.transform(trainAllData)
val trainDataNotes = assemblerNotes.transform(trainAllData)
val trainDataStructuredFeatures = assemblerStructuredFeatures.transform(trainAllData)

val toDouble = udf[Double, String]( _.toDouble)

val trainAllFeaturesLabeled = trainAllFeatures.withColumn("Label", toDouble(trainAllFeatures("hosp_death")))
val trainDataNotesLabeled = trainDataNotes.withColumn("Label", toDouble(trainDataNotes("hosp_death")))
val trainDataStructuredFeaturesLabeled = trainDataStructuredFeatures.withColumn("Label", toDouble(trainDataStructuredFeatures("hosp_death")))

//train logisticRegression model
val lr = new LogisticRegression().
    setMaxIter(10).
    setRegParam(0.0).
    setFeaturesCol("ModelFeatures").
    setLabelCol("Label").
    setPredictionCol("ModelPrediction").
    setProbabilityCol("ModelProbability") 

// Fit the model

val lrAllFeatures = lr.fit(trainAllFeaturesLabeled)
val lrNotes = lr.fit(trainDataNotesLabeled)
val lrStructuredFeatures = lr.fit(trainDataStructuredFeaturesLabeled)

// calculate the auc for test data using linear regression model for all features
// prepare test data for predict
val testQuery = s"""
SELECT noteevents.subject_id,noteevents.hadm_id,combindString(collect_list(cleanString(regexp_replace(regexp_replace(text,"[^a-zA-Z\\s]",' '),"NEWLINE",' ')))) as Status
FROM noteevents 
JOIN patient_groups
ON patient_groups.subject_id = noteevents.subject_id
-- only select the train dataset
AND patient_groups.test = 1
WHERE storetime is not null
AND iserror = '' 
AND category != 'Discharge summary'
GROUP BY noteevents.subject_id, noteevents.hadm_id
"""

val testDocuments = sqlContext.sql(testQuery).cache()
val testRegexTokenized = regexTokenizer.transform(testDocuments)
val testCleanedTokenized = remover.transform(testRegexTokenized)
val testCountTokenized = cvModel.transform(testCleanedTokenized)
val testIdfTokenized = idfModel.transform(testCountTokenized)
val testLdaTokenized = ldaModel.transform(testIdfTokenized)
testLdaTokenized.registerTempTable("testLdaTokenized")

val testAllData = sqlContext.sql("""
 SELECT testLdaTokenized.TopicVector, data.* 
 FROM testLdaTokenized
 JOIN data
 ON data.subject_id = testLdaTokenized.subject_id
 AND data.hadm_id = testLdaTokenized.hadm_id
 """)

 val testAllFeatures = assemblerAllFeatures.transform(testAllData)

 val testAllFeatures2 = testAllFeatures.withColumn("Label", toDouble(testAllFeatures("hosp_death")))

 val testResultsAllFeatures = lrAllFeatures.transform(testAllFeatures2)

 val testScoreAndLabel = testResultsAllFeatures.
 select("Label","ModelProbability").
 map{ case Row(l:Double,p:Vector) => (p(1),l) }
 val testMetrics = new BinaryClassificationMetrics(testScoreAndLabel)
 val testAuROC = testMetrics.areaUnderROC()

// calculate the auc for kaggle data using linear regression model for all features
val kaggleQuery = s"""
SELECT noteevents.subject_id,noteevents.hadm_id,combindString(collect_list(cleanString(regexp_replace(regexp_replace(text,"[^a-zA-Z\\s]",' '),"NEWLINE",' ')))) as Status
FROM noteevents 
JOIN patient_groups
ON patient_groups.subject_id = noteevents.subject_id
-- only select the train dataset
AND patient_groups.test = 2
WHERE storetime is not null
AND iserror = '' 
AND category != 'Discharge summary'
GROUP BY noteevents.subject_id, noteevents.hadm_id
"""

val kaggleDocuments = sqlContext.sql(kaggleQuery).cache()
val kaggleRegexTokenized = regexTokenizer.transform(kaggleDocuments)
val kaggleCleanedTokenized = remover.transform(kaggleRegexTokenized)
val kaggleCountTokenized = cvModel.transform(kaggleCleanedTokenized)
val kaggleIdfTokenized = idfModel.transform(kaggleCountTokenized)
val kaggleLdaTokenized = ldaModel.transform(kaggleIdfTokenized)
kaggleLdaTokenized.registerTempTable("kaggleLdaTokenized")

val kaggleAllData = sqlContext.sql("""
SELECT kaggleLdaTokenized.TopicVector, data.* 
FROM kaggleLdaTokenized 
JOIN data
ON data.subject_id = kaggleLdaTokenized.subject_id
AND data.hadm_id = kaggleLdaTokenized.hadm_id
""")


 val kaggleAllFeatures = assemblerAllFeatures.transform(kaggleAllData)

 val kaggleAllFeatures2 = kaggleAllFeatures.withColumn("Label", toDouble(kaggleAllFeatures("hosp_death")))

 val kaggleResultsAllFeatures = lrAllFeatures.transform(kaggleAllFeatures2)

 val kaggleScoreAndLabel = kaggleResultsAllFeatures.
     select("Label","ModelProbability").
     map{ case Row(l:Double,p:Vector) => (p(1),l) }
 val kaggleMetrics = new BinaryClassificationMetrics(kaggleScoreAndLabel)
 val kaggleAuROC = kaggleMetrics.areaUnderROC()


val kaggleProb = kaggleResultsAllFeatures.
            select("subject_id","ModelProbability").
            map{ case Row(id:Int,p:Vector) => (id, p(1))}.
            map{case(k:Int, v:Double ) => Array(k, v).mkString(",") }


val file = uri + folder + "kaggleResult.csv"

kaggleProb.saveAsTextFile(file)

//function to calculate the auc for logistic regression model with all features, notes only and structured data only 

def calculate_lr_auc(documents: DataFrame): (Double,Double, Double) = {

    val regexTokenized = regexTokenizer.transform(documents)
    val cleanedTokenized = remover.transform(regexTokenized)
    val countTokenized = cvModel.transform(cleanedTokenized)
    val idfTokenized = idfModel.transform(countTokenized)
    val ldaTokenized = ldaModel.transform(idfTokenized)
    ldaTokenized.registerTempTable("ldaTokenized")

    val allData = sqlContext.sql("""
    SELECT ldaTokenized.TopicVector, data.* 
    FROM ldaTokenized 
    JOIN data
    ON data.subject_id = ldaTokenized.subject_id
    AND data.hadm_id = ldaTokenized.hadm_id
    """)
//calculate auc from all features
    val allFeatures = assemblerAllFeatures.transform(allData)
    val allFeatures2 = allFeatures.withColumn("Label", toDouble(allFeatures("hosp_death")))
    val resultsAllFeatures = lrAllFeatures.transform(allFeatures2)

    val allScoreAndLabel = resultsAllFeatures.
    select("Label","ModelProbability").
    map{ case Row(l:Double,p:Vector) => (p(1),l) }
    val allMetrics = new BinaryClassificationMetrics(allScoreAndLabel)
    val auROCAllFeature = allMetrics.areaUnderROC()
// calculate auc from note features
    val notesFeatures = assemblerNotes.transform(allData)
    val notesFeatures2 = notesFeatures.withColumn("Label", toDouble(notesFeatures("hosp_death")))
    val resultsNotesFeatures = lrNotes.transform(notesFeatures2)

    val notesScoreAndLabel = resultsNotesFeatures.
    select("Label","ModelProbability").
    map{ case Row(l:Double,p:Vector) => (p(1),l) }
    val notesMetrics = new BinaryClassificationMetrics(notesScoreAndLabel)
    val auROCNotes = notesMetrics.areaUnderROC()
// calculate auc from structured features
    val structuredFeatures = assemblerStructuredFeatures.transform(allData)
    val structuredFeatures2 = structuredFeatures.withColumn("Label", toDouble(structuredFeatures("hosp_death")))
    val resultsStructuredFeatures = lrStructuredFeatures.transform(structuredFeatures2)

    val structuredScoreAndLabel = resultsStructuredFeatures.
    select("Label","ModelProbability").
    map{ case Row(l:Double,p:Vector) => (p(1),l) }
    val structuredMetrics = new BinaryClassificationMetrics(structuredScoreAndLabel)
    val auROCStructured = structuredMetrics.areaUnderROC()
    
    (auROCAllFeature, auROCNotes, auROCStructured)
}

//auc for all features, notes, structured features
println("auc for test dataset of 1) all features, 2) only notes data and 3) only structured data on linear regression model: ")
val test_auc =calculate_lr_auc(testDocuments)
println("auc for kaggle dataset of 1) all features, 2) only notes data and 3) only structured data on linear regression model: ")
val kaggle_auc = calculate_lr_auc(kaggleDocuments)
