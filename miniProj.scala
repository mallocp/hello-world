import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

class miniProj(spark: SparkSession) {

  val inputText = spark.sparkContext.textFile("C:\\Users\\malloc\\IdeaProjects\\Safari\\testscala5\\src\\Sentiment_Analysis_Dataset10k.csv")
  val sentenceDF:DataFrame = spark.createDataFrame(inputText.map(x=>(x.split(",")(0),x.split(",")(1),x.split(",")(2)))).toDF("id","label","sentence")

  val tokenizer:Tokenizer = new Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")

  val wordsDF:DataFrame = tokenizer.transform(sentenceDF)

  val remover:StopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
  val noStopWordsDF:DataFrame = remover.transform(wordsDF)


  val countVectorizer:CountVectorizer = new CountVectorizer().setInputCol("filteredWords").setOutputCol("features")
  val countVectorizerModel:CountVectorizerModel = countVectorizer.fit(noStopWordsDF)

  val countVectorizerDF:DataFrame = countVectorizerModel.transform(noStopWordsDF)

  val inputData:DataFrame=countVectorizerDF.select("label", "features").withColumn("label", col("label").cast("double"))

  val Array(trainingData, testData) = inputData.randomSplit(Array(0.8, 0.2))

  val lr = new LogisticRegression()

  var lrModel = lr.fit(trainingData)
  var lrCoef =lrModel.coefficients
  var lrInt = lrModel.intercept

  val summary = lrModel.summary
  val bSummary = summary.asInstanceOf[BinaryLogisticRegressionSummary]

  val auc = bSummary.areaUnderROC
  val roc = bSummary.roc



  val training = lrModel.transform(trainingData)
  val test = lrModel.transform(testData)





}
