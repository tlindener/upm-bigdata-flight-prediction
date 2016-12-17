package es.upm
/**
  * @author Tobias Lindener
  */


import java.io.BufferedOutputStream

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}



object SparkPredict {
  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder()
      .appName("UPM Big Data Group 10")
      .getOrCreate()



    val pathTrainingData = args(0)
    def prepareData (path: String) : DataFrame = {
      val data = sparkSession.read.option("header","true").
        csv(path).drop("ActualElapsedTime","AirTime","TaxiIn","Diverted","ArrTime", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

      data.select( data("Year").cast(IntegerType).as("Year"),
        data("Month").cast(IntegerType),
        data("DayofMonth").cast(IntegerType).as("DayOfMonth"),
        data("DayofWeek").cast(IntegerType).as("DayOfWeek"),
        data("DepTime").cast(IntegerType).as("DepTime"),
        data("CRSDepTime").cast(IntegerType).as("CRSDepTime"),
        data("CRSArrTime").cast(IntegerType).as("CRSArrTime"),
        data("FlightNum").cast(IntegerType).as("FlightNum"),
        data("CRSElapsedTime").cast(IntegerType).as("CRSElapsedTime"),
        data("ArrDelay").cast(DoubleType).as("label"),
        data("DepDelay").cast(IntegerType).as("DepDelay"),
        data("Distance").cast(IntegerType).as("Distance"),
        data("TaxiOut").cast(IntegerType).as("TaxiOut"),
        data("UniqueCarrier"),
        data("TailNum"),
        data("Origin"),
        data("Dest").as("Destination")
      ).na.drop()

    }

    val trainingData =  prepareData(pathTrainingData)

    val monthIndexer = new OneHotEncoder().setInputCol("Month").setOutputCol("MonthCat")
    val dayofMonthIndexer = new OneHotEncoder().setInputCol("DayOfMonth").setOutputCol("DayOfMonthCat")
    val dayOfWeekIndexer = new OneHotEncoder().setInputCol("DayOfWeek").setOutputCol("DayOfWeekCat")
    val uniqueCarrierIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierCat")


    //assemble raw feature
    val assembler = new VectorAssembler()
      .setInputCols(Array("MonthCat", "DayOfMonthCat", "DayOfWeekCat", "UniqueCarrierCat",  "DepTime", "CRSDepTime", "TaxiOut","CRSArrTime","CRSElapsedTime","DepDelay", "Distance"))
      .setOutputCol("features")

    def preppedLRPipeline():TrainValidationSplit = {
      val lr = new LinearRegression()

      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .addGrid(lr.fitIntercept)
        .addGrid(lr.elasticNetParam, Array(0.25, 0.5, 0.75, 1.0))
        .build()


      val pipeline = new Pipeline()
        .setStages(Array(monthIndexer, dayofMonthIndexer,
          dayOfWeekIndexer, uniqueCarrierIndexer,
          assembler, lr))

      val tvs = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(new RegressionEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(0.75)
      tvs
    }

    def savePredictions(predictions:DataFrame, testRaw:DataFrame) = {
      val tdOut = testRaw
        .select("Id")
        .distinct()
        .join(predictions, testRaw("Id") === predictions("PredId"), "outer")
        .select("Id", "label")
        .na.fill(0:Double) // some of our inputs were null so we have to
      // fill these with something
      tdOut
        .coalesce(1)
        .write.format("com.databricks.spark.csv")
        .option("header", "true")
        .save("linear_regression_predictions.csv")
    }

    def fitModel(tvs:TrainValidationSplit, data:DataFrame) = {
      val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)
      println("Fitting data")

      val model = tvs.fit(training)
      println("Now performing test on hold out set")
      val holdout = model.transform(test).select("prediction","label")

      // have to do a type conversion for RegressionMetrics
      val rm = new RegressionMetrics(holdout.rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

      // Hadoop Config is accessible from SparkContext
      val fs = FileSystem.get(sparkSession.sparkContext.hadoopConfiguration);
      // Output file can be created from file system.
      val output = fs.create(new Path("results.txt"));
      // But BufferedOutputStream must be used to output an actual text file.
      val os = new BufferedOutputStream(output)
      os.write("Test Metrics \\n".getBytes("UTF-8"))
      os.write("Test Explained Variance: ".getBytes("UTF-8"))
      os.write(rm.explainedVariance.toString.getBytes("UTF-8"))
      os.write("\\nTest R^2 Coef: ".getBytes("UTF-8"))
      os.write(rm.r2.toString.getBytes("UTF-8"))
      os.write("\\nTest MSE: ".getBytes("UTF-8"))
      os.write(rm.meanSquaredError.toString.getBytes("UTF-8"))
      os.write("\\nTest RMSE: ".getBytes("UTF-8"))
      os.write(rm.rootMeanSquaredError.toString.getBytes("UTF-8"))
      os.close()



      model
    }

    // The linear Regression Pipeline
    val linearTvs = preppedLRPipeline()
    println("evaluating linear regression")
    val lrModel = fitModel(linearTvs, trainingData)
  }
}
