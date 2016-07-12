package classifierCore;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.bson.BSONObject;

import com.mongodb.hadoop.MongoInputFormat;

import classifierCore.model.Flight;
import classifierCore.functions.FilterCancelled;
import classifierCore.functions.ManagingFlights;
import classifierCore.functions.ParsingFuction01;

import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;

public class DelayClassifierRF {
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Linear Regression Example").setMaster("local");
		sc = new JavaSparkContext(conf);

		Configuration input2014Config = new Configuration();
		input2014Config.set("mongo.input.uri", "mongodb://localhost:27017/airplaneDB.input2014");

		JavaPairRDD<Object, BSONObject> input2014RDDMongo = sc.newAPIHadoopRDD(
				input2014Config,            // Configuration
				MongoInputFormat.class,   // InputFormat: read from a live cluster.
				Object.class,             // Key class
				BSONObject.class          // Value class
				).filter(new FilterCancelled());

		// Caricare i dati da MongoDB
		JavaRDD<Flight> input2014train = input2014RDDMongo.map(new ManagingFlights());
		
		Configuration input2015Config = new Configuration();
		input2015Config.set("mongo.input.uri", "mongodb://localhost:27017/airplaneDB.input2015");

		JavaPairRDD<Object, BSONObject> input2015RDDMongo = sc.newAPIHadoopRDD(
				input2015Config,            // Configuration
				MongoInputFormat.class,   // InputFormat: read from a live cluster.
				Object.class,             // Key class
				BSONObject.class          // Value class
				).filter(new FilterCancelled());

		// Caricare i dati da MongoDB
		JavaRDD<Flight> input2015test = input2015RDDMongo.map(new ManagingFlights());

		// Parsing dei dati estraendo solo le feature scelte
		JavaRDD<LabeledPoint> trainingData = input2014train.map(new ParsingFuction01());
		JavaRDD<LabeledPoint> testData = input2015test.map(new ParsingFuction01());

		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 6;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		Integer numTrees = 10; // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		Integer maxDepth = 6;
		Integer maxBins = 32;
		Integer seed = 12345;

		final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
				  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
				  seed);

		// Evaluate model on test instances and compute test error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = trainingData.map(
				new Function<LabeledPoint, Tuple2<Double, Double>>() {
					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<Double, Double>(prediction, point.label());
					}
				}
				);
		double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				new Function<Tuple2<Double, Double>, Object>() {
					public Object call(Tuple2<Double, Double> pair) {
						return Math.pow(pair._1() - pair._2(), 2.0);
					}
				}
				).rdd()).mean();
		System.out.println("training Mean Error = " + Math.sqrt(MSE));
		
		JavaPairRDD<Double, Double> predictionAndLabel =
				testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		double precision = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Double, Double> pl) {
				boolean rtn = pl._1().equals(pl._2());
				return rtn;
			}
		}).count() / (double) testData.count();

		System.out.println("Precisione: "+precision);

		// Save and load model
		model.save(sc.sc(), "myRandomForestClassificationModelTest3");
	}
}