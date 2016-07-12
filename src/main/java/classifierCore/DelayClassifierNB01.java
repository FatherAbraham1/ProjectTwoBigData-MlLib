package classifierCore;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.bson.BSONObject;

import com.mongodb.hadoop.MongoInputFormat;

import classifierCore.model.Flight;
import classifierCore.functions.FilterCancelled;
import classifierCore.functions.ManagingFlights;
import classifierCore.functions.ParsingFuction01;

import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;

public class DelayClassifierNB01 {
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Naive Bayes Predict 0-1").setMaster("local");
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

		final NaiveBayesModel model = NaiveBayes.train(trainingData.rdd(), 1.0);

		JavaPairRDD<Double, Double> predictionAndLabel =
				testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		
		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Double, Double> pl) {
				boolean rtn = pl._1().equals(pl._2());
				return rtn;
			}
		}).count() / (double) testData.count();

		System.out.println("Accuratezza: "+accuracy);

		// Save and load model
		model.save(sc.sc(), "myNaiveBayesModel01");
	}
}