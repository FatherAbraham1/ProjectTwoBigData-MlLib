package classifierCore;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.bson.BSONObject;

import com.mongodb.hadoop.MongoInputFormat;

import classifierCore.model.Flight;
import classifierCore.functions.FilterCancelled;
import classifierCore.functions.ManagingFlights;
import classifierCore.functions.ParsingFuction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;

public class TestingDelayClassifierLR {
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Linear Regression Example").setMaster("local");
		sc = new JavaSparkContext(conf);
		
		final Map<Double, List<LabeledPoint>> predictedMap = new HashMap<Double, List<LabeledPoint>>();

		Configuration inputConfig = new Configuration();
		inputConfig.set("mongo.input.uri", "mongodb://localhost:27017/airplaneDB.input");

		JavaPairRDD<Object, BSONObject> inputRDDMongo = sc.newAPIHadoopRDD(
				inputConfig,            // Configuration
				MongoInputFormat.class,   // InputFormat: read from a live cluster.
				Object.class,             // Key class
				BSONObject.class          // Value class
				).filter(new FilterCancelled());

		// Caricare i dati da MongoDB
		JavaRDD<Flight> input = inputRDDMongo.map(new ManagingFlights());

		// Parsing dei dati estraendo solo le feature scelte
		JavaRDD<LabeledPoint> parsedInput = input.map(new ParsingFuction());

		// Costruzione del modello
		//	    int numIterations = 100;
		//	    double stepSize = 0.00000001;
		//	    final LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(parsedInput), numIterations, stepSize);

//		final LogisticRegressionModel model = LogisticRegressionModel.load(sc.sc(), "myModelPathLogistic");
		
//		final LinearRegressionModel model = LinearRegressionModel.load(sc.sc(), "myModelPath4");
		
//		final SVMModel model = SVMModel.load(sc.sc(), "myModelSVM2");
		
//		final RandomForestModel model = RandomForestModel.load(sc.sc(), "myRandomForestClassificationModelTest2");

		final NaiveBayesModel model = NaiveBayesModel.load(sc.sc(), "myNaiveBayesModelCategory");
//		System.out.println(model.weights().toString());
//
		Flight flight = new Flight();

		flight.setYear(2016);
		flight.setMonth(3);
		flight.setDayofMonth(6);
		flight.setDayOfWeek(5);
		flight.setUniqueCarrier("AA");
		flight.setOrigin("JFK");
		flight.setDest("ORD");
		flight.setCRSDepTime(900);
		flight.setCRSArrTime(1530);
		
//		List<String> airports = getAllAirports(sc);
//		List<String> carriers = getAllCarriers(sc);
//
		Map<String, Double> map = flight.getMapFeaturesV2();
//		System.out.println(map.toString());
		
		for(String key:map.keySet())
			System.out.println(key+" - "+map.get(key));
//		
//		
//		for(double a:flight.getArrayFeatures(map))
//			System.out.println(a);
//		System.out.println(flight.getVectorFeaturesLR(airports, carriers).size());
		
		

//		double prediction = model.predict(flight.getVectorFeaturesLR(airports, carriers));
		
//		System.out.println(prediction);

		// Evaluate model on training examples and compute training error
		//	    JavaRDD<Tuple2<Double, Double>> valuesAndPreds = parsedInput.map(
		//	      new Function<LabeledPoint, Tuple2<Double, Double>>() {
		//	        public Tuple2<Double, Double> call(LabeledPoint point) {
		//	          double prediction = sameModel.predict(point.features());
		//	          return new Tuple2<Double, Double>(prediction, point.label());
		//	        }
		//	      }
		//	    );
		//	    double MSE = new JavaDoubleRDD(valuesAndPreds.map(
		//	      new Function<Tuple2<Double, Double>, Object>() {
		//	        public Object call(Tuple2<Double, Double> pair) {
		//	          return Math.pow(pair._1() - pair._2(), 2.0);
		//	        }
		//	      }
		////	    ).rdd()).mean();
		//	    System.out.println(prediction);

		// Save and load model
		//	    model.save(sc.sc(), "myModelPath");

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = parsedInput.map(
				new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						double prediction = model.predict(p.features());
						System.out.println("Predizione: "+prediction+" - Valore atteso: "+p.label());
						List<LabeledPoint> list = predictedMap.get(prediction);
						
						Vector prob = model.predictProbabilities(p.features());
						System.out.println(Arrays.toString(prob.toArray()));
						
						if(list == null)
							list = new ArrayList<LabeledPoint>();
						list.add(p);
						
						predictedMap.put(prediction, list);
						return new Tuple2<Object, Object>(prediction, p.label());
					}
				}
				);
		

		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double precision = metrics.precision();
		double fmeasure = metrics.fMeasure();
		System.out.println("Precision = " + precision+"\nFMeasure = "+fmeasure);


//		System.out.println(prediction);
		
//		for(Double key: predictedMap.keySet())
//			System.out.println("Predicted value: "+key+" -> "+predictedMap.get(key).size());

	}
}