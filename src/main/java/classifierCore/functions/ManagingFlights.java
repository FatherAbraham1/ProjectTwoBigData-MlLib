package classifierCore.functions;

import org.apache.spark.api.java.function.Function;
import org.bson.BSONObject;

import classifierCore.model.Flight;
import scala.Tuple2;

public class ManagingFlights implements Function<Tuple2<Object, BSONObject>, Flight> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public Flight call(Tuple2<Object, BSONObject> arg) {
		Flight flight = new Flight();

		flight.setYear((Integer) arg._2.get("Year"));
		flight.setMonth((Integer) arg._2.get("Month"));
		flight.setDayofMonth((Integer) arg._2.get("DayofMonth"));
		flight.setDayOfWeek((Integer) arg._2.get("DayOfWeek"));
		flight.setUniqueCarrier((String) arg._2.get("UniqueCarrier"));
		flight.setOrigin((String) arg._2.get("Origin"));
		flight.setDest((String) arg._2.get("Dest"));
		flight.setCRSDepTime(Integer.parseInt(arg._2.get("CRSDepTime").toString()));
		flight.setDepDelay((Double) arg._2.get("DepDelay"));
		flight.setDepDelayMinutes((Double) arg._2.get("DepDelayMinutes"));
		flight.setCRSArrTime(Integer.parseInt(arg._2.get("CRSArrTime").toString()));
		

		flight.setArrDelay((Double) arg._2.get("ArrDelay"));
		flight.setArrDelayMinutes((Double) arg._2.get("ArrDelayMinutes"));

		flight.setCancellationCode((String) arg._2.get("CancellationCode"));
		flight.setDistance((Double) arg._2.get("Distance"));

		return flight;
	}
}
