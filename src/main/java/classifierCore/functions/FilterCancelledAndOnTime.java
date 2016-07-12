package classifierCore.functions;

import org.apache.spark.api.java.function.Function;
import org.bson.BSONObject;

import scala.Tuple2;

public class FilterCancelledAndOnTime implements Function<Tuple2<Object, BSONObject>, Boolean> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public Boolean call(Tuple2<Object, BSONObject> v1) throws Exception {
		if((Double)v1._2.get("Cancelled") == 1 || ((Double)v1._2.get("Diverted") == 1 || (Integer)v1._2.get("Year") < 2015 || (Double)v1._2.get("DepDelay") <= 0))
			return false;
		return true;
	}
}