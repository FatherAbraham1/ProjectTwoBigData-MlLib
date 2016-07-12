package classifierCore.functions;

import org.apache.spark.api.java.function.Function;
import org.bson.BSONObject;

import scala.Tuple2;

public class FilterCancelled implements Function<Tuple2<Object, BSONObject>, Boolean> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public Boolean call(Tuple2<Object, BSONObject> v1) throws Exception {
		if((double)v1._2.get("Cancelled") == 1 || (double)v1._2.get("Diverted") == 1 ||
				"".equals(v1._2.get("ArrDelayMinutes")) || "".equals(v1._2.get("DepDelayMinutes")) ||
				"".equals(v1._2.get("CRSArrTime")))
			return false;
		return true;
	}
}