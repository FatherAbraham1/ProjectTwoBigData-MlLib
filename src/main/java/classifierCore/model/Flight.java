package classifierCore.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Flight {
	private Integer year;
	private Integer quarter;
	private Integer month;
	private Integer dayofMonth;
	private Integer dayOfWeek;
	private String uniqueCarrier;
	private String carrier;
	private String origin;
	private String dest;
	private Integer CRSDepTime;
	private Integer depTime;
	private Double depDelay;
	private Double depDelayMinutes;
	private Double depDel15;
	private Integer departureDelayGroups;
	private String depTimeBlk;
	private Integer CRSArrTime;
	private Integer arrTime;
	private Double arrDelay;
	private Double arrDelayMinutes;
	private Double arrDel15;
	private Integer arrivalDelayGroups;
	private String arrTimeBlk;
	private Double cancelled;
	private String cancellationCode;
	private Double diverted;
	private Double CRSElapsedTime;
	private Double distance;
	private Integer distanceGroup;
	private Double carrierDelay;
	private Double weatherDelay;
	private Double NASDelay;
	private Double securityDelay;
	private Double lateAircraftDelay;

	private final double maxHashAirport = 89370+1;
	private final double minHashAirport = 64545-1;

	private final double maxHashCarrier = 2880+1;
	private final double minHashCarrier = 1536-1;

	private final double maxValueYear = 2016+1;
	private final double minValueYear = 1987-1;

	private final double maxValueMonth = 12+1;
	private final double minValueMonth = 1-1;

	private final double maxValueDayOfMonth = 31+1;
	private final double minValueDayOfMonth = 1-1;

	private final double maxValueDayOfWeek = 7+1;
	private final double minValueDayOfWeek = 1-1;

	private final double maxValueHalfHour = 47+1;

	/**
	 * No args constructor for use in serialization
	 * 
	 */
	public Flight() {
	}

	public Flight(Integer year, Integer quarter, Integer month, Integer dayofMonth, Integer dayOfWeek,
			String uniqueCarrier, String carrier, String origin, String dest, Integer CRSDepTime, Integer depTime,
			Double depDelay, Double depDelayMinutes, Double depDel15, Integer departureDelayGroups,
			String depTimeBlk, Integer CRSArrTime, Integer arrTime, Double arrDelay, Double arrDelayMinutes,
			Double arrDel15, Integer arrivalDelayGroups, String arrTimeBlk, Double cancelled, String cancellationCode,
			Double diverted, Double CRSElapsedTime, Double distance, Integer distanceGroup, Double carrierDelay,
			Double weatherDelay, Double NASDelay, Double securityDelay, Double lateAircraftDelay) {
		super();
		this.year = year;
		this.quarter = quarter;
		this.month = month;
		this.dayofMonth = dayofMonth;
		this.dayOfWeek = dayOfWeek;
		this.uniqueCarrier = uniqueCarrier;
		this.carrier = carrier;
		this.origin = origin;
		this.dest = dest;
		this.CRSDepTime = CRSDepTime;
		this.depTime = depTime;
		this.depDelay = depDelay;
		this.depDelayMinutes = depDelayMinutes;
		this.depDel15 = depDel15;
		this.departureDelayGroups = departureDelayGroups;
		this.depTimeBlk = depTimeBlk;
		this.CRSArrTime = CRSArrTime;
		this.arrTime = arrTime;
		this.arrDelay = arrDelay;
		this.arrDelayMinutes = arrDelayMinutes;
		this.arrDel15 = arrDel15;
		this.arrivalDelayGroups = arrivalDelayGroups;
		this.arrTimeBlk = arrTimeBlk;
		this.cancelled = cancelled;
		this.cancellationCode = cancellationCode;
		this.diverted = diverted;
		this.CRSElapsedTime = CRSElapsedTime;
		this.distance = distance;
		this.distanceGroup = distanceGroup;
		this.carrierDelay = carrierDelay;
		this.weatherDelay = weatherDelay;
		this.NASDelay = NASDelay;
		this.securityDelay = securityDelay;
		this.lateAircraftDelay = lateAircraftDelay;
	}

	/**
	 * 
	 * @return
	 * The classifier features
	 */
	public Map<String, Double> getMapFeatures() {
		Map<String, Double> features = new HashMap<String, Double>();

		features.put("Origin", parseAirportCode(this.origin));
		features.put("Dest", parseAirportCode(this.dest));
		features.put("Carrier", parseCarrierCode(this.uniqueCarrier));
		features.put("CRSDepTime", parseTime(this.CRSDepTime));
		features.put("CRSArrTime", parseTime(this.CRSArrTime));
		//features.put("Year", parseYear(this.year));
		features.put("Month", parseMonth(this.month));
		features.put("DayOfMonth", parseDayOfMonth(this.dayofMonth));
		features.put("DayOfWeek", parseDayOfWeek(this.dayOfWeek));

		return features;
	}

	/**
	 * 
	 * @return
	 * The classifier features
	 */
	public Map<String, Double> getMapFeaturesV2() {
		Map<String, Double> features = new TreeMap<String, Double>();

		features.put("Origin", parseAirportCode(this.origin));
		features.put("Dest", parseAirportCode(this.dest));
		features.put("Carrier", parseCarrierCode(this.uniqueCarrier));
		features.putAll(parseTimeV2(this.CRSDepTime, "Dep"));
		features.putAll(parseTimeV2(this.CRSArrTime, "Arr"));
		//features.put("Year", parseYear(this.year));
		features.putAll(parseMonthV2(this.month));
		features.put("DayOfMonth", parseDayOfMonth(this.dayofMonth));
		features.putAll(parseDayOfWeekV2(this.dayOfWeek));

		return features;
	}
	
	/**
	 * 
	 * @return
	 * The classifier features
	 */
	public Map<String, Double> getMapFeaturesLR(List<String> airports, List<String> carriers) {
		Map<String, Double> features = new TreeMap<String, Double>();

		features.putAll(parseAirportCodeV2(this.origin, airports, "Dep"));
		features.putAll(parseAirportCodeV2(this.dest, airports, "Arr"));
		features.putAll(parseCarrierCodeV2(this.uniqueCarrier, carriers));
		features.putAll(parseTimeV2(this.CRSDepTime, "Dep"));
		features.putAll(parseTimeV2(this.CRSArrTime, "Arr"));
		//features.put("Year", parseYear(this.year));
		features.putAll(parseMonthV2(this.month));
		//features.put("DayOfMonth", parseDayOfMonth(this.dayofMonth));
		features.putAll(parseDayOfWeekV2(this.dayOfWeek));

		return features;
	}

	/**
	 * 
	 * @return
	 * The classifier features array
	 */
	public double[] getArrayFeatures(Map<String, Double> features) {
		double[] featureArray = new double[features.size()];

		int count = 0;

		for(String key: features.keySet()){
			featureArray[count] = features.get(key);
			count++;
			if(features.get(key) < 0){
				throw new RuntimeException("Feature negativa! "+features.toString());
			}
		}

		return featureArray;
	}

	/**
	 * 
	 * @return
	 * The classifier features array
	 */
	public Vector getVectorFeatures() {
		Map<String, Double> features = getMapFeatures();
		return Vectors.dense(getArrayFeatures(features));
	}

	/**
	 * 
	 * @return
	 * The classifier features array
	 */
	public Vector getVectorFeaturesV2() {
		Map<String, Double> features = getMapFeaturesV2();
		return Vectors.dense(getArrayFeatures(features));
	}
	
	/**
	 * 
	 * @return
	 * The classifier features array
	 */
	public Vector getVectorFeaturesLR(List<String> airports, List<String> carriers) {
		Map<String, Double> features = getMapFeaturesLR(airports, carriers);
		return Vectors.dense(getArrayFeatures(features));
	}

	/**
	 * 
	 * @return
	 * The feature airport code
	 */
	private double parseAirportCode(String iata){
		int hash = iata.hashCode();

		return (hash-this.minHashAirport)/(this.maxHashAirport-this.minHashAirport);
	}

	/**
	 * 
	 * @return
	 * The feature carrier code
	 */
	private double parseCarrierCode(String codeCarrier){
		double hash = codeCarrier.substring(0, 2).hashCode();

		return (hash-this.minHashCarrier)/(this.maxHashCarrier-this.minHashCarrier);
	}

	/**
	 * 
	 * @return
	 * The feature time
	 */
	private double parseTime(Integer time){	
		int hours = time/100;
		int minutes = time - (hours*100);
		double half = (double)hours*2;

		if(minutes>30)
			half++;

		return half/this.maxValueHalfHour;
	}
	
	/**
	 * 
	 * @return
	 * The feature airports V2
	 */
	private Map<String, Double> parseAirportCodeV2(String airport, List<String> airports, String depOrArr){
		Map<String, Double> featuresAirports = new TreeMap<String, Double>();

		for(String current: airports){
			if(!current.equals(airport))
				featuresAirports.put("Airport"+depOrArr+current, 0.0);
			else
				featuresAirports.put("Airport"+depOrArr+current, 1.0);
		}

		return featuresAirports;
	}

	/**
	 * 
	 * @return
	 * The feature carriers dep V2
	 */
	private Map<String, Double> parseCarrierCodeV2(String carrier, List<String> carriers){
		Map<String, Double> featuresAirports = new TreeMap<String, Double>();

		for(String current: carriers){
			if(!current.equals(carrier))
				featuresAirports.put("Carrier"+current, 0.0);
			else
				featuresAirports.put("Carrier"+current, 1.0);
		}

		return featuresAirports;
	}

	/**
	 * 
	 * @return
	 * The feature time V2
	 */
	private Map<String, Double> parseTimeV2(Integer time, String depOrArr){
		Map<String, Double> featuresTime = new TreeMap<String, Double>();

		int hours = time/100;
		int minutes = time - (hours*100);
		double half = (double)hours*2;

		if(minutes>=30)
			half++;

		for(int i=1; i<=48; i++){
			if(i != half)
				if(i>=10)
					featuresTime.put("HalfHourBlock"+depOrArr+i, 0.0);
				else
					featuresTime.put("HalfHourBlock"+depOrArr+"0"+i, 0.0);
			else
				if(i>=10)
					featuresTime.put("HalfHourBlock"+depOrArr+i, 1.0);
				else
					featuresTime.put("HalfHourBlock"+depOrArr+"0"+i, 1.0);
		}

		return featuresTime;
	}

	/**
	 * 
	 * @return
	 * The feature month
	 */
	private double parseMonth(Integer month){
		return ((double)month-this.minValueMonth)/(this.maxValueMonth-this.minValueMonth);
	}

	/**
	 * 
	 * @return
	 * The feature month V2
	 */
	private Map<String, Double> parseMonthV2(Integer month){
		Map<String, Double> featuresMonth = new TreeMap<String, Double>();

		for(int i=1; i<=12; i++){
			if(i!=month){
				if(i>=10)
					featuresMonth.put("Month"+i, 0.0);
				else
					featuresMonth.put("Month0"+i, 0.0);
			}else{
				if(i>=10)
					featuresMonth.put("Month"+i, 1.0);
				else
					featuresMonth.put("Month0"+i, 1.0);
			}
		}

		return featuresMonth;
	}

	/**
	 * 
	 * @return
	 * The feature year
	 */
	@SuppressWarnings("unused")
	private double parseYear(Integer year){
		return ((double)year-this.minValueYear)/(this.maxValueYear-this.minValueYear);
	}

	/**
	 * 
	 * @return
	 * The feature dayOfMonth
	 */
	private double parseDayOfMonth(Integer dayOfMonth){
		return ((double)dayOfMonth-this.minValueDayOfMonth)/(this.maxValueDayOfMonth-this.minValueDayOfMonth);
	}

	/**
	 * 
	 * @return
	 * The feature dayOfWeek
	 */
	private double parseDayOfWeek(Integer dayOfWeek){
		return ((double)dayOfWeek-this.minValueDayOfWeek)/(this.maxValueDayOfWeek-this.minValueDayOfWeek);
	}

	/**
	 * 
	 * @return
	 * The feature month V2
	 */
	private Map<String, Double> parseDayOfWeekV2(Integer week){
		Map<String, Double> featuresWeek = new TreeMap<String, Double>();

		for(int i=1; i<=7; i++){
			if(i!=week)
				featuresWeek.put("Week"+i, 0.0);
			else
				featuresWeek.put("Week"+i, 1.0);
		}

		return featuresWeek;
	}

	/**
	 * 
	 * @return
	 * The year
	 */
	public Integer getYear() {
		return year;
	}

	/**
	 * 
	 * @param year
	 * The Year
	 */
	public void setYear(Integer year) {
		this.year = year;
	}

	/**
	 * 
	 * @return
	 * The quarter
	 */
	public Integer getQuarter() {
		return quarter;
	}

	/**
	 * 
	 * @param quarter
	 * The Quarter
	 */
	public void setQuarter(Integer quarter) {
		this.quarter = quarter;
	}

	/**
	 * 
	 * @return
	 * The month
	 */
	public Integer getMonth() {
		return month;
	}

	/**
	 * 
	 * @param month
	 * The Month
	 */
	public void setMonth(Integer month) {
		this.month = month;
	}

	/**
	 * 
	 * @return
	 * The dayofMonth
	 */
	public Integer getDayofMonth() {
		return dayofMonth;
	}

	/**
	 * 
	 * @param dayofMonth
	 * The DayofMonth
	 */
	public void setDayofMonth(Integer dayofMonth) {
		this.dayofMonth = dayofMonth;
	}

	/**
	 * 
	 * @return
	 * The dayOfWeek
	 */
	public Integer getDayOfWeek() {
		return dayOfWeek;
	}

	/**
	 * 
	 * @param dayOfWeek
	 * The DayOfWeek
	 */
	public void setDayOfWeek(Integer dayOfWeek) {
		this.dayOfWeek = dayOfWeek;
	}

	/**
	 * 
	 * @return
	 * The uniqueCarrier
	 */
	public String getUniqueCarrier() {
		return uniqueCarrier;
	}

	/**
	 * 
	 * @param string
	 * The UniqueCarrier
	 */
	public void setUniqueCarrier(String string) {
		this.uniqueCarrier = string;
	}

	/**
	 * 
	 * @return
	 * The carrier
	 */
	public String getCarrier() {
		return carrier;
	}

	/**
	 * 
	 * @param carrier
	 * The Carrier
	 */
	public void setCarrier(String carrier) {
		this.carrier = carrier;
	}

	/**
	 * 
	 * @return
	 * The origin
	 */
	public String getOrigin() {
		return origin;
	}

	/**
	 * 
	 * @param integer
	 * The Origin
	 */
	public void setOrigin(String integer) {
		this.origin = integer;
	}

	/**
	 * 
	 * @return
	 * The dest
	 */
	public String getDest() {
		return dest;
	}

	/**
	 * 
	 * @param dest
	 * The Dest
	 */
	public void setDest(String dest) {
		this.dest = dest;
	}

	/**
	 * 
	 * @return
	 * The cRSDepTime
	 */
	public Integer getCRSDepTime() {
		return CRSDepTime;
	}

	/**
	 * 
	 * @param cRSDepTime
	 * The CRSDepTime
	 */
	public void setCRSDepTime(Integer cRSDepTime) {
		this.CRSDepTime = cRSDepTime;
	}

	/**
	 * 
	 * @return
	 * The depTime
	 */
	public Integer getDepTime() {
		return depTime;
	}

	/**
	 * 
	 * @param depTime
	 * The DepTime
	 */
	public void setDepTime(Integer depTime) {
		this.depTime = depTime;
	}

	/**
	 * 
	 * @return
	 * The depDelay
	 */
	public Double getDepDelay() {
		return depDelay;
	}

	/**
	 * 
	 * @param depDelay
	 * The DepDelay
	 */
	public void setDepDelay(Double depDelay) {
		this.depDelay = depDelay;
	}

	/**
	 * 
	 * @return
	 * The depDelayMinutes
	 */
	public Double getDepDelayMinutes() {
		return depDelayMinutes;
	}

	/**
	 * 
	 * @param depDelayMinutes
	 * The DepDelayMinutes
	 */
	public void setDepDelayMinutes(Double depDelayMinutes) {
		this.depDelayMinutes = depDelayMinutes;
	}

	/**
	 * 
	 * @return
	 * The depDel15
	 */
	public Double getDepDel15() {
		return depDel15;
	}

	/**
	 * 
	 * @param depDel15
	 * The DepDel15
	 */
	public void setDepDel15(Double depDel15) {
		this.depDel15 = depDel15;
	}

	/**
	 * 
	 * @return
	 * The departureDelayGroups
	 */
	public Integer getDepartureDelayGroups() {
		return departureDelayGroups;
	}

	/**
	 * 
	 * @param departureDelayGroups
	 * The DepartureDelayGroups
	 */
	public void setDepartureDelayGroups(Integer departureDelayGroups) {
		this.departureDelayGroups = departureDelayGroups;
	}

	/**
	 * 
	 * @return
	 * The depTimeBlk
	 */
	public String getDepTimeBlk() {
		return depTimeBlk;
	}

	/**
	 * 
	 * @param depTimeBlk
	 * The DepTimeBlk
	 */
	public void setDepTimeBlk(String depTimeBlk) {
		this.depTimeBlk = depTimeBlk;
	}

	/**
	 * 
	 * @return
	 * The cRSArrTime
	 */
	public Integer getCRSArrTime() {
		return CRSArrTime;
	}

	/**
	 * 
	 * @param cRSArrTime
	 * The CRSArrTime
	 */
	public void setCRSArrTime(Integer cRSArrTime) {
		this.CRSArrTime = cRSArrTime;
	}

	/**
	 * 
	 * @return
	 * The arrTime
	 */
	public Integer getArrTime() {
		return arrTime;
	}

	/**
	 * 
	 * @param arrTime
	 * The ArrTime
	 */
	public void setArrTime(Integer arrTime) {
		this.arrTime = arrTime;
	}

	/**
	 * 
	 * @return
	 * The arrDelay
	 */
	public Double getArrDelay() {
		return arrDelay;
	}

	/**
	 * 
	 * @param arrDelay
	 * The ArrDelay
	 */
	public void setArrDelay(Double arrDelay) {
		this.arrDelay = arrDelay;
	}

	/**
	 * 
	 * @return
	 * The arrDelayMinutes
	 */
	public Double getArrDelayMinutes() {
		return arrDelayMinutes;
	}

	/**
	 * 
	 * @param arrDelayMinutes
	 * The ArrDelayMinutes
	 */
	public void setArrDelayMinutes(Double arrDelayMinutes) {
		this.arrDelayMinutes = arrDelayMinutes;
	}

	/**
	 * 
	 * @return
	 * The arrDel15
	 */
	public Double getArrDel15() {
		return arrDel15;
	}

	/**
	 * 
	 * @param arrDel15
	 * The ArrDel15
	 */
	public void setArrDel15(Double arrDel15) {
		this.arrDel15 = arrDel15;
	}

	/**
	 * 
	 * @return
	 * The arrivalDelayGroups
	 */
	public Integer getArrivalDelayGroups() {
		return arrivalDelayGroups;
	}

	/**
	 * 
	 * @param arrivalDelayGroups
	 * The ArrivalDelayGroups
	 */
	public void setArrivalDelayGroups(Integer arrivalDelayGroups) {
		this.arrivalDelayGroups = arrivalDelayGroups;
	}

	/**
	 * 
	 * @return
	 * The arrTimeBlk
	 */
	public String getArrTimeBlk() {
		return arrTimeBlk;
	}

	/**
	 * 
	 * @param arrTimeBlk
	 * The ArrTimeBlk
	 */
	public void setArrTimeBlk(String arrTimeBlk) {
		this.arrTimeBlk = arrTimeBlk;
	}

	/**
	 * 
	 * @return
	 * The cancelled
	 */
	public Double getCancelled() {
		return cancelled;
	}

	/**
	 * 
	 * @param cancelled
	 * The Cancelled
	 */
	public void setCancelled(Double cancelled) {
		this.cancelled = cancelled;
	}

	/**
	 * 
	 * @return
	 * The cancellationCode
	 */
	public String getCancellationCode() {
		return cancellationCode;
	}

	/**
	 * 
	 * @param cancellationCode
	 * The CancellationCode
	 */
	public void setCancellationCode(String cancellationCode) {
		this.cancellationCode = cancellationCode;
	}

	/**
	 * 
	 * @return
	 * The diverted
	 */
	public Double getDiverted() {
		return diverted;
	}

	/**
	 * 
	 * @param diverted
	 * The Diverted
	 */
	public void setDiverted(Double diverted) {
		this.diverted = diverted;
	}

	/**
	 * 
	 * @return
	 * The cRSElapsedTime
	 */
	public Double getCRSElapsedTime() {
		return CRSElapsedTime;
	}

	/**
	 * 
	 * @param cRSElapsedTime
	 * The CRSElapsedTime
	 */
	public void setCRSElapsedTime(Double cRSElapsedTime) {
		this.CRSElapsedTime = cRSElapsedTime;
	}

	/**
	 * 
	 * @return
	 * The distance
	 */
	public Double getDistance() {
		return distance;
	}

	/**
	 * 
	 * @param distance
	 * The Distance
	 */
	public void setDistance(Double distance) {
		this.distance = distance;
	}

	/**
	 * 
	 * @return
	 * The distanceGroup
	 */
	public Integer getDistanceGroup() {
		return distanceGroup;
	}

	/**
	 * 
	 * @param distanceGroup
	 * The DistanceGroup
	 */
	public void setDistanceGroup(Integer distanceGroup) {
		this.distanceGroup = distanceGroup;
	}

	/**
	 * 
	 * @return
	 * The carrierDelay
	 */
	public Double getCarrierDelay() {
		return carrierDelay;
	}

	/**
	 * 
	 * @param carrierDelay
	 * The CarrierDelay
	 */
	public void setCarrierDelay(Double carrierDelay) {
		this.carrierDelay = carrierDelay;
	}

	/**
	 * 
	 * @return
	 * The weatherDelay
	 */
	public Double getWeatherDelay() {
		return weatherDelay;
	}

	/**
	 * 
	 * @param weatherDelay
	 * The WeatherDelay
	 */
	public void setWeatherDelay(Double weatherDelay) {
		this.weatherDelay = weatherDelay;
	}

	/**
	 * 
	 * @return
	 * The nASDelay
	 */
	public Double getNASDelay() {
		return NASDelay;
	}

	/**
	 * 
	 * @param nASDelay
	 * The NASDelay
	 */
	public void setNASDelay(Double NASDelay) {
		this.NASDelay = NASDelay;
	}

	/**
	 * 
	 * @return
	 * The securityDelay
	 */
	public Double getSecurityDelay() {
		return securityDelay;
	}

	/**
	 * 
	 * @param securityDelay
	 * The SecurityDelay
	 */
	public void setSecurityDelay(Double securityDelay) {
		this.securityDelay = securityDelay;
	}

	/**
	 * 
	 * @return
	 * The lateAircraftDelay
	 */
	public Double getLateAircraftDelay() {
		return lateAircraftDelay;
	}

	/**
	 * 
	 * @param lateAircraftDelay
	 * The LateAircraftDelay
	 */
	public void setLateAircraftDelay(Double lateAircraftDelay) {
		this.lateAircraftDelay = lateAircraftDelay;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((CRSArrTime == null) ? 0 : CRSArrTime.hashCode());
		result = prime * result + ((CRSDepTime == null) ? 0 : CRSDepTime.hashCode());
		result = prime * result + ((CRSElapsedTime == null) ? 0 : CRSElapsedTime.hashCode());
		result = prime * result + ((arrDel15 == null) ? 0 : arrDel15.hashCode());
		result = prime * result + ((arrDelay == null) ? 0 : arrDelay.hashCode());
		result = prime * result + ((arrDelayMinutes == null) ? 0 : arrDelayMinutes.hashCode());
		result = prime * result + ((arrTime == null) ? 0 : arrTime.hashCode());
		result = prime * result + ((arrTimeBlk == null) ? 0 : arrTimeBlk.hashCode());
		result = prime * result + ((arrivalDelayGroups == null) ? 0 : arrivalDelayGroups.hashCode());
		result = prime * result + ((cancellationCode == null) ? 0 : cancellationCode.hashCode());
		result = prime * result + ((cancelled == null) ? 0 : cancelled.hashCode());
		result = prime * result + ((carrier == null) ? 0 : carrier.hashCode());
		result = prime * result + ((carrierDelay == null) ? 0 : carrierDelay.hashCode());
		result = prime * result + ((dayOfWeek == null) ? 0 : dayOfWeek.hashCode());
		result = prime * result + ((dayofMonth == null) ? 0 : dayofMonth.hashCode());
		result = prime * result + ((depDel15 == null) ? 0 : depDel15.hashCode());
		result = prime * result + ((depDelay == null) ? 0 : depDelay.hashCode());
		result = prime * result + ((depDelayMinutes == null) ? 0 : depDelayMinutes.hashCode());
		result = prime * result + ((depTime == null) ? 0 : depTime.hashCode());
		result = prime * result + ((depTimeBlk == null) ? 0 : depTimeBlk.hashCode());
		result = prime * result + ((departureDelayGroups == null) ? 0 : departureDelayGroups.hashCode());
		result = prime * result + ((dest == null) ? 0 : dest.hashCode());
		result = prime * result + ((distance == null) ? 0 : distance.hashCode());
		result = prime * result + ((distanceGroup == null) ? 0 : distanceGroup.hashCode());
		result = prime * result + ((diverted == null) ? 0 : diverted.hashCode());
		result = prime * result + ((lateAircraftDelay == null) ? 0 : lateAircraftDelay.hashCode());
		result = prime * result + ((month == null) ? 0 : month.hashCode());
		result = prime * result + ((NASDelay == null) ? 0 : NASDelay.hashCode());
		result = prime * result + ((origin == null) ? 0 : origin.hashCode());
		result = prime * result + ((quarter == null) ? 0 : quarter.hashCode());
		result = prime * result + ((securityDelay == null) ? 0 : securityDelay.hashCode());
		result = prime * result + ((uniqueCarrier == null) ? 0 : uniqueCarrier.hashCode());
		result = prime * result + ((weatherDelay == null) ? 0 : weatherDelay.hashCode());
		result = prime * result + ((year == null) ? 0 : year.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Flight other = (Flight) obj;
		if (CRSArrTime == null) {
			if (other.CRSArrTime != null)
				return false;
		} else if (!CRSArrTime.equals(other.CRSArrTime))
			return false;
		if (CRSDepTime == null) {
			if (other.CRSDepTime != null)
				return false;
		} else if (!CRSDepTime.equals(other.CRSDepTime))
			return false;
		if (CRSElapsedTime == null) {
			if (other.CRSElapsedTime != null)
				return false;
		} else if (!CRSElapsedTime.equals(other.CRSElapsedTime))
			return false;
		if (arrDel15 == null) {
			if (other.arrDel15 != null)
				return false;
		} else if (!arrDel15.equals(other.arrDel15))
			return false;
		if (arrDelay == null) {
			if (other.arrDelay != null)
				return false;
		} else if (!arrDelay.equals(other.arrDelay))
			return false;
		if (arrDelayMinutes == null) {
			if (other.arrDelayMinutes != null)
				return false;
		} else if (!arrDelayMinutes.equals(other.arrDelayMinutes))
			return false;
		if (arrTime == null) {
			if (other.arrTime != null)
				return false;
		} else if (!arrTime.equals(other.arrTime))
			return false;
		if (arrTimeBlk == null) {
			if (other.arrTimeBlk != null)
				return false;
		} else if (!arrTimeBlk.equals(other.arrTimeBlk))
			return false;
		if (arrivalDelayGroups == null) {
			if (other.arrivalDelayGroups != null)
				return false;
		} else if (!arrivalDelayGroups.equals(other.arrivalDelayGroups))
			return false;
		if (cancellationCode == null) {
			if (other.cancellationCode != null)
				return false;
		} else if (!cancellationCode.equals(other.cancellationCode))
			return false;
		if (cancelled == null) {
			if (other.cancelled != null)
				return false;
		} else if (!cancelled.equals(other.cancelled))
			return false;
		if (carrier == null) {
			if (other.carrier != null)
				return false;
		} else if (!carrier.equals(other.carrier))
			return false;
		if (carrierDelay == null) {
			if (other.carrierDelay != null)
				return false;
		} else if (!carrierDelay.equals(other.carrierDelay))
			return false;
		if (dayOfWeek == null) {
			if (other.dayOfWeek != null)
				return false;
		} else if (!dayOfWeek.equals(other.dayOfWeek))
			return false;
		if (dayofMonth == null) {
			if (other.dayofMonth != null)
				return false;
		} else if (!dayofMonth.equals(other.dayofMonth))
			return false;
		if (depDel15 == null) {
			if (other.depDel15 != null)
				return false;
		} else if (!depDel15.equals(other.depDel15))
			return false;
		if (depDelay == null) {
			if (other.depDelay != null)
				return false;
		} else if (!depDelay.equals(other.depDelay))
			return false;
		if (depDelayMinutes == null) {
			if (other.depDelayMinutes != null)
				return false;
		} else if (!depDelayMinutes.equals(other.depDelayMinutes))
			return false;
		if (depTime == null) {
			if (other.depTime != null)
				return false;
		} else if (!depTime.equals(other.depTime))
			return false;
		if (depTimeBlk == null) {
			if (other.depTimeBlk != null)
				return false;
		} else if (!depTimeBlk.equals(other.depTimeBlk))
			return false;
		if (departureDelayGroups == null) {
			if (other.departureDelayGroups != null)
				return false;
		} else if (!departureDelayGroups.equals(other.departureDelayGroups))
			return false;
		if (dest == null) {
			if (other.dest != null)
				return false;
		} else if (!dest.equals(other.dest))
			return false;
		if (distance == null) {
			if (other.distance != null)
				return false;
		} else if (!distance.equals(other.distance))
			return false;
		if (distanceGroup == null) {
			if (other.distanceGroup != null)
				return false;
		} else if (!distanceGroup.equals(other.distanceGroup))
			return false;
		if (diverted == null) {
			if (other.diverted != null)
				return false;
		} else if (!diverted.equals(other.diverted))
			return false;
		if (lateAircraftDelay == null) {
			if (other.lateAircraftDelay != null)
				return false;
		} else if (!lateAircraftDelay.equals(other.lateAircraftDelay))
			return false;
		if (month == null) {
			if (other.month != null)
				return false;
		} else if (!month.equals(other.month))
			return false;
		if (NASDelay == null) {
			if (other.NASDelay != null)
				return false;
		} else if (!NASDelay.equals(other.NASDelay))
			return false;
		if (origin == null) {
			if (other.origin != null)
				return false;
		} else if (!origin.equals(other.origin))
			return false;
		if (quarter == null) {
			if (other.quarter != null)
				return false;
		} else if (!quarter.equals(other.quarter))
			return false;
		if (securityDelay == null) {
			if (other.securityDelay != null)
				return false;
		} else if (!securityDelay.equals(other.securityDelay))
			return false;
		if (uniqueCarrier == null) {
			if (other.uniqueCarrier != null)
				return false;
		} else if (!uniqueCarrier.equals(other.uniqueCarrier))
			return false;
		if (weatherDelay == null) {
			if (other.weatherDelay != null)
				return false;
		} else if (!weatherDelay.equals(other.weatherDelay))
			return false;
		if (year == null) {
			if (other.year != null)
				return false;
		} else if (!year.equals(other.year))
			return false;
		return true;
	}

	public double getCategoryDelay() {
		double cat = 0;
		if(this.arrDelay>0 && this.arrDelay<=20)
			cat = 1;
		if(this.arrDelay>20 && this.arrDelay<=90)
			cat = 2;
		if(this.arrDelay>90)
			cat = 3;

		return cat;
	}
	
	public double getCategoryDelaySecond() {
		double cat = 1;
		if(this.arrDelay>15 && this.arrDelay<=60)
			cat = 2;
		if(this.arrDelay>60 && this.arrDelay<=180)
			cat = 3;
		if(this.arrDelay>180 && this.arrDelay<=1440)
			cat = 4;
		if(this.arrDelay>1440)
			cat = 5;

		return cat;
	}

	public double isDelay() {
		double cat = 0;

		if(this.arrDelay>0)
			cat = 1;

		return cat;
	}
}