#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <stdexcept>

namespace UnivariateCppInternal {

// Enumeration for percentile definition methods
enum class PctldefType {
  WEIGHTED_AVERAGE_AT_XNP = 1,  // Weighted average at x(np)
    CLOSEST_OBSERVATION = 2,      // Observation closest to np
    EMPIRICAL_CDF = 3,            // Empirical cumulative distribution function
    WEIGHTED_AVERAGE_AT_XNP_PLUS1 = 4, // Weighted average at x(np+1)
    EMPIRICAL_CDF_WITH_AVERAGING = 5    // Default - Empirical CDF with averaging
};

// Enumeration for variance definition types
enum class VardefType {
  DF = 0,     // Degrees of freedom (n-1)
    N = 1,      // Population (n)
    WEIGHT = 2, // Sum of weights
    WDF = 3     // Weighted degrees of freedom (sum_weights - 1)
};

// Convert string to PctldefType
PctldefType parsePctldef(int pctldef_int) {
  switch (pctldef_int) {
  case 1: return PctldefType::WEIGHTED_AVERAGE_AT_XNP;
  case 2: return PctldefType::CLOSEST_OBSERVATION;
  case 3: return PctldefType::EMPIRICAL_CDF;
  case 4: return PctldefType::WEIGHTED_AVERAGE_AT_XNP_PLUS1;
  case 5: return PctldefType::EMPIRICAL_CDF_WITH_AVERAGING;
  default: 
    throw std::invalid_argument("Invalid PCTLDEF option: " + std::to_string(pctldef_int));
  }
}

// Convert string to VardefType
VardefType parseVardef(const std::string& vardef_str) {
  if (vardef_str == "DF") return VardefType::DF;
  if (vardef_str == "N") return VardefType::N;
  if (vardef_str == "WEIGHT") return VardefType::WEIGHT;
  if (vardef_str == "WDF") return VardefType::WDF;
  throw std::invalid_argument("Invalid VARDEF option: " + vardef_str);
}

// Comprehensive accumulator for univariate analysis with raw data storage
class UnivariateAccumulator {
private:
  // Moment-based statistics (Welford algorithm)
  double n_obs = 0.0;           // Count of observations
  double sum_weights = 0.0;     // Sum of weights
  double mean = 0.0;            // Running mean
  double M2 = 0.0;              // Sum of squared deviations (for variance)
  double M3 = 0.0;              // Third moment (for skewness)
  double M4 = 0.0;              // Fourth moment (for kurtosis)
  double sum_values = 0.0;      // Weighted sum
  size_t n_missing = 0;         // Count of missing values
  
  // Raw data storage for order-based statistics
  std::vector<double> raw_data;
  std::vector<double> weights;
  bool data_sorted = false;
  
  // Cached order statistics
  mutable std::vector<double> sorted_data;
  mutable std::vector<double> sorted_weights;
  mutable bool cache_valid = false;
  
  // Ensure data is sorted for order-based calculations
  void ensureSorted() const {
    if (!cache_valid || sorted_data.size() != raw_data.size()) {
      // Create pairs of (value, weight) for sorting
      std::vector<std::pair<double, double>> value_weight_pairs;
      value_weight_pairs.reserve(raw_data.size());
      
      for (size_t i = 0; i < raw_data.size(); ++i) {
        value_weight_pairs.emplace_back(raw_data[i], weights[i]);
      }
      
      // Sort by value
      std::sort(value_weight_pairs.begin(), value_weight_pairs.end());
      
      // Extract sorted values and weights
      sorted_data.clear();
      sorted_weights.clear();
      sorted_data.reserve(raw_data.size());
      sorted_weights.reserve(raw_data.size());
      
      for (const auto& pair : value_weight_pairs) {
        sorted_data.push_back(pair.first);
        sorted_weights.push_back(pair.second);
      }
      
      cache_valid = true;
    }
  }
  
public:
  // Update accumulator with new observation using extended Welford algorithm
  void update(double value, double weight = 1.0) {
    if (Rcpp::NumericVector::is_na(value) || !std::isfinite(value)) {
      n_missing++;
      return;
    }
    
    if (weight <= 0.0 || !std::isfinite(weight)) {
      return; // Skip invalid weights
    }
    
    // Store raw data for order-based statistics
    raw_data.push_back(value);
    weights.push_back(weight);
    cache_valid = false; // Invalidate sorted data cache
    
    // Extended Welford algorithm for moments up to 4th order
    double old_sum_weights = sum_weights;
    sum_weights += weight;
    n_obs += 1.0;
    sum_values += weight * value;
    
    double delta = value - mean;
    double delta_w = delta * weight;
    double delta_w2 = delta_w * delta;
    double delta_w3 = delta_w2 * delta;
    double delta_w4 = delta_w3 * delta;
    
    double old_mean = mean;
    mean += delta_w / sum_weights;
    
    // Update higher-order moments
    if (old_sum_weights > 0) {
      double factor = old_sum_weights / sum_weights;
      
      M4 += delta_w4 * factor * (sum_weights * sum_weights - 3 * weight * sum_weights + 3 * weight * weight) / (sum_weights * sum_weights * sum_weights)
        + 6 * delta_w2 * M2 * factor / (sum_weights * sum_weights)
        + 4 * delta_w * M3 * factor / sum_weights;
        
        M3 += delta_w3 * factor * (sum_weights * sum_weights - 3 * weight * sum_weights + 2 * weight * weight) / (sum_weights * sum_weights * sum_weights)
          + 3 * delta_w * M2 * factor / sum_weights;
        
        M2 += delta_w2 * factor;
    }
  }
  
  // Basic getters
  bool hasData() const { return n_obs > 0 && sum_weights > 0.0; }
  double getN() const { return n_obs; }
  size_t getNMiss() const { return n_missing; }
  double getSumWeights() const { return sum_weights; }
  double getMean() const { return hasData() ? mean : std::numeric_limits<double>::quiet_NaN(); }
  double getSum() const { return sum_values; }
  
  // Variance with different divisor types
  double getVariance(VardefType vardef) const {
    if (!hasData() || M2 <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double divisor;
    switch (vardef) {
    case VardefType::DF:
      divisor = sum_weights - 1.0;
      break;
    case VardefType::N:
      divisor = sum_weights;
      break;
    case VardefType::WEIGHT:
      divisor = sum_weights;
      break;
    case VardefType::WDF:
      divisor = sum_weights - 1.0;
      break;
    default:
      divisor = sum_weights - 1.0;
    }
    
    return divisor > 0.0 ? M2 / divisor : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Standard deviation
  double getStdDev(VardefType vardef) const {
    double var = getVariance(vardef);
    return std::isfinite(var) && var >= 0.0 ? std::sqrt(var) : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Skewness (third moment)
  double getSkewness(VardefType vardef) const {
    if (!hasData() || M2 <= 0.0 || sum_weights < 3.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double variance = getVariance(vardef);
    if (!std::isfinite(variance) || variance <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double n = sum_weights;
    double skew_numerator = M3 / n;
    double skew_denominator = std::pow(variance, 1.5);
    
    return skew_denominator > 0.0 ? skew_numerator / skew_denominator : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Kurtosis (fourth moment) - excess kurtosis (subtract 3 for normal distribution = 0)
  double getKurtosis(VardefType vardef) const {
    if (!hasData() || M2 <= 0.0 || sum_weights < 4.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double variance = getVariance(vardef);
    if (!std::isfinite(variance) || variance <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double n = sum_weights;
    double kurt_numerator = M4 / n;
    double kurt_denominator = variance * variance;
    
    double kurtosis = kurt_denominator > 0.0 ? kurt_numerator / kurt_denominator : std::numeric_limits<double>::quiet_NaN();
    
    // Return excess kurtosis (normal distribution = 0)
    return std::isfinite(kurtosis) ? kurtosis - 3.0 : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Calculate quantile using specified PCTLDEF method
  double getQuantile(double p, PctldefType pctldef) const {
    if (!hasData() || p < 0.0 || p > 1.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    ensureSorted();
    
    if (sorted_data.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    size_t n = sorted_data.size();
    if (n == 1) {
      return sorted_data[0];
    }
    
    double result;
    
    switch (pctldef) {
    case PctldefType::WEIGHTED_AVERAGE_AT_XNP: {
      // Method 1: Weighted average at x(np)
      double np = n * p;
      size_t j = static_cast<size_t>(std::floor(np));
      double g = np - j;
      
      if (j == 0) {
        result = sorted_data[0];
      } else if (j >= n) {
        result = sorted_data[n - 1];
      } else {
        result = (1.0 - g) * sorted_data[j - 1] + g * sorted_data[j];
      }
      break;
    }
      
    case PctldefType::CLOSEST_OBSERVATION: {
      // Method 2: Observation closest to np
      double np = n * p + 0.5;
      size_t i = static_cast<size_t>(std::round(np));
      i = std::max(1UL, std::min(i, n)) - 1; // Convert to 0-based index
      result = sorted_data[i];
      break;
    }
      
    case PctldefType::EMPIRICAL_CDF: {
      // Method 3: Empirical cumulative distribution function
      double np = n * p;
      size_t j = static_cast<size_t>(std::ceil(np));
      j = std::max(1UL, std::min(j, n)) - 1; // Convert to 0-based index
      result = sorted_data[j];
      break;
    }
      
    case PctldefType::WEIGHTED_AVERAGE_AT_XNP_PLUS1: {
      // Method 4: Weighted average at x(np+1)
      double np = (n + 1) * p;
      size_t j = static_cast<size_t>(std::floor(np));
      double g = np - j;
      
      if (j == 0) {
        result = sorted_data[0];
      } else if (j >= n) {
        result = sorted_data[n - 1];
      } else {
        result = (1.0 - g) * sorted_data[j - 1] + g * sorted_data[j];
      }
      break;
    }
      
    case PctldefType::EMPIRICAL_CDF_WITH_AVERAGING: {
      // Method 5: Empirical CDF with averaging (SAS default)
      double np = n * p;
      size_t j = static_cast<size_t>(std::floor(np));
      double g = np - j;
      
      if (j == 0) {
        result = sorted_data[0];
      } else if (j >= n) {
        result = sorted_data[n - 1];
      } else {
        if (g == 0.0) {
          // Exact match - average with next observation
          result = (sorted_data[j - 1] + sorted_data[j]) / 2.0;
        } else {
          result = sorted_data[j];
        }
      }
      break;
    }
      
    default:
      result = std::numeric_limits<double>::quiet_NaN();
    }
    
    return result;
  }
  
  // Get minimum and maximum values
  std::pair<double, double> getMinMax() const {
    if (!hasData()) {
      return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    
    ensureSorted();
    return {sorted_data.front(), sorted_data.back()};
  }
  
  // Get extreme observations (lowest and highest values)
  std::vector<double> getExtremeObs(size_t n_extreme) const {
    if (!hasData()) {
      return {};
    }
    
    ensureSorted();
    
    std::vector<double> extremes;
    size_t n_data = sorted_data.size();
    size_t n_each = std::min(n_extreme, n_data / 2);
    
    // Add lowest values
    for (size_t i = 0; i < n_each; ++i) {
      extremes.push_back(sorted_data[i]);
    }
    
    // Add highest values
    for (size_t i = n_data - n_each; i < n_data; ++i) {
      extremes.push_back(sorted_data[i]);
    }
    
    return extremes;
  }
  
  // Calculate trimmed mean
  double getTrimmedMean(double trim_fraction) const {
    if (!hasData() || trim_fraction < 0.0 || trim_fraction >= 0.5) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    ensureSorted();
    
    size_t n_data = sorted_data.size();
    size_t n_trim = static_cast<size_t>(std::floor(n_data * trim_fraction));
    
    if (n_trim * 2 >= n_data) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = n_trim; i < n_data - n_trim; ++i) {
      sum += sorted_data[i] * sorted_weights[i];
      weight_sum += sorted_weights[i];
    }
    
    return weight_sum > 0.0 ? sum / weight_sum : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Calculate Winsorized mean
  double getWinsorizedMean(double trim_fraction) const {
    if (!hasData() || trim_fraction < 0.0 || trim_fraction >= 0.5) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    ensureSorted();
    
    size_t n_data = sorted_data.size();
    size_t n_trim = static_cast<size_t>(std::floor(n_data * trim_fraction));
    
    if (n_trim * 2 >= n_data) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double sum = 0.0;
    double weight_sum = 0.0;
    
    // Winsorize: replace extreme values with boundary values
    double lower_bound = sorted_data[n_trim];
    double upper_bound = sorted_data[n_data - n_trim - 1];
    
    for (size_t i = 0; i < n_data; ++i) {
      double value = sorted_data[i];
      
      // Winsorize the value
      if (value < lower_bound) value = lower_bound;
      if (value > upper_bound) value = upper_bound;
      
      sum += value * sorted_weights[i];
      weight_sum += sorted_weights[i];
    }
    
    return weight_sum > 0.0 ? sum / weight_sum : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Standard error of the mean
  double getStdErr(VardefType vardef) const {
    if (!hasData() || sum_weights <= 1.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    double var = getVariance(vardef);
    if (!std::isfinite(var) || var <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    return std::sqrt(var / sum_weights);
  }
  
  // Get raw data for further analysis
  const std::vector<double>& getRawData() const { return raw_data; }
  const std::vector<double>& getWeights() const { return weights; }
};

// Group key management (same as meansCpp)
class GroupKey {
private:
  std::vector<std::string> values;
  std::size_t hash_value;
  
  void computeHash() {
    hash_value = 0;
    for (const auto& val : values) {
      hash_value ^= std::hash<std::string>{}(val) + 0x9e3779b9 + 
        (hash_value << 6) + (hash_value >> 2);
    }
  }
  
public:
  GroupKey() : hash_value(0) {}
  GroupKey(const std::vector<std::string>& vals) : values(vals) { computeHash(); }
  
  bool operator==(const GroupKey& other) const { return values == other.values; }
  std::size_t getHash() const { return hash_value; }
  const std::vector<std::string>& getValues() const { return values; }
  std::string getValue(size_t index) const {
    return index < values.size() ? values[index] : "";
  }
  size_t size() const { return values.size(); }
};

struct GroupKeyHash {
  std::size_t operator()(const GroupKey& key) const { return key.getHash(); }
};

// Main processor class
class UnivariateProcessor {
private:
  std::vector<std::string> var_names;
  std::vector<std::string> class_names;
  VardefType vardef_type;
  PctldefType pctldef_type;
  double alpha_level;
  double trim_fraction;
  
  using GroupMap = std::unordered_map<GroupKey, std::vector<UnivariateAccumulator>, GroupKeyHash>;
  
  void validateInputs(const Rcpp::DataFrame& data,
                      const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                      const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars) {
    
    if (data.nrows() == 0) {
      throw std::invalid_argument("Input data frame is empty");
    }
    
    Rcpp::CharacterVector data_names = data.names();
    std::vector<std::string> available_names = Rcpp::as<std::vector<std::string>>(data_names);
    
    // Validate analysis variables
    if (var_vars.isNotNull()) {
      Rcpp::CharacterVector vars = Rcpp::as<Rcpp::CharacterVector>(var_vars);
      for (int i = 0; i < vars.size(); ++i) {
        std::string var_name = Rcpp::as<std::string>(vars[i]);
        if (std::find(available_names.begin(), available_names.end(), var_name) == available_names.end()) {
          throw std::invalid_argument("Variable not found in data: " + var_name);
        }
      }
    }
    
    // Validate class variables
    if (class_vars.isNotNull()) {
      Rcpp::CharacterVector classes = Rcpp::as<Rcpp::CharacterVector>(class_vars);
      for (int i = 0; i < classes.size(); ++i) {
        std::string class_name = Rcpp::as<std::string>(classes[i]);
        if (std::find(available_names.begin(), available_names.end(), class_name) == available_names.end()) {
          throw std::invalid_argument("Class variable not found in data: " + class_name);
        }
      }
    }
  }
  
  void setupConfiguration(const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                          const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars,
                          const std::string& vardef,
                          int pctldef,
                          double alpha,
                          double trim) {
    
    if (var_vars.isNotNull()) {
      var_names = Rcpp::as<std::vector<std::string>>(var_vars);
    }
    
    if (class_vars.isNotNull()) {
      class_names = Rcpp::as<std::vector<std::string>>(class_vars);
    }
    
    vardef_type = parseVardef(vardef);
    pctldef_type = parsePctldef(pctldef);
    alpha_level = alpha;
    trim_fraction = trim;
  }
  
  GroupMap processData(const Rcpp::DataFrame& data) {
    GroupMap accumulators;
    
    int n_rows = data.nrows();
    int n_vars = var_names.size();
    
    if (n_vars == 0 || n_rows == 0) {
      return accumulators;
    }
    
    // Pre-extract columns
    std::vector<Rcpp::NumericVector> var_columns;
    std::vector<SEXP> class_columns;
    
    for (const auto& var_name : var_names) {
      var_columns.push_back(data[var_name]);
    }
    
    for (const auto& class_name : class_names) {
      class_columns.push_back(data[class_name]);
    }
    
    // Main processing loop
    for (int row = 0; row < n_rows; ++row) {
      // Build group key
      std::vector<std::string> group_values;
      group_values.reserve(class_names.size());
      
      for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
        SEXP col = class_columns[class_idx];
        std::string value;
        
        if (TYPEOF(col) == STRSXP) {
          Rcpp::CharacterVector char_col(col);
          value = Rcpp::as<std::string>(char_col[row]);
        } else if (TYPEOF(col) == INTSXP) {
          Rcpp::IntegerVector int_col(col);
          value = Rcpp::IntegerVector::is_na(int_col[row]) ? "NA" : std::to_string(int_col[row]);
        } else if (TYPEOF(col) == REALSXP) {
          Rcpp::NumericVector num_col(col);
          value = Rcpp::NumericVector::is_na(num_col[row]) ? "NA" : std::to_string(num_col[row]);
        } else {
          value = "NA";
        }
        
        group_values.push_back(value);
      }
      
      GroupKey group_key(group_values);
      
      // Initialize accumulators for new groups
      if (accumulators.find(group_key) == accumulators.end()) {
        accumulators[group_key].resize(n_vars);
      }
      
      // Update each variable's accumulator
      for (int var_idx = 0; var_idx < n_vars; ++var_idx) {
        double value = var_columns[var_idx][row];
        accumulators[group_key][var_idx].update(value, 1.0);
      }
    }
    
    return accumulators;
  }
  
  // Build comprehensive output following PROC UNIVARIATE structure
  Rcpp::List buildOutput(const GroupMap& group_data) {
    Rcpp::List result;
    
    // Basic Statistics Table
    std::vector<std::string> group_labels;
    std::vector<std::string> variable_names;
    std::vector<double> n_values, mean_values, std_values;
    std::vector<double> skew_values, kurt_values, min_values, max_values;
    
    // Quantiles Table
    std::vector<std::string> quant_group_labels, quant_var_names;
    std::vector<double> q1_values, median_values, q3_values;
    std::vector<double> p10_values, p90_values, p95_values, p99_values;
    
    // Extreme Observations
    std::vector<std::string> extreme_group_labels, extreme_var_names;
    std::vector<std::string> extreme_types;
    std::vector<double> extreme_values;
    
    // Tests for Location
    std::vector<std::string> test_group_labels, test_var_names;
    std::vector<double> t_stat_values, t_prob_values;
    std::vector<double> sign_stat_values, sign_prob_values;
    
    // Robust Estimators
    std::vector<std::string> robust_group_labels, robust_var_names;
    std::vector<double> trimmed_mean_values, winsorized_mean_values;
    
    for (const auto& group_pair : group_data) {
      const GroupKey& group_key = group_pair.first;
      const std::vector<UnivariateAccumulator>& accumulators = group_pair.second;
      
      // Create group label
      std::string group_label;
      if (group_key.size() > 0) {
        std::ostringstream oss;
        for (size_t i = 0; i < group_key.size(); ++i) {
          if (i > 0) oss << " ";
          oss << group_key.getValue(i);
        }
        group_label = oss.str();
      } else {
        group_label = "Overall";
      }
      
      for (size_t var_idx = 0; var_idx < var_names.size(); ++var_idx) {
        const UnivariateAccumulator& acc = accumulators[var_idx];
        const std::string& var_name = var_names[var_idx];
        
        if (!acc.hasData()) continue;
        
        // Basic Statistics
        group_labels.push_back(group_label);
        variable_names.push_back(var_name);
        n_values.push_back(acc.getN());
        mean_values.push_back(acc.getMean());
        std_values.push_back(acc.getStdDev(vardef_type));
        skew_values.push_back(acc.getSkewness(vardef_type));
        kurt_values.push_back(acc.getKurtosis(vardef_type));
        
        auto minmax = acc.getMinMax();
        min_values.push_back(minmax.first);
        max_values.push_back(minmax.second);
        
        // Quantiles
        quant_group_labels.push_back(group_label);
        quant_var_names.push_back(var_name);
        q1_values.push_back(acc.getQuantile(0.25, pctldef_type));
        median_values.push_back(acc.getQuantile(0.50, pctldef_type));
        q3_values.push_back(acc.getQuantile(0.75, pctldef_type));
        p10_values.push_back(acc.getQuantile(0.10, pctldef_type));
        p90_values.push_back(acc.getQuantile(0.90, pctldef_type));
        p95_values.push_back(acc.getQuantile(0.95, pctldef_type));
        p99_values.push_back(acc.getQuantile(0.99, pctldef_type));
        
        // Extreme Observations (5 lowest and 5 highest)
        auto extremes = acc.getExtremeObs(5);
        for (size_t i = 0; i < extremes.size(); ++i) {
          extreme_group_labels.push_back(group_label);
          extreme_var_names.push_back(var_name);
          extreme_types.push_back(i < extremes.size() / 2 ? "Low" : "High");
          extreme_values.push_back(extremes[i]);
        }
        
        // Tests for Location (H0: mu = 0)
        test_group_labels.push_back(group_label);
        test_var_names.push_back(var_name);
        
        double mean_val = acc.getMean();
        double stderr = acc.getStdErr(vardef_type);
        double t_stat = std::numeric_limits<double>::quiet_NaN();
        
        if (std::isfinite(stderr) && stderr > 0.0) {
          t_stat = mean_val / stderr;
        }
        
        t_stat_values.push_back(t_stat);
        
        // T-test p-value (simplified - would need proper t-distribution implementation)
        // For now, use normal approximation for large samples
        double t_prob = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(t_stat) && acc.getN() > 30) {
          // Normal approximation: P(|Z| > |t_stat|) = 2 * (1 - Phi(|t_stat|))
          // Using complementary error function approximation
          double abs_t = std::abs(t_stat);
          t_prob = 2.0 * (1.0 - 0.5 * std::erfc(-abs_t / std::sqrt(2.0)));
        }
        t_prob_values.push_back(t_prob);
        
        // Sign test (simplified)
        const std::vector<double>& raw_data = acc.getRawData();
        double sign_stat = 0.0;
        for (double val : raw_data) {
          if (std::isfinite(val) && val > 0.0) sign_stat += 1.0;
        }
        sign_stat_values.push_back(sign_stat);
        
        // Sign test p-value (binomial approximation)
        double n_valid = acc.getN();
        double sign_prob = std::numeric_limits<double>::quiet_NaN();
        if (n_valid > 10) {
          // Normal approximation to binomial: mean = n/2, var = n/4
          double expected = n_valid / 2.0;
          double variance = n_valid / 4.0;
          double z_score = (sign_stat - expected) / std::sqrt(variance);
          sign_prob = 2.0 * (1.0 - 0.5 * std::erfc(-std::abs(z_score) / std::sqrt(2.0)));
        }
        sign_prob_values.push_back(sign_prob);
        
        // Robust Estimators
        robust_group_labels.push_back(group_label);
        robust_var_names.push_back(var_name);
        trimmed_mean_values.push_back(acc.getTrimmedMean(trim_fraction));
        winsorized_mean_values.push_back(acc.getWinsorizedMean(trim_fraction));
      }
    }
    
    // Build result list with multiple tables (like PROC UNIVARIATE output)
    
    // Basic Statistics Table
    Rcpp::DataFrame basic_stats = Rcpp::DataFrame::create(
      Rcpp::Named("Group") = group_labels,
      Rcpp::Named("Variable") = variable_names,
      Rcpp::Named("N") = n_values,
      Rcpp::Named("Mean") = mean_values,
      Rcpp::Named("Std_Dev") = std_values,
      Rcpp::Named("Skewness") = skew_values,
      Rcpp::Named("Kurtosis") = kurt_values,
      Rcpp::Named("Minimum") = min_values,
      Rcpp::Named("Maximum") = max_values
    );
    
    // Quantiles Table
    Rcpp::DataFrame quantiles = Rcpp::DataFrame::create(
      Rcpp::Named("Group") = quant_group_labels,
      Rcpp::Named("Variable") = quant_var_names,
      Rcpp::Named("P10") = p10_values,
      Rcpp::Named("Q1") = q1_values,
      Rcpp::Named("Median") = median_values,
      Rcpp::Named("Q3") = q3_values,
      Rcpp::Named("P90") = p90_values,
      Rcpp::Named("P95") = p95_values,
      Rcpp::Named("P99") = p99_values
    );
    
    // Extreme Observations
    Rcpp::DataFrame extreme_obs = Rcpp::DataFrame::create(
      Rcpp::Named("Group") = extreme_group_labels,
      Rcpp::Named("Variable") = extreme_var_names,
      Rcpp::Named("Type") = extreme_types,
      Rcpp::Named("Value") = extreme_values
    );
    
    // Tests for Location
    Rcpp::DataFrame location_tests = Rcpp::DataFrame::create(
      Rcpp::Named("Group") = test_group_labels,
      Rcpp::Named("Variable") = test_var_names,
      Rcpp::Named("t_Statistic") = t_stat_values,
      Rcpp::Named("t_Prob") = t_prob_values,
      Rcpp::Named("Sign_Statistic") = sign_stat_values,
      Rcpp::Named("Sign_Prob") = sign_prob_values
    );
    
    // Robust Estimators
    Rcpp::DataFrame robust_stats = Rcpp::DataFrame::create(
      Rcpp::Named("Group") = robust_group_labels,
      Rcpp::Named("Variable") = robust_var_names,
      Rcpp::Named("Trimmed_Mean") = trimmed_mean_values,
      Rcpp::Named("Winsorized_Mean") = winsorized_mean_values
    );
    
    result["BasicStatistics"] = basic_stats;
    result["Quantiles"] = quantiles;
    result["ExtremeObservations"] = extreme_obs;
    result["TestsForLocation"] = location_tests;
    result["RobustEstimators"] = robust_stats;
    
    return result;
  }
  
public:
  Rcpp::List process(const Rcpp::DataFrame& data,
                     const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                     const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars,
                     const std::string& vardef,
                     int pctldef,
                     double alpha,
                     double trim) {
    
    // Phase 1: Validation and setup
    validateInputs(data, var_vars, class_vars);
    setupConfiguration(var_vars, class_vars, vardef, pctldef, alpha, trim);
    
    // Phase 2: Process data into accumulators
    GroupMap group_accumulators = processData(data);
    
    // Phase 3: Build comprehensive output
    return buildOutput(group_accumulators);
  }
};

} // namespace UnivariateCppInternal


//' Comprehensive Univariate Statistical Analysis (SAS PROC UNIVARIATE equivalent)
//'
//' This function performs detailed univariate analysis of numeric variables, providing
//' comprehensive distributional statistics, quantile analysis, robust estimators, 
//' and basic hypothesis tests. It replicates core PROC UNIVARIATE functionality.
//'
//' @param data A data.frame containing the variables to analyze
//' @param var_vars Character vector of numeric variable names for analysis. If NULL,
//'   all numeric variables will be analyzed
//' @param class_vars Character vector of grouping variable names for stratified analysis
//' @param vardef Variance divisor definition. Options:
//'   \itemize{
//'     \item "DF" - Degrees of freedom (n-1), produces unbiased estimator (default)
//'     \item "N" - Population divisor (n), maximum likelihood estimator
//'     \item "WEIGHT" - Sum of weights
//'     \item "WDF" - Weighted degrees of freedom (sum of weights - 1)
//'   }
//' @param pctldef Percentile definition method (1-5, default 5):
//'   \itemize{
//'     \item 1 - Weighted average at x(np) 
//'     \item 2 - Observation closest to np
//'     \item 3 - Empirical cumulative distribution function
//'     \item 4 - Weighted average at x(np+1)
//'     \item 5 - Empirical CDF with averaging (SAS default)
//'   }
//' @param alpha Significance level for confidence intervals and hypothesis tests (default 0.05)
//' @param trim Fraction of observations to trim/Winsorize from each tail (default 0.1)
//'
//' @return A list with multiple components replicating PROC UNIVARIATE output:
//'   \itemize{
//'     \item BasicStatistics - N, mean, std dev, skewness, kurtosis, min, max
//'     \item Quantiles - P10, Q1, Median, Q3, P90, P95, P99
//'     \item ExtremeObservations - Lowest and highest values
//'     \item TestsForLocation - t-test and sign test for H0: mu = 0
//'     \item RobustEstimators - Trimmed and Winsorized means
//'   }
//'
//' @details
//' The function implements extended Welford's algorithm for stable computation of 
//' higher-order moments (up to 4th order for skewness and kurtosis). It supports
//' multiple quantile definitions following Hyndman & Fan (1996) taxonomy, and
//' provides robust location estimators to handle outliers.
//'
//' The output structure mirrors SAS PROC UNIVARIATE, with separate tables for
//' different types of statistics. This facilitates detailed distributional analysis
//' and assumption checking for statistical models.
//'
//' @examples
//' \dontrun{
//' # Basic univariate analysis
//' result <- univariateCpp(mtcars, var_vars = "mpg")
//' 
//' # Grouped analysis with custom settings
//' result <- univariateCpp(mtcars, 
//'                        var_vars = c("mpg", "hp"), 
//'                        class_vars = "cyl",
//'                        pctldef = 4,
//'                        trim = 0.05)
//' 
//' # Access specific output tables
//' basic_stats <- result$BasicStatistics
//' quantiles <- result$Quantiles
//' extremes <- result$ExtremeObservations
//' }
//'
//' @references
//' Hyndman, R.J. and Fan, Y. (1996). Sample quantiles in statistical packages.
//' American Statistician, 50(4), 361-365.
//' 
//' Welford, B.P. (1962). Note on a method for calculating corrected sums of
//' squares and products. Technometrics, 4(3), 419-420.
//'
//' @author Statistical Computing Implementation
//' @export
// [[Rcpp::export]]
Rcpp::List univariateCpp(
   Rcpp::DataFrame data,
   Rcpp::Nullable<Rcpp::CharacterVector> var_vars = R_NilValue,
   Rcpp::Nullable<Rcpp::CharacterVector> class_vars = R_NilValue,
   std::string vardef = "DF",
   int pctldef = 5,
   double alpha = 0.05,
   double trim = 0.1) {
 
 try {
   // Handle default case when var_vars is NULL - find all numeric columns
   Rcpp::CharacterVector final_var_vars;
   if (var_vars.isNull()) {
     Rcpp::CharacterVector data_names = data.names();
     std::vector<std::string> numeric_vars;
     
     for (int i = 0; i < data_names.size(); ++i) {
       std::string col_name = Rcpp::as<std::string>(data_names[i]);
       SEXP col = data[col_name];
       
       // Check if column is numeric
       if (TYPEOF(col) == REALSXP || TYPEOF(col) == INTSXP) {
         // Skip if it's a class variable
         bool is_class_var = false;
         if (class_vars.isNotNull()) {
           Rcpp::CharacterVector class_vec = Rcpp::as<Rcpp::CharacterVector>(class_vars);
           for (int j = 0; j < class_vec.size(); ++j) {
             if (col_name == Rcpp::as<std::string>(class_vec[j])) {
               is_class_var = true;
               break;
             }
           }
         }
         
         if (!is_class_var) {
           numeric_vars.push_back(col_name);
         }
       }
     }
     
     final_var_vars = Rcpp::wrap(numeric_vars);
   } else {
     final_var_vars = Rcpp::as<Rcpp::CharacterVector>(var_vars);
   }
   
   // Create processor and run analysis
   UnivariateCppInternal::UnivariateProcessor processor;
   return processor.process(data, final_var_vars, class_vars, 
                            vardef, pctldef, alpha, trim);
   
 } catch (const std::exception& e) {
   Rcpp::stop("univariateCpp error: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("univariateCpp: Unknown error occurred during processing");
 }
}
