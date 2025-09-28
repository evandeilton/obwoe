#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace MeansCppInternal {

// Enumeration for variance definition types
enum class VardefType {
  DF = 0,     // Degrees of freedom (n-1)
    N = 1,      // Population (n) 
    WEIGHT = 2, // Sum of weights
    WDF = 3     // Weighted degrees of freedom (sum_weights - 1)
};

// Convert string to VardefType
VardefType parseVardef(const std::string& vardef_str) {
  if (vardef_str == "DF") return VardefType::DF;
  if (vardef_str == "N") return VardefType::N; 
  if (vardef_str == "WEIGHT") return VardefType::WEIGHT;
  if (vardef_str == "WDF") return VardefType::WDF;
  throw std::invalid_argument("Invalid VARDEF option: " + vardef_str);
}

// Numerically stable accumulator using Welford's algorithm
class WelfordAccumulator {
private:
  double n_obs = 0.0;           // Count of observations
  double sum_weights = 0.0;     // Sum of weights
  double sum_freq = 0.0;        // Sum of frequencies
  double mean = 0.0;            // Running mean
  double M2 = 0.0;              // Sum of squared deviations
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  double sum_values = 0.0;      // Weighted sum
  double sum_squares = 0.0;     // Uncorrected sum of squares (USS)
  size_t n_missing = 0;         // Count of missing values
  
public:
  // Update accumulator with new observation
  void update(double value, double weight = 1.0, double freq = 1.0) {
    if (Rcpp::NumericVector::is_na(value) || !std::isfinite(value)) {
      n_missing++;
      return;
    }
    
    if (weight <= 0.0 || freq <= 0.0 || !std::isfinite(weight) || !std::isfinite(freq)) {
      return; // Skip invalid weights/frequencies
    }
    
    // Effective weight combines weight and frequency
    double effective_weight = weight * freq;
    double old_sum_weights = sum_weights;
    
    sum_weights += effective_weight;
    sum_freq += freq;
    n_obs += freq;
    
    // Welford's algorithm for numerical stability
    double delta = value - mean;
    double R = (effective_weight * delta) / sum_weights;
    mean += R;
    M2 += old_sum_weights * delta * R;
    
    // Other accumulations
    sum_values += effective_weight * value;
    sum_squares += effective_weight * value * value;
    min_val = std::min(min_val, value);
    max_val = std::max(max_val, value);
  }
  
  // Check if accumulator has any valid observations
  bool hasData() const {
    return sum_weights > 0.0 && n_obs > 0.0;
  }
  
  // Get count of observations
  double getN() const { return n_obs; }
  
  // Get count of missing values  
  size_t getNMiss() const { return n_missing; }
  
  // Get sum of weights
  double getSumWeights() const { return sum_weights; }
  
  // Get mean
  double getMean() const { 
    return hasData() ? mean : std::numeric_limits<double>::quiet_NaN(); 
  }
  
  // Get sum
  double getSum() const { return sum_values; }
  
  // Get minimum value
  double getMin() const {
    return hasData() ? min_val : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Get maximum value
  double getMax() const {
    return hasData() ? max_val : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Get range
  double getRange() const {
    return hasData() ? (max_val - min_val) : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Get uncorrected sum of squares
  double getUSS() const { return sum_squares; }
  
  // Get corrected sum of squares
  double getCSS() const { return M2; }
  
  // Compute variance with specified divisor type
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
    
    if (divisor <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    return M2 / divisor;
  }
  
  // Get standard deviation
  double getStdDev(VardefType vardef) const {
    double var = getVariance(vardef);
    return std::isfinite(var) && var >= 0.0 ? std::sqrt(var) : std::numeric_limits<double>::quiet_NaN();
  }
  
  // Get standard error of mean
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
};

// Group key for efficient hashing and comparison
class GroupKey {
private:
  std::vector<std::string> values;
  std::size_t hash_value;
  
  // Compute hash for the group key
  void computeHash() {
    hash_value = 0;
    for (const auto& val : values) {
      hash_value ^= std::hash<std::string>{}(val) + 0x9e3779b9 + 
        (hash_value << 6) + (hash_value >> 2);
    }
  }
  
public:
  GroupKey() : hash_value(0) {}
  
  GroupKey(const std::vector<std::string>& vals) : values(vals) {
    computeHash();
  }
  
  // Equality operator for unordered_map
  bool operator==(const GroupKey& other) const {
    return values == other.values;
  }
  
  // Get hash value
  std::size_t getHash() const { return hash_value; }
  
  // Get values
  const std::vector<std::string>& getValues() const { return values; }
  
  // Get value at index
  std::string getValue(size_t index) const {
    return index < values.size() ? values[index] : "";
  }
  
  // Get size
  size_t size() const { return values.size(); }
};

// Hash function for GroupKey
struct GroupKeyHash {
  std::size_t operator()(const GroupKey& key) const {
    return key.getHash();
  }
};

// Type mask calculator for SAS _TYPE_ system
class TypeMaskCalculator {
private:
  size_t n_class_vars;
  
public:
  TypeMaskCalculator(size_t n_vars) : n_class_vars(n_vars) {}
  
  // Generate all possible TYPE combinations (power set of class variables)
  std::vector<int> generateAllTypes() const {
    std::vector<int> types;
    int max_type = n_class_vars == 0 ? 0 : (1 << n_class_vars) - 1;
    
    for (int type = 0; type <= max_type; ++type) {
      types.push_back(type);
    }
    
    return types;
  }
  
  // Check if a group key matches a specific TYPE pattern
  bool keyMatchesType(const GroupKey& key, int type_mask) const {
    if (key.size() != n_class_vars) return false;
    
    for (size_t i = 0; i < n_class_vars; ++i) {
      bool bit_set = (type_mask >> i) & 1;
      std::string value = key.getValue(i);
      bool has_specific_value = !value.empty() && value != "_ALL_";
      
      if (bit_set != has_specific_value) return false;
    }
    
    return true;
  }
  
  // Create a group key for a specific TYPE level
  GroupKey createKeyForType(const GroupKey& original_key, int type_mask) const {
    std::vector<std::string> new_values;
    new_values.reserve(n_class_vars);
    
    for (size_t i = 0; i < n_class_vars; ++i) {
      bool include_var = (type_mask >> i) & 1;
      if (include_var) {
        new_values.push_back(original_key.getValue(i));
      } else {
        new_values.push_back("_ALL_");
      }
    }
    
    return GroupKey(new_values);
  }
};

// Main processor class
class MeansProcessor {
private:
  std::vector<std::string> var_names;
  std::vector<std::string> class_names;
  std::vector<std::string> requested_stats;
  VardefType vardef_type;
  std::string weight_var;
  std::string freq_var;
  double alpha_level;
  
  // Type alias for the main data structure
  using GroupMap = std::unordered_map<GroupKey, std::vector<WelfordAccumulator>, GroupKeyHash>;
  
  // Validate input parameters
  void validateInputs(const Rcpp::DataFrame& data,
                      const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                      const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars,
                      const Rcpp::CharacterVector& statistics) {
    
    if (data.nrows() == 0) {
      throw std::invalid_argument("Input data frame is empty");
    }
    
    // Check variable names exist in data
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
  
  // Setup configuration from parameters
  void setupConfiguration(const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                          const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars,
                          const Rcpp::CharacterVector& statistics,
                          const std::string& vardef,
                          const Rcpp::Nullable<Rcpp::String>& weight_var_param,
                          const Rcpp::Nullable<Rcpp::String>& freq_var_param,
                          double alpha) {
    
    // Set up analysis variables (default to all numeric if not specified)
    if (var_vars.isNotNull()) {
      var_names = Rcpp::as<std::vector<std::string>>(var_vars);
    }
    // Note: If var_vars is null, we'd need to detect numeric columns from data
    
    // Set up class variables
    if (class_vars.isNotNull()) {
      class_names = Rcpp::as<std::vector<std::string>>(class_vars);
    }
    
    // Set up requested statistics
    requested_stats = Rcpp::as<std::vector<std::string>>(statistics);
    
    // Parse VARDEF option
    vardef_type = parseVardef(vardef);
    
    // Set up weight and frequency variables
    weight_var = weight_var_param.isNotNull() ? Rcpp::as<std::string>(weight_var_param) : "";
    freq_var = freq_var_param.isNotNull() ? Rcpp::as<std::string>(freq_var_param) : "";
    
    alpha_level = alpha;
  }
  
  // Process data and build accumulators
  GroupMap processData(const Rcpp::DataFrame& data) {
    GroupMap accumulators;
    
    int n_rows = data.nrows();
    int n_vars = var_names.size();
    
    if (n_vars == 0 || n_rows == 0) {
      return accumulators;
    }
    
    // Pre-extract all columns for efficiency
    std::vector<Rcpp::NumericVector> var_columns;
    std::vector<SEXP> class_columns;
    Rcpp::NumericVector weight_column, freq_column;
    
    // Extract analysis variable columns
    for (const auto& var_name : var_names) {
      var_columns.push_back(data[var_name]);
    }
    
    // Extract class variable columns
    for (const auto& class_name : class_names) {
      class_columns.push_back(data[class_name]);
    }
    
    // Extract weight and frequency columns if specified
    if (!weight_var.empty()) {
      weight_column = data[weight_var];
    }
    if (!freq_var.empty()) {
      freq_column = data[freq_var];
    }
    
    // Main processing loop
    for (int row = 0; row < n_rows; ++row) {
      // Build group key for this row
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
          if (Rcpp::IntegerVector::is_na(int_col[row])) {
            value = "NA";
          } else {
            value = std::to_string(int_col[row]);
          }
        } else if (TYPEOF(col) == REALSXP) {
          Rcpp::NumericVector num_col(col);
          if (Rcpp::NumericVector::is_na(num_col[row])) {
            value = "NA";
          } else {
            value = std::to_string(num_col[row]);
          }
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
      
      // Extract weights and frequencies for this row
      double weight = 1.0;
      double freq = 1.0;
      
      if (!weight_var.empty()) {
        weight = weight_column[row];
        if (Rcpp::NumericVector::is_na(weight) || weight <= 0.0) {
          continue; // Skip rows with invalid weights
        }
      }
      
      if (!freq_var.empty()) {
        freq = freq_column[row];
        if (Rcpp::NumericVector::is_na(freq) || freq <= 0.0) {
          continue; // Skip rows with invalid frequencies
        }
      }
      
      // Update each variable's accumulator
      for (int var_idx = 0; var_idx < n_vars; ++var_idx) {
        double value = var_columns[var_idx][row];
        accumulators[group_key][var_idx].update(value, weight, freq);
      }
    }
    
    return accumulators;
  }
  
  // Generate hierarchical results using TYPE system
  std::unordered_map<int, GroupMap> generateHierarchicalResults(const GroupMap& nway_data) {
    std::unordered_map<int, GroupMap> result;
    
    if (class_names.empty()) {
      // No grouping variables - only TYPE 0 (grand total)
      result[0] = nway_data;
      return result;
    }
    
    TypeMaskCalculator type_calc(class_names.size());
    std::vector<int> all_types = type_calc.generateAllTypes();
    
    // Start with the highest TYPE (NWAY) which contains the raw data
    int nway_type = class_names.empty() ? 0 : (1 << class_names.size()) - 1;
    result[nway_type] = nway_data;
    
    // Generate lower TYPE levels by aggregating higher levels
    for (int target_type = nway_type - 1; target_type >= 0; --target_type) {
      GroupMap& target_map = result[target_type];
      
      // Find all groups from higher TYPE levels that should contribute to this TYPE
      for (const auto& source_pair : nway_data) {
        const GroupKey& source_key = source_pair.first;
        const std::vector<WelfordAccumulator>& source_accums = source_pair.second;
        
        // Create the key for this TYPE level
        GroupKey target_key = type_calc.createKeyForType(source_key, target_type);
        
        // Initialize target accumulators if this is a new group
        if (target_map.find(target_key) == target_map.end()) {
          target_map[target_key].resize(source_accums.size());
        }
        
        // Aggregate data from source to target
        // Note: This is a simplified aggregation - proper implementation would
        // need to combine WelfordAccumulator instances properly
        for (size_t var_idx = 0; var_idx < source_accums.size(); ++var_idx) {
          const WelfordAccumulator& source_acc = source_accums[var_idx];
          
          // For now, we'll create a new observation for each aggregated statistic
          // A proper implementation would need a more sophisticated aggregation method
          if (source_acc.hasData()) {
            double mean_val = source_acc.getMean();
            double weight = source_acc.getSumWeights();
            target_map[target_key][var_idx].update(mean_val, weight, 1.0);
          }
        }
      }
    }
    
    return result;
  }
  
  // Build output DataFrame from hierarchical results
  Rcpp::DataFrame buildOutputDataFrame(const std::unordered_map<int, GroupMap>& hierarchical_data) {
    std::vector<int> type_values;
    std::vector<std::vector<std::string>> class_values(class_names.size());
    std::vector<std::string> var_values;
    std::vector<std::string> stat_values;
    std::vector<double> numeric_values;
    
    // Process each TYPE level
    for (const auto& type_pair : hierarchical_data) {
      int type_value = type_pair.first;
      const GroupMap& group_data = type_pair.second;
      
      // Process each group within this TYPE
      for (const auto& group_pair : group_data) {
        const GroupKey& group_key = group_pair.first;
        const std::vector<WelfordAccumulator>& accumulators = group_pair.second;
        
        // Process each variable
        for (size_t var_idx = 0; var_idx < var_names.size(); ++var_idx) {
          const WelfordAccumulator& acc = accumulators[var_idx];
          
          // Generate requested statistics for this variable/group combination
          for (const std::string& stat_name : requested_stats) {
            double stat_value = std::numeric_limits<double>::quiet_NaN();
            
            // Compute the requested statistic
            if (stat_name == "N") {
              stat_value = acc.getN();
            } else if (stat_name == "NMISS") {
              stat_value = static_cast<double>(acc.getNMiss());
            } else if (stat_name == "MEAN") {
              stat_value = acc.getMean();
            } else if (stat_name == "SUM") {
              stat_value = acc.getSum();
            } else if (stat_name == "MIN") {
              stat_value = acc.getMin();
            } else if (stat_name == "MAX") {
              stat_value = acc.getMax();
            } else if (stat_name == "RANGE") {
              stat_value = acc.getRange();
            } else if (stat_name == "USS") {
              stat_value = acc.getUSS();
            } else if (stat_name == "CSS") {
              stat_value = acc.getCSS();
            } else if (stat_name == "VAR") {
              stat_value = acc.getVariance(vardef_type);
            } else if (stat_name == "STD" || stat_name == "STDDEV") {
              stat_value = acc.getStdDev(vardef_type);
            } else if (stat_name == "STDERR") {
              stat_value = acc.getStdErr(vardef_type);
            } else if (stat_name == "SUMWGT") {
              stat_value = acc.getSumWeights();
            }
            
            // Add this observation to output vectors
            type_values.push_back(type_value);
            
            for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
              class_values[class_idx].push_back(group_key.getValue(class_idx));
            }
            
            var_values.push_back(var_names[var_idx]);
            stat_values.push_back(stat_name);
            numeric_values.push_back(stat_value);
          }
        }
      }
    }
    
    // Build the output DataFrame
    Rcpp::List result_list;
    result_list["_TYPE_"] = type_values;
    
    for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
      result_list[class_names[class_idx]] = class_values[class_idx];
    }
    
    result_list["_VAR_"] = var_values;
    result_list["_STAT_"] = stat_values;
    result_list["_VALUE_"] = numeric_values;
    
    return Rcpp::DataFrame(result_list);
  }
  
public:
  // Main processing function
  Rcpp::DataFrame process(const Rcpp::DataFrame& data,
                          const Rcpp::Nullable<Rcpp::CharacterVector>& var_vars,
                          const Rcpp::Nullable<Rcpp::CharacterVector>& class_vars,
                          const Rcpp::CharacterVector& statistics,
                          const std::string& vardef,
                          const Rcpp::Nullable<Rcpp::String>& weight_var,
                          const Rcpp::Nullable<Rcpp::String>& freq_var,
                          double alpha) {
    
    // Phase 1: Validation and setup
    validateInputs(data, var_vars, class_vars, statistics);
    setupConfiguration(var_vars, class_vars, statistics, vardef, weight_var, freq_var, alpha);
    
    // Phase 2: Process data into accumulators
    GroupMap group_accumulators = processData(data);
    
    // Phase 3: Generate hierarchical results
    auto hierarchical_results = generateHierarchicalResults(group_accumulators);
    
    // Phase 4: Build output DataFrame
    return buildOutputDataFrame(hierarchical_results);
  }
};

} // namespace MeansCppInternal


//' Advanced Statistical Summary Processor (SAS PROC MEANS equivalent)
//'
//' This function computes comprehensive descriptive statistics with flexible grouping 
//' capabilities, replicating the core functionality of SAS PROC MEANS. It uses 
//' numerically stable algorithms and supports weighted/frequency analysis.
//'
//' @param data A data.frame containing the variables to analyze
//' @param var_vars Character vector of numeric variable names for analysis. If NULL, 
//'   all numeric variables will be analyzed
//' @param class_vars Character vector of grouping variable names. Creates hierarchical 
//'   summaries using SAS _TYPE_ system
//' @param statistics Character vector of requested statistics. Available options:
//'   "N", "NMISS", "MEAN", "SUM", "MIN", "MAX", "RANGE", "USS", "CSS", "VAR", "STD", "STDERR", "SUMWGT"
//' @param vardef Variance divisor definition. Options:
//'   \itemize{
//'     \item "DF" - Degrees of freedom (n-1), produces unbiased estimator
//'     \item "N" - Population divisor (n), maximum likelihood estimator
//'     \item "WEIGHT" - Sum of weights
//'     \item "WDF" - Weighted degrees of freedom (sum of weights - 1)
//'   }
//' @param weight_var Optional character string naming the weight variable for weighted analysis
//' @param freq_var Optional character string naming the frequency variable (treats each 
//'   observation as representing multiple identical observations)
//' @param alpha Significance level for confidence intervals (currently not implemented)
//'
//' @return A data.frame with computed statistics including:
//'   \itemize{
//'     \item _TYPE_ - Hierarchical grouping level (SAS _TYPE_ system)
//'     \item Class variables - Values of grouping variables
//'     \item _VAR_ - Analysis variable name
//'     \item _STAT_ - Statistic name
//'     \item _VALUE_ - Computed statistic value
//'   }
//'
//' @details
//' The function implements Welford's algorithm for numerically stable variance computation
//' and supports the complete SAS _TYPE_ hierarchical grouping system. When multiple class
//' variables are specified, it generates all possible combinations of grouping levels.
//'
//' @examples
//' \dontrun{
//' # Basic usage - analyze all numeric variables
//' result <- meansCpp(mtcars)
//' 
//' # Analyze specific variables with grouping
//' result <- meansCpp(mtcars, 
//'                   var_vars = c("mpg", "hp"), 
//'                   class_vars = c("cyl", "gear"))
//' 
//' # Weighted analysis with custom statistics
//' result <- meansCpp(data, 
//'                   var_vars = "value", 
//'                   class_vars = "group",
//'                   statistics = c("N", "MEAN", "STD"),
//'                   weight_var = "weights",
//'                   vardef = "DF")
//' }
//'
//' @references
//' Welford, B. P. (1962). Note on a method for calculating corrected sums of 
//' squares and products. Technometrics, 4(3), 419-420.
//'
//' @author Statistical Computing Implementation
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame meansCpp(
   Rcpp::DataFrame data,
   Rcpp::Nullable<Rcpp::CharacterVector> var_vars = R_NilValue,
   Rcpp::Nullable<Rcpp::CharacterVector> class_vars = R_NilValue,
   Rcpp::CharacterVector statistics = Rcpp::CharacterVector::create("N", "MEAN", "STD", "MIN", "MAX"),
   std::string vardef = "DF",
   Rcpp::Nullable<Rcpp::String> weight_var = R_NilValue,
   Rcpp::Nullable<Rcpp::String> freq_var = R_NilValue,
   double alpha = 0.05) {
 
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
         // Skip if it's a class variable, weight variable, or frequency variable
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
         
         bool is_weight_var = false;
         if (weight_var.isNotNull()) {
           is_weight_var = (col_name == Rcpp::as<std::string>(weight_var));
         }
         
         bool is_freq_var = false;
         if (freq_var.isNotNull()) {
           is_freq_var = (col_name == Rcpp::as<std::string>(freq_var));
         }
         
         if (!is_class_var && !is_weight_var && !is_freq_var) {
           numeric_vars.push_back(col_name);
         }
       }
     }
     
     final_var_vars = Rcpp::wrap(numeric_vars);
   } else {
     final_var_vars = Rcpp::as<Rcpp::CharacterVector>(var_vars);
   }
   
   // Create processor and run analysis
   MeansCppInternal::MeansProcessor processor;
   return processor.process(data, final_var_vars, class_vars, statistics, 
                            vardef, weight_var, freq_var, alpha);
   
 } catch (const std::exception& e) {
   Rcpp::stop("meansCpp error: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("meansCpp: Unknown error occurred during processing");
 }
}
