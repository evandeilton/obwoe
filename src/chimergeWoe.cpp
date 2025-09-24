// chimerge_woe.cpp (revised & completed)

#include <Rcpp.h>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <set>
#include <queue>
#include <chrono>
#include <exception>
#include <memory>
#include <limits>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace std;

// =====================================================================
// SECTION 0: LOGGING SYSTEM
// =====================================================================

namespace chimerge_logging {

enum class LogLevel {
  NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4
};

class Logger {
private:
  bool verbose_;
  LogLevel log_level_;
  
public:
  Logger(bool verbose = false, LogLevel level = LogLevel::INFO)
    : verbose_(verbose), log_level_(level) {}
  
  template<typename... Args>
  void error(Args&&... args) {
    if (verbose_ && log_level_ >= LogLevel::ERROR) {
      log("ERROR", std::forward<Args>(args)...);
    }
  }
  
  template<typename... Args>
  void warning(Args&&... args) {
    if (verbose_ && log_level_ >= LogLevel::WARNING) {
      log("WARNING", std::forward<Args>(args)...);
    }
  }
  
  template<typename... Args>
  void info(Args&&... args) {
    if (verbose_ && log_level_ >= LogLevel::INFO) {
      log("INFO", std::forward<Args>(args)...);
    }
  }
  
  template<typename... Args>
  void debug(Args&&... args) {
    if (verbose_ && log_level_ >= LogLevel::DEBUG) {
      log("DEBUG", std::forward<Args>(args)...);
    }
  }
  
  template<typename... Args>
  void log(const std::string& level, Args&&... args) {
    if (!verbose_) return;
    Rcout << "[" << level << "] ";
    log_impl(std::forward<Args>(args)...);
    Rcout << std::endl;
  }
  
  class Timer {
  private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string description_;
    Logger* logger_;
    bool verbose_;
    
  public:
    Timer(Logger* logger, const std::string& description, bool verbose = true)
      : logger_(logger), description_(description),
        verbose_(verbose && logger->is_verbose()) {
      if (verbose_) {
        start_time_ = std::chrono::high_resolution_clock::now();
        logger_->debug("Starting: ", description_);
      }
    }
    
    ~Timer() {
      if (verbose_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time_);
        logger_->debug("Completed: ", description_, " (", duration.count(), " ms)");
      }
    }
  };
  
  bool is_verbose() const { return verbose_; }
  LogLevel get_level() const { return log_level_; }
  
  void set_verbose(bool verbose) { verbose_ = verbose; }
  void set_level(LogLevel level) { log_level_ = level; }
  
private:
  void log_impl() {}
  
  template<typename T, typename... Args>
  void log_impl(T&& t, Args&&... args) {
    Rcout << t;
    log_impl(std::forward<Args>(args)...);
  }
};

static Logger global_logger;

} // namespace chimerge_logging

#define CHIMERGE_LOG_TRACE(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
#define CHIMERGE_LOG_ERROR(...) chimerge_logging::global_logger.error(__VA_ARGS__)
#define CHIMERGE_LOG_WARNING(...) chimerge_logging::global_logger.warning(__VA_ARGS__)
#define CHIMERGE_LOG_INFO(...) chimerge_logging::global_logger.info(__VA_ARGS__)
#define CHIMERGE_LOG_DEBUG(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
#define CHIMERGE_TIMER(description) chimerge_logging::Logger::Timer timer(&chimerge_logging::global_logger, description)

// =====================================================================
// SECTION 1: ENUMS AND CORE DATA STRUCTURES
// =====================================================================

enum class VariableType {
  NUMERIC_CONTINUOUS,
  NUMERIC_DISCRETE,
  CATEGORICAL,
  BOOLEAN,
  DATE_TIME,
  UNSUPPORTED
};

enum class MissingPolicy {
  SEPARATE,
  REMOVE,
  IMPUTE,
  MERGE
};

enum class BinningMethod {
  CHIMERGE,
  CHIMERGE_MOD,
  CHIMERGE_OPT,
  D2,
  HYBRID
};

struct DataPoint {
  double numeric_value;
  std::string categorical_value;
  bool target;
  bool is_missing;
  bool is_special;
  size_t original_index;
  double weight;
  
  DataPoint() : numeric_value(0.0), target(false), is_missing(false),
  is_special(false), original_index(0), weight(1.0) {}
};

struct NumericBin {
  double lower_bound = 0.0;
  double upper_bound = 0.0;
  size_t count_0 = 0;
  size_t count_1 = 0;
  double woe = 0.0;
  double iv_contribution = 0.0;
  std::string label;
  double weight_sum = 0.0;
  
  size_t total() const { return count_0 + count_1; }
  double event_rate() const {
    return total() > 0 ? static_cast<double>(count_1) / static_cast<double>(total()) : 0.0;
  }
};

struct CategoricalBin {
  std::vector<std::string> categories;
  size_t count_0 = 0;
  size_t count_1 = 0;
  double woe = 0.0;
  double iv_contribution = 0.0;
  std::string label;
  double weight_sum = 0.0;
  
  size_t total() const { return count_0 + count_1; }
  double event_rate() const {
    return total() > 0 ? static_cast<double>(count_1) / static_cast<double>(total()) : 0.0;
  }
};

struct SpecialBin {
  std::string type; // "missing", "special_numeric"
  std::vector<std::string> values; // textual representation (e.g., exact numeric as string)
  size_t count_0 = 0;
  size_t count_1 = 0;
  double woe = 0.0;
  double iv_contribution = 0.0;
  std::string label;
  double weight_sum = 0.0;
  
  size_t total() const { return count_0 + count_1; }
  double event_rate() const {
    return total() > 0 ? static_cast<double>(count_1) / static_cast<double>(total()) : 0.0;
  }
};

struct CategoryStats {
  std::string category;
  size_t count_0 = 0;
  size_t count_1 = 0;
  double event_rate = 0.0;
  double weight_sum = 0.0;
  
  size_t total() const { return count_0 + count_1; }
};

struct ValidationResult {
  bool is_valid;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  std::vector<std::string> messages;
  
  void add_error(const std::string& msg) {
    errors.push_back(msg);
    is_valid = false;
  }
  
  void add_warning(const std::string& msg) { warnings.push_back(msg); }
  void add_message(const std::string& msg) { messages.push_back(msg); }
};

// =====================================================================
// SECTION 2: EXCEPTION HANDLING
// =====================================================================

class BinningException : public std::exception {
private:
  std::string message;
public:
  explicit BinningException(const std::string& msg) : message(msg) {}
  const char* what() const noexcept override { return message.c_str(); }
};

// =====================================================================
// SECTION 3: TYPE DETECTION
// =====================================================================

VariableType detect_r_variable_type(SEXP x) {
  CHIMERGE_LOG_TRACE("Detecting variable type");
  
  // Date/DateTime
  if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") ||
      Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
    CHIMERGE_LOG_DEBUG("Detected date/time variable");
    return VariableType::DATE_TIME;
  }
  
  // Factor
  if (Rf_isFactor(x)) {
    SEXP levels = Rf_getAttrib(x, R_LevelsSymbol);
    int n_levels = Rf_length(levels);
    CHIMERGE_LOG_DEBUG("Detected factor with ", n_levels, " levels");
    if (n_levels <= 2) return VariableType::BOOLEAN;
    return VariableType::CATEGORICAL;
  }
  
  // Logical
  if (Rf_isLogical(x)) {
    CHIMERGE_LOG_DEBUG("Detected logical variable");
    return VariableType::BOOLEAN;
  }
  
  // Character
  if (Rf_isString(x)) {
    CharacterVector cv = as<CharacterVector>(x);
    std::set<std::string> unique_vals;
    for (int i = 0; i < cv.size(); ++i) {
      if (!CharacterVector::is_na(cv[i])) unique_vals.insert(as<std::string>(cv[i]));
    }
    CHIMERGE_LOG_DEBUG("Character variable with ", unique_vals.size(), " unique values");
    if (unique_vals.size() <= 2) return VariableType::BOOLEAN;
    return VariableType::CATEGORICAL;
  }
  
  // Integer
  if (Rf_isInteger(x)) {
    IntegerVector iv = as<IntegerVector>(x);
    if (Rf_inherits(x, "integer64")) {
      CHIMERGE_LOG_DEBUG("Detected integer64 variable");
      return VariableType::NUMERIC_CONTINUOUS;
    }
    std::set<int> unique_vals;
    for (int i = 0; i < iv.size(); ++i) {
      if (!IntegerVector::is_na(iv[i])) {
        unique_vals.insert(iv[i]);
        if (unique_vals.size() > 20) break;
      }
    }
    if (unique_vals.size() <= 2) {
      bool is_binary = true;
      for (int v : unique_vals) if (v != 0 && v != 1) { is_binary = false; break; }
      if (is_binary) return VariableType::BOOLEAN;
    }
    return unique_vals.size() <= 20 ? VariableType::NUMERIC_DISCRETE
    : VariableType::NUMERIC_CONTINUOUS;
  }
  
  // Numeric
  if (Rf_isNumeric(x)) {
    NumericVector nv = as<NumericVector>(x);
    std::set<double> unique_vals;
    bool all_integers = true;
    for (int i = 0; i < nv.size(); ++i) {
      if (!NumericVector::is_na(nv[i])) {
        unique_vals.insert(nv[i]);
        if (nv[i] != std::floor(nv[i])) all_integers = false;
        if (unique_vals.size() > 20 && !all_integers) break;
      }
    }
    if (all_integers && unique_vals.size() <= 2) {
      bool is_binary = true;
      for (double v : unique_vals) if (v != 0.0 && v != 1.0) { is_binary = false; break; }
      if (is_binary) return VariableType::BOOLEAN;
    }
    if (all_integers && unique_vals.size() <= 20) return VariableType::NUMERIC_DISCRETE;
    return VariableType::NUMERIC_CONTINUOUS;
  }
  
  CHIMERGE_LOG_WARNING("Unsupported variable type detected");
  return VariableType::UNSUPPORTED;
}

// =====================================================================
// SECTION 4: INPUT VALIDATION
// =====================================================================

class InputValidator {
public:
  static ValidationResult validate_binning_parameters(
      int min_bins, int max_bins, double sig_level, double min_size,
      double smooth, bool monotonic, double min_iv, int digits,
      const std::string& miss_policy, int max_cat, double rare_pct,
      const std::string& method, bool parallel, bool cache) {
    
    CHIMERGE_LOG_INFO("Validating binning parameters");
    ValidationResult result; result.is_valid = true;
    
    if (min_bins < 2) result.add_error("min_bins must be at least 2");
    if (max_bins < min_bins) result.add_error("max_bins must be >= min_bins");
    if (max_bins > 100) result.add_warning("max_bins > 100 may lead to overfitting");
    
    if (digits < 0 || digits > 10) result.add_error("digits must be between 0 and 10");
    if (sig_level <= 0.0 || sig_level >= 1.0) result.add_error("sig_level must be in (0,1)");
    if (min_size <= 0.0 || min_size >= 1.0) result.add_error("min_size must be in (0,1)");
    if (smooth < 0.0) result.add_error("smooth must be non-negative");
    if (min_iv < 0.0) result.add_error("min_iv must be non-negative");
    
    std::set<std::string> valid_policies = {"separate", "remove", "impute", "merge"};
    if (valid_policies.find(miss_policy) == valid_policies.end())
      result.add_error("miss_policy must be one of: separate, remove, impute, merge");
    
    if (max_cat < 2) result.add_error("max_cat must be at least 2");
    if (rare_pct < 0.0 || rare_pct >= 0.5) result.add_error("rare_pct must be in [0,0.5)");
    
    std::set<std::string> valid_methods = {"chimerge", "chimerge_mod", "chimerge_opt", "d2", "hybrid"};
                                           if (valid_methods.find(method) == valid_methods.end())
                                             result.add_error("invalid method: " + method);
                                           
                                           CHIMERGE_LOG_DEBUG("Validation completed - valid: ", result.is_valid);
                                           return result;
  }
};

// =====================================================================
// SECTION 5: CHI-SQUARE CACHE
// =====================================================================

// Map method string to enum and normalize spelling
static BinningMethod parse_method(const std::string& m) {
  std::string s = m;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "chimerge" || s == "chi-merge" || s == "chi") return BinningMethod::CHIMERGE;
  if (s == "chimerge_mod" || s == "chimerge-mod" || s == "chimerge_yates") return BinningMethod::CHIMERGE_MOD;
  if (s == "chimerge_opt" || s == "opt" || s == "optimal") return BinningMethod::CHIMERGE_OPT;
  if (s == "d2" || s == "distance2") return BinningMethod::D2;
  if (s == "hybrid") return BinningMethod::HYBRID;
  // default fallback to modified (safer for small expected counts)
  return BinningMethod::CHIMERGE_MOD;
}


struct PairHash {
  size_t operator()(const std::pair<size_t, size_t>& p) const {
    return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 1);
  }
};

class ChiSquareCache {
private:
  std::unordered_map<std::pair<size_t, size_t>, double, PairHash> cache;
  size_t max_cache_size;
  bool enabled;
  
public:
  ChiSquareCache(size_t max_size = 100000, bool use_cache = true)
    : max_cache_size(max_size), enabled(use_cache) {}
  
  void initialize(size_t) {
    if (!enabled) return;
    CHIMERGE_LOG_DEBUG("Initializing chi-square cache");
    cache.clear();
  }
  
  double get_cached_value(size_t i, size_t j) {
    if (!enabled) return -1.0;
    auto key = std::make_pair(i, j);
    auto it = cache.find(key);
    return (it != cache.end()) ? it->second : -1.0;
  }
  
  // Invalidate any entry that touches indices in [start, end]
  void invalidate_range(size_t start, size_t end) {
    if (!enabled) return;
    CHIMERGE_LOG_TRACE("Invalidating cache range [", start, ", ", end, "]");
    for (auto it = cache.begin(); it != cache.end();) {
      const auto &key = it->first;
      if ((key.first >= start && key.first <= end) ||
          (key.second >= start && key.second <= end) ||
          (key.first > end) || (key.second > end)) {
        it = cache.erase(it);
      } else {
        ++it;
      }
    }
  }
  
  void update_cache(size_t i, size_t j, double value) {
    if (!enabled) return;
    if (cache.size() < max_cache_size) {
      cache[std::make_pair(i, j)] = value;
    }
  }
};

// =====================================================================
// SECTION 6: STATISTICAL FUNCTIONS
// =====================================================================

// A simple D^2 distance between adjacent bins based on event-rate difference,
// scaled by harmonic-like weight to favor merging small similar bins.
// Smaller D2 => more similar => better candidate to merge.
static inline double calculate_d2(const NumericBin& b1, const NumericBin& b2) {
  const double t1 = static_cast<double>(b1.total());
  const double t2 = static_cast<double>(b2.total());
  if (t1 <= 0.0 || t2 <= 0.0) return 0.0; // if any empty, merging is trivial
  const double r1 = b1.event_rate();
  const double r2 = b2.event_rate();
  const double w = (t1 * t2) / (t1 + t2); // harmonic-like weight
  const double diff = (r1 - r2);
  return w * diff * diff;
}

double get_chi_square_threshold(double significance_level, int df) {
  CHIMERGE_LOG_TRACE("Getting chi-square threshold for sig_level: ", significance_level, ", df: ", df);
  if (df != 1) throw BinningException("Only df=1 is supported for chi-square test");
  
  if (significance_level >= 0.25) return 1.323;
  else if (significance_level >= 0.20) return 1.642;
  else if (significance_level >= 0.15) return 2.072;
  else if (significance_level >= 0.10) return 2.706;
  else if (significance_level >= 0.05) return 3.841;
  else if (significance_level >= 0.025) return 5.024;
  else if (significance_level >= 0.01) return 6.635;
  else if (significance_level >= 0.005) return 7.879;
  else if (significance_level >= 0.001) return 10.828;
  else return 15.138;
}

double calculate_chi_square(const NumericBin& bin1, const NumericBin& bin2) {
  double total = static_cast<double>(bin1.total() + bin2.total());
  if (total == 0) return 0.0;
  
  double total_0 = static_cast<double>(bin1.count_0 + bin2.count_0);
  double total_1 = static_cast<double>(bin1.count_1 + bin2.count_1);
  if (total_0 == 0 || total_1 == 0) return 0.0;
  
  double chi_square = 0.0;
  std::vector<std::vector<double>> observed = {
    {static_cast<double>(bin1.count_0), static_cast<double>(bin1.count_1)},
    {static_cast<double>(bin2.count_0), static_cast<double>(bin2.count_1)}
  };
  
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double row_total = observed[i][0] + observed[i][1];
      double col_total = (j == 0) ? total_0 : total_1;
      double expected = (row_total * col_total) / total;
      if (expected > 0) chi_square += std::pow(observed[i][j] - expected, 2.0) / expected;
    }
  }
  CHIMERGE_LOG_TRACE("Chi-square calculated: ", chi_square);
  return chi_square;
}

double calculate_chi_square_modified(const NumericBin& bin1, const NumericBin& bin2) {
  double total = static_cast<double>(bin1.total() + bin2.total());
  if (total == 0) return 0.0;
  
  double total_0 = static_cast<double>(bin1.count_0 + bin2.count_0);
  double total_1 = static_cast<double>(bin1.count_1 + bin2.count_1);
  if (total_0 == 0 || total_1 == 0) return 0.0;
  
  double chi_square = 0.0;
  std::vector<std::vector<double>> observed = {
    {static_cast<double>(bin1.count_0), static_cast<double>(bin1.count_1)},
    {static_cast<double>(bin2.count_0), static_cast<double>(bin2.count_1)}
  };
  
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double row_total = observed[i][0] + observed[i][1];
      double col_total = (j == 0) ? total_0 : total_1;
      double expected = (row_total * col_total) / total;
      
      // Yates' continuity correction
      double diff = std::abs(observed[i][j] - expected) - 0.5;
      if (diff < 0) diff = 0.0;
      double denom = std::max(expected, 0.5);
      chi_square += (diff * diff) / denom;
    }
  }
  CHIMERGE_LOG_TRACE("Modified chi-square calculated: ", chi_square);
  return chi_square;
}

// Likelihood-ratio statistic G^2 (asymptotically chi-square with df=1)
double calculate_g2(const NumericBin& bin1, const NumericBin& bin2) {
  double total = static_cast<double>(bin1.total() + bin2.total());
  if (total == 0) return 0.0;
  
  double n00 = static_cast<double>(bin1.count_0);
  double n01 = static_cast<double>(bin1.count_1);
  double n10 = static_cast<double>(bin2.count_0);
  double n11 = static_cast<double>(bin2.count_1);
  
  double row1 = n00 + n01, row2 = n10 + n11;
  double col1 = n00 + n10, col2 = n01 + n11;
  if (row1 == 0 || row2 == 0 || col1 == 0 || col2 == 0) return 0.0;
  
  double e00 = row1 * col1 / total;
  double e01 = row1 * col2 / total;
  double e10 = row2 * col1 / total;
  double e11 = row2 * col2 / total;
  
  auto term = [](double o, double e) -> double {
    if (o <= 0.0 || e <= 0.0) return 0.0;
    return 2.0 * o * std::log(o / e);
  };
  
  double g2 = term(n00, e00) + term(n01, e01) + term(n10, e10) + term(n11, e11);
  CHIMERGE_LOG_TRACE("G^2 calculated: ", g2);
  return g2;
}

double calculate_correlation(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size() || x.empty()) return 0.0;
  
  double n = static_cast<double>(x.size());
  double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
  
  for (size_t i = 0; i < x.size(); ++i) {
    sum_xy += x[i] * y[i];
    sum_x2 += x[i] * x[i];
    sum_y2 += y[i] * y[i];
  }
  
  double numerator = n * sum_xy - sum_x * sum_y;
  double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
  double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;
  CHIMERGE_LOG_TRACE("Correlation calculated: ", correlation);
  return correlation;
}

// =====================================================================
// SECTION 7: BIN LABEL FORMATTING
// =====================================================================

std::string format_numeric_bin_label(const NumericBin& bin, bool is_first,
                                     bool is_last, int digits) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(digits);
  
  if (is_first && !std::isfinite(bin.lower_bound)) {
    oss << "(-inf," << bin.upper_bound << ")";
  } else if (is_last && !std::isfinite(bin.upper_bound)) {
    oss << "[" << bin.lower_bound << ",+inf)";
  } else {
    oss << "[" << bin.lower_bound << "," << bin.upper_bound << ")";
  }
  std::string label = oss.str();
  CHIMERGE_LOG_TRACE("Formatted numeric bin label: ", label);
  return label;
}

std::string format_categorical_bin_label(const CategoricalBin& bin,
                                         const std::string& separator) {
  if (bin.categories.size() == 1) return bin.categories[0];
  std::ostringstream oss;
  for (size_t i = 0; i < bin.categories.size(); ++i) {
    if (i > 0) oss << separator;
    oss << bin.categories[i];
  }
  std::string label = oss.str();
  CHIMERGE_LOG_TRACE("Formatted categorical bin label: ", label);
  return label;
}

// =====================================================================
// SECTION 8: WOE CALCULATOR (monotonic optional via PAVA on WoE)
// =====================================================================

class WoECalculator {
private:
  double total_positives;
  double total_negatives;
  double laplace_smoothing;
  double total_iv;
  bool enforce_monotonic;
  
public:
  WoECalculator(size_t pos_count, size_t neg_count, double smoothing, bool monotonic)
    : total_positives(static_cast<double>(pos_count)),
      total_negatives(static_cast<double>(neg_count)),
      laplace_smoothing(smoothing), total_iv(0.0),
      enforce_monotonic(monotonic) {}
  
  template<typename BinType>
  void calculate_woe_iv(std::vector<BinType>& bins) {
    CHIMERGE_LOG_INFO("Calculating WoE and IV for ", bins.size(), " bins");
    CHIMERGE_TIMER("WoE/IV calculation");
    
    total_iv = 0.0;
    const size_t num_bins = bins.size();
    if (num_bins == 0) {
      CHIMERGE_LOG_WARNING("Empty bins vector, skipping WoE calculation");
      return;
    }
    
    std::vector<double> woe_values(num_bins);
    std::vector<double> dist_pos(num_bins);
    std::vector<double> dist_neg(num_bins);
    
    double smoothed_total_pos = total_positives + num_bins * laplace_smoothing;
    double smoothed_total_neg = total_negatives + num_bins * laplace_smoothing;
    
    if (smoothed_total_pos == 0.0 || smoothed_total_neg == 0.0) {
      CHIMERGE_LOG_WARNING("Zero total positives or negatives after smoothing");
      for (auto& bin : bins) { bin.woe = 0.0; bin.iv_contribution = 0.0; }
      return;
    }
    
    for (size_t i = 0; i < num_bins; ++i) {
      double smoothed_pos = static_cast<double>(bins[i].count_1) + laplace_smoothing;
      double smoothed_neg = static_cast<double>(bins[i].count_0) + laplace_smoothing;
      
      dist_pos[i] = smoothed_pos / smoothed_total_pos;
      dist_neg[i] = smoothed_neg / smoothed_total_neg;
      
      if (dist_pos[i] <= 0.0 || dist_neg[i] <= 0.0) {
        woe_values[i] = 0.0;
      } else {
        woe_values[i] = std::log(dist_pos[i] / dist_neg[i]);
        if (woe_values[i] > 5.0) woe_values[i] = 5.0;
        if (woe_values[i] < -5.0) woe_values[i] = -5.0;
      }
      CHIMERGE_LOG_DEBUG("Bin ", i, " initial WoE: ", woe_values[i]);
    }
    
    if (enforce_monotonic && bins.size() > 2) {
      CHIMERGE_LOG_DEBUG("Enforcing WoE monotonicity via PAVA");
      apply_pava_to_woe(woe_values, bins);
    }
    
    for (size_t i = 0; i < num_bins; ++i) {
      bins[i].woe = woe_values[i];
      bins[i].iv_contribution = (dist_pos[i] - dist_neg[i]) * bins[i].woe;
      total_iv += bins[i].iv_contribution;
      CHIMERGE_LOG_DEBUG("Bin ", i, " final WoE: ", bins[i].woe,
                         ", IV contribution: ", bins[i].iv_contribution);
    }
    
    CHIMERGE_LOG_INFO("Total IV calculated: ", total_iv);
  }
  
  double get_total_iv() const { return total_iv; }
  
private:
  template<typename BinType>
  void apply_pava_to_woe(std::vector<double>& woe_values,
                         const std::vector<BinType>& bins) {
    size_t n = woe_values.size();
    if (n <= 2) return;
    
    std::vector<double> positions(n), event_rates(n);
    for (size_t i = 0; i < n; ++i) { positions[i] = static_cast<double>(i); event_rates[i] = bins[i].event_rate(); }
    double correlation = calculate_correlation(positions, event_rates);
    bool increasing = (correlation >= 0.0);
    CHIMERGE_LOG_DEBUG("PAVA direction: ", (increasing ? "increasing" : "decreasing"));
    
    std::vector<double> weights(n);
    for (size_t i = 0; i < n; ++i) weights[i] = static_cast<double>(bins[i].total());
    
    if (increasing) apply_pava_increasing(woe_values, weights);
    else apply_pava_decreasing(woe_values, weights);
  }
  
  void apply_pava_increasing(std::vector<double>& values,
                             std::vector<double>& weights) {
    const size_t n = values.size();
    std::vector<double> work_values = values;
    std::vector<double> work_weights = weights;
    std::vector<size_t> blocks(n); for (size_t i = 0; i < n; ++i) blocks[i] = i;
    
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i + 1 < n; ++i) {
        if (blocks[i] == blocks[i + 1]) continue;
        if (work_values[blocks[i]] > work_values[blocks[i + 1]]) {
          double pw = work_weights[blocks[i]] + work_weights[blocks[i + 1]];
          double pv = (work_weights[blocks[i]] * work_values[blocks[i]] +
                       work_weights[blocks[i + 1]] * work_values[blocks[i + 1]]) / pw;
          size_t b0 = blocks[i], b1 = blocks[i + 1];
          work_values[b0] = pv; work_weights[b0] = pw;
          for (size_t j = 0; j < n; ++j) if (blocks[j] == b1) blocks[j] = b0;
          changed = true; break;
        }
      }
    }
    for (size_t i = 0; i < n; ++i) values[i] = work_values[blocks[i]];
  }
  
  void apply_pava_decreasing(std::vector<double>& values,
                             std::vector<double>& weights) {
    const size_t n = values.size();
    std::vector<double> work_values = values;
    std::vector<double> work_weights = weights;
    std::vector<size_t> blocks(n); for (size_t i = 0; i < n; ++i) blocks[i] = i;
    
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i + 1 < n; ++i) {
        if (blocks[i] == blocks[i + 1]) continue;
        if (work_values[blocks[i]] < work_values[blocks[i + 1]]) {
          double pw = work_weights[blocks[i]] + work_weights[blocks[i + 1]];
          double pv = (work_weights[blocks[i]] * work_values[blocks[i]] +
                       work_weights[blocks[i + 1]] * work_values[blocks[i + 1]]) / pw;
          size_t b0 = blocks[i], b1 = blocks[i + 1];
          work_values[b0] = pv; work_weights[b0] = pw;
          for (size_t j = 0; j < n; ++j) if (blocks[j] == b1) blocks[j] = b0;
          changed = true; break;
        }
      }
    }
    for (size_t i = 0; i < n; ++i) values[i] = work_values[blocks[i]];
  }
};


// =====================================================================
// SECTION 8A: MONOTONIC POOLING BY PAVA (BIN MERGING)
// =====================================================================

// Merge-adjacent PAVA to enforce monotonicity on event rates for numeric bins.
// This changes the *counts* (by pooling bins), not just the WoE values, so that
// WoE computed from counts matches the output bins table.
static void enforce_monotonic_pooling(std::vector<NumericBin>& bins) {
  if (bins.size() <= 2) return;
  
  // Pre-merge zero-total bins into neighbors to avoid degenerate blocks
  for (size_t i = 0; i < bins.size();) {
    if (bins[i].total() == 0) {
      if (i > 0) {
        // merge into left
        bins[i-1].upper_bound = bins[i].upper_bound;
        bins[i-1].count_0 += bins[i].count_0;
        bins[i-1].count_1 += bins[i].count_1;
        bins[i-1].weight_sum += bins[i].weight_sum;
        bins.erase(bins.begin() + i);
      } else if (bins.size() > 1) {
        // merge into right
        bins[i+1].lower_bound = bins[i].lower_bound;
        bins[i+1].count_0 += bins[i].count_0;
        bins[i+1].count_1 += bins[i].count_1;
        bins[i+1].weight_sum += bins[i].weight_sum;
        bins.erase(bins.begin() + i);
      } else {
        break;
      }
    } else {
      ++i;
    }
  }
  if (bins.size() <= 2) return;
  
  // Determine direction using correlation of index vs. event rate
  std::vector<double> pos(bins.size()), tot(bins.size()), er(bins.size()), idx(bins.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    pos[i] = static_cast<double>(bins[i].count_1);
    tot[i] = static_cast<double>(bins[i].total());
    er[i]  = (tot[i] > 0.0) ? (pos[i] / tot[i]) : 0.0;
    idx[i] = static_cast<double>(i);
  }
  auto corr = calculate_correlation(idx, er);
  const bool increasing = (corr >= 0.0);
  
  struct Block {
    size_t start, end;
    double c0, c1, wsum;
    double rate() const { double t = c0 + c1; return t > 0.0 ? (c1 / t) : 0.0; }
  };
  
  std::vector<Block> blocks;
  blocks.reserve(bins.size());
  for (size_t i = 0; i < bins.size(); ++i) {
    Block b{ i, i,
             static_cast<double>(bins[i].count_0),
             static_cast<double>(bins[i].count_1),
             bins[i].weight_sum };
    blocks.push_back(b);
    // PAVA pooling condition based on direction
    while (blocks.size() >= 2) {
      auto &b2 = blocks.back();
      auto &b1 = blocks[blocks.size() - 2];
      bool violate = increasing ? (b1.rate() > b2.rate())
        : (b1.rate() < b2.rate());
      if (!violate) break;
      // pool b2 into b1
      b1.end  = b2.end;
      b1.c0  += b2.c0;
      b1.c1  += b2.c1;
      b1.wsum += b2.wsum;
      blocks.pop_back();
    }
  }
  
  // Rebuild pooled bins
  std::vector<NumericBin> pooled;
  pooled.reserve(blocks.size());
  for (const auto& b : blocks) {
    NumericBin nb;
    nb.lower_bound = bins[b.start].lower_bound;
    nb.upper_bound = bins[b.end].upper_bound;
    nb.count_0 = static_cast<size_t>(std::llround(b.c0));
    nb.count_1 = static_cast<size_t>(std::llround(b.c1));
    nb.weight_sum = b.wsum;
    // labels/woe/iv will be (re)computed later
    pooled.push_back(nb);
  }
  
  bins.swap(pooled);
};

// =====================================================================
// SECTION 9: CHIMERGE IMPLEMENTATIONS (NUMERIC)
// =====================================================================

class ChiMergeAlgorithm {
protected:
  int min_bins;
  int max_bins;
  double sig_level;
  double min_size;
  ChiSquareCache cache;
  size_t total_samples;
  
  // Strategy for pair statistic (chi^2, Yates, G^2...)
  std::function<double(const NumericBin&, const NumericBin&)> stat_fn;
  
  void merge_numeric_bins(std::vector<NumericBin>& bins, size_t index) {
    if (index >= bins.size() - 1) return;
    CHIMERGE_LOG_DEBUG("Merging bins ", index, " and ", (index + 1));
    bins[index].upper_bound = bins[index + 1].upper_bound;
    bins[index].count_0 += bins[index + 1].count_0;
    bins[index].count_1 += bins[index + 1].count_1;
    bins[index].weight_sum += bins[index + 1].weight_sum;
    bins.erase(bins.begin() + index + 1);
    cache.invalidate_range(index, bins.size());
  }
  
  bool all_bins_meet_min_size(const std::vector<NumericBin>& bins) const {
    for (const auto& bin : bins) {
      if (static_cast<double>(bin.total()) < min_size * static_cast<double>(total_samples))
        return false;
    }
    return true;
  }
  
  // Stop if: (a) reached min_bins, OR (b) smallest pair stat >= threshold (i.e., independence),
  // otherwise continue. min_size is enforced by continuing to merge until satisfied.
  bool should_stop_merging(const std::vector<NumericBin>& bins, double min_stat) {
    if (bins.size() <= static_cast<size_t>(min_bins)) {
      CHIMERGE_LOG_DEBUG("Stopping merge: bins.size() <= min_bins");
      return true;
    }
    const double threshold = get_chi_square_threshold(sig_level, 1);
    if (min_stat >= threshold && all_bins_meet_min_size(bins)) {
      CHIMERGE_LOG_DEBUG("Stopping merge: min_stat >= threshold and min_size satisfied (",
                         min_stat, " >= ", threshold, ")");
      return true;
    }
    return false;
  }
  
  std::vector<NumericBin> create_initial_numeric_bins(const std::vector<DataPoint>& data) {
    CHIMERGE_LOG_INFO("Creating initial numeric bins from ", data.size(), " data points");
    CHIMERGE_TIMER("Initial bin creation");
    
    std::map<double, NumericBin> unique_bins;
    for (const auto& point : data) {
      if (point.is_missing || point.is_special) continue;
      auto& bin = unique_bins[point.numeric_value];
      bin.lower_bound = point.numeric_value;
      bin.upper_bound = point.numeric_value;
      if (point.target) bin.count_1++; else bin.count_0++;
      bin.weight_sum += point.weight;
    }
    
    std::vector<NumericBin> bins;
    bins.reserve(unique_bins.size());
    for (const auto& kv : unique_bins) bins.push_back(kv.second);
    
    CHIMERGE_LOG_DEBUG("Created ", bins.size(), " initial bins");
    
    // Adjust boundaries to be half-open [low, next_low)
    for (size_t i = 0; i < bins.size(); ++i) {
      if (i == 0) bins[i].lower_bound = -std::numeric_limits<double>::infinity();
      if (i == bins.size() - 1) bins[i].upper_bound = std::numeric_limits<double>::infinity();
      else bins[i].upper_bound = bins[i + 1].lower_bound;
    }
    return bins;
  }
  
public:
  ChiMergeAlgorithm(int min_b, int max_b, double sig, double min_s, bool use_cache,
                    std::function<double(const NumericBin&, const NumericBin&)> stat)
    : min_bins(min_b), max_bins(max_b), sig_level(sig), min_size(min_s),
      cache(100000, use_cache), total_samples(0), stat_fn(std::move(stat)) {}
  
  virtual ~ChiMergeAlgorithm() = default;
  
  virtual std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) = 0;
};

// Standard ChiMerge (Pearson chi-square)
class ChiMergeStandard : public ChiMergeAlgorithm {
public:
  ChiMergeStandard(int min_b, int max_b, double sig, double min_s, bool use_cache)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_chi_square) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying ChiMerge standard algorithm");
    CHIMERGE_TIMER("ChiMerge Standard");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0; for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    cache.initialize(bins.size());
    
    // Merge zero-count neighbors first (stabilizes WoE/IV)
    for (size_t i = 0; i + 1 < bins.size();) {
      if (bins[i].count_0 == 0 || bins[i].count_1 == 0 ||
          bins[i + 1].count_0 == 0 || bins[i + 1].count_1 == 0) {
        merge_numeric_bins(bins, i);
      } else {
        ++i;
      }
    }
    
    // Main loop: merge the adjacent pair with smallest statistic while criteria unmet
    while (bins.size() > static_cast<size_t>(min_bins)) {
      size_t min_index = 0;
      double min_stat = std::numeric_limits<double>::infinity();
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = cache.get_cached_value(i, i + 1);
        if (s < 0.0) { s = stat_fn(bins[i], bins[i + 1]); cache.update_cache(i, i + 1, s); }
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      
      if (should_stop_merging(bins, min_stat)) break;
      
      merge_numeric_bins(bins, min_index);
      if (bins.size() <= static_cast<size_t>(max_bins)) break;
    }
    
    // Force reduction if still above max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t min_index = 0; double min_stat = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = stat_fn(bins[i], bins[i + 1]);
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      merge_numeric_bins(bins, min_index);
    }
    
    CHIMERGE_LOG_INFO("ChiMerge Standard completed with ", bins.size(), " bins");
    return bins;
  }
};

// ChiMerge with Yates' continuity correction
class ChiMergeModified : public ChiMergeAlgorithm {
public:
  ChiMergeModified(int min_b, int max_b, double sig, double min_s, bool use_cache)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_chi_square_modified) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying ChiMerge Modified algorithm");
    CHIMERGE_TIMER("ChiMerge Modified");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0; for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    cache.initialize(bins.size());
    
    // Merge zero-count neighbors
    for (size_t i = 0; i + 1 < bins.size();) {
      if (bins[i].count_0 == 0 || bins[i].count_1 == 0 ||
          bins[i + 1].count_0 == 0 || bins[i + 1].count_1 == 0) {
        merge_numeric_bins(bins, i);
      } else {
        ++i;
      }
    }
    
    while (bins.size() > static_cast<size_t>(min_bins)) {
      size_t min_index = 0;
      double min_stat = std::numeric_limits<double>::infinity();
      
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = cache.get_cached_value(i, i + 1);
        if (s < 0.0) { s = stat_fn(bins[i], bins[i + 1]); cache.update_cache(i, i + 1, s); }
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      
      if (should_stop_merging(bins, min_stat)) break;
      
      merge_numeric_bins(bins, min_index);
      if (bins.size() <= static_cast<size_t>(max_bins)) break;
    }
    
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t min_index = 0; double min_stat = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = stat_fn(bins[i], bins[i + 1]);
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      merge_numeric_bins(bins, min_index);
    }
    
    CHIMERGE_LOG_INFO("ChiMerge Modified completed with ", bins.size(), " bins");
    return bins;
  }
};

// ---------------------------------------------------------------------
// D2 distance-based merging (ignores chi-square threshold; purely proximity)
class ChiMergeD2 : public ChiMergeAlgorithm {
public:
  ChiMergeD2(int min_b, int max_b, double sig, double min_s, bool use_cache)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_d2) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying D2 distance merging");
    CHIMERGE_TIMER("ChiMerge D2");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0;
    for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    // Greedy: always merge pair with smallest D2 until reaching max_bins
    while (bins.size() > static_cast<size_t>(max_bins) && bins.size() > 2) {
      size_t min_idx = 0;
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double d2 = stat_fn(bins[i], bins[i + 1]);
        if (d2 < best) { best = d2; min_idx = i; }
      }
      merge_numeric_bins(bins, min_idx);
    }
    
    // Additional merging if needed to meet min_bins constraint
    while (bins.size() > static_cast<size_t>(min_bins) && bins.size() > 2) {
      size_t min_idx = 0;
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double d2 = stat_fn(bins[i], bins[i + 1]);
        if (d2 < best) { best = d2; min_idx = i; }
      }
      merge_numeric_bins(bins, min_idx);
    }
    
    CHIMERGE_LOG_INFO("ChiMerge D2 completed with ", bins.size(), " bins");
    return bins;
  }
};

// ---------------------------------------------------------------------
// Hybrid: use D2 to quickly reduce, then finish with standard chi-square
class ChiMergeHybrid : public ChiMergeAlgorithm {
public:
  ChiMergeHybrid(int min_b, int max_b, double sig, double min_s, bool use_cache)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_chi_square) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying Hybrid (D2 pre-merge + ChiMerge standard)");
    CHIMERGE_TIMER("ChiMerge Hybrid");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0;
    for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    // Stage 1: D2 until a relaxed target (e.g., 2*max_bins)
    size_t target1 = std::max(static_cast<size_t>(max_bins) * 2, static_cast<size_t>(max_bins + 1));
    while (bins.size() > target1 && bins.size() > 2) {
      size_t min_idx = 0;
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double d2 = calculate_d2(bins[i], bins[i + 1]);
        if (d2 < best) { best = d2; min_idx = i; }
      }
      merge_numeric_bins(bins, min_idx);
    }
    
    // Stage 2: Standard ChiMerge to fine-tune up to max_bins / threshold
    cache.initialize(bins.size());
    while (bins.size() > static_cast<size_t>(min_bins) && bins.size() > 2) {
      size_t min_idx = 0;
      double min_chi = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double chi = cache.get_cached_value(i, i + 1);
        if (chi < 0) {
          chi = stat_fn(bins[i], bins[i + 1]);
          cache.update_cache(i, i + 1, chi);
        }
        if (chi < min_chi) { min_chi = chi; min_idx = i; }
      }
      if (should_stop_merging(bins, min_chi)) break;
      merge_numeric_bins(bins, min_idx);
      if (bins.size() <= static_cast<size_t>(max_bins)) break;
    }
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t min_idx = 0; double min_chi = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double chi = stat_fn(bins[i], bins[i + 1]);
        if (chi < min_chi) { min_chi = chi; min_idx = i; }
      }
      merge_numeric_bins(bins, min_idx);
    }
    return bins;
  }
};

// ---------------------------------------------------------------------
// "Optimal" flavor: ignore significance threshold and *force* exactly max_bins
// based on the chosen chi-square (here we use modified/Yates for robustness).
class ChiMergeOptimal : public ChiMergeAlgorithm {
public:
  ChiMergeOptimal(int min_b, int max_b, double sig, double min_s, bool use_cache):
  ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_chi_square_modified) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying ChiMerge Optimal (force max_bins, Yates)");
    CHIMERGE_TIMER("ChiMerge Optimal");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0;
    for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(max_bins)) return bins;
    
    // Greedy global: keep merging min chi-square (modified) until exactly max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t min_idx = 0; double min_chi = std::numeric_limits<double>::max();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double chi = stat_fn(bins[i], bins[i + 1]);
        if (chi < min_chi) { min_chi = chi; min_idx = i; }
      }
      merge_numeric_bins(bins, min_idx);
    }
    return bins;
  }
};


// Optimized ChiMerge using a priority queue of adjacent pairs
class ChiMergeOptimized : public ChiMergeAlgorithm {
  struct Pair {
    size_t left;
    size_t right;
    double stat;
    bool operator<(const Pair& other) const { return stat > other.stat; } // min-heap
  };
  
public:
  ChiMergeOptimized(int min_b, int max_b, double sig, double min_s, bool use_cache, bool yates)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache,
      yates ? calculate_chi_square_modified : calculate_chi_square) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying ChiMerge Optimized algorithm");
    CHIMERGE_TIMER("ChiMerge Optimized");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0; for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    auto recompute_pair = [&](size_t i) -> Pair {
      Pair p; p.left = i; p.right = i + 1; p.stat = stat_fn(bins[i], bins[i + 1]); return p;
    };
    
    std::priority_queue<Pair> pq;
    for (size_t i = 0; i + 1 < bins.size(); ++i) pq.push(recompute_pair(i));
    
    while (bins.size() > static_cast<size_t>(min_bins)) {
      // Extract current minimal pair; due to stale indices, we may need to skip
      Pair best;
      bool found = false;
      while (!pq.empty()) {
        best = pq.top(); pq.pop();
        if (best.left < bins.size() - 1 && best.right == best.left + 1) {
          // Re-evaluate due to possible staleness
          double current = stat_fn(bins[best.left], bins[best.right]);
          if (std::abs(current - best.stat) > 1e-12) {
            best.stat = current; pq.push(best);
            continue;
          }
          found = true; break;
        }
      }
      if (!found) break;
      
      if (should_stop_merging(bins, best.stat)) break;
      
      // Merge the best pair
      size_t merge_index = best.left;
      merge_numeric_bins(bins, merge_index);
      
      // Push affected neighbors
      if (merge_index > 0) pq.push(recompute_pair(merge_index - 1));
      if (merge_index + 1 < bins.size()) pq.push(recompute_pair(merge_index));
      if (bins.size() <= static_cast<size_t>(max_bins)) break;
    }
    
    while (bins.size() > static_cast<size_t>(max_bins)) {
      // Fallback: linear scan to reduce further
      size_t min_index = 0; double min_stat = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = stat_fn(bins[i], bins[i + 1]);
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      merge_numeric_bins(bins, min_index);
    }
    
    CHIMERGE_LOG_INFO("ChiMerge Optimized completed with ", bins.size(), " bins");
    return bins;
  }
};

// D2 method implemented via G^2 (likelihood-ratio) merges
class D2Merge : public ChiMergeAlgorithm {
public:
  D2Merge(int min_b, int max_b, double sig, double min_s, bool use_cache)
    : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache, calculate_g2) {}
  
  std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
    CHIMERGE_LOG_INFO("Applying D2 (G^2) merging");
    CHIMERGE_TIMER("D2 (G^2) Merge");
    
    auto bins = create_initial_numeric_bins(data);
    total_samples = 0; for (const auto& b : bins) total_samples += b.total();
    if (bins.size() <= static_cast<size_t>(min_bins)) return bins;
    
    while (bins.size() > static_cast<size_t>(min_bins)) {
      size_t min_index = 0; double min_stat = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = stat_fn(bins[i], bins[i + 1]); // G^2
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      if (should_stop_merging(bins, min_stat)) break;
      merge_numeric_bins(bins, min_index);
      if (bins.size() <= static_cast<size_t>(max_bins)) break;
    }
    
    while (bins.size() > static_cast<size_t>(max_bins)) {
      size_t min_index = 0; double min_stat = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double s = stat_fn(bins[i], bins[i + 1]);
        if (s < min_stat) { min_stat = s; min_index = i; }
      }
      merge_numeric_bins(bins, min_index);
    }
    
    CHIMERGE_LOG_INFO("D2 completed with ", bins.size(), " bins");
    return bins;
  }
};

// =====================================================================
// SECTION 10: CATEGORICAL BINNING
// =====================================================================

std::vector<CategoryStats> calculate_category_statistics(const std::vector<DataPoint>& data) {
  CHIMERGE_LOG_INFO("Calculating category statistics for ", data.size(), " data points");
  CHIMERGE_TIMER("Category statistics");
  
  std::map<std::string, CategoryStats> stats_map;
  for (const auto& point : data) {
    if (point.is_missing || point.is_special) continue;
    auto& s = stats_map[point.categorical_value];
    s.category = point.categorical_value;
    if (point.target) s.count_1++; else s.count_0++;
    s.weight_sum += point.weight;
  }
  
  std::vector<CategoryStats> stats_vector; stats_vector.reserve(stats_map.size());
  for (const auto& kv : stats_map) {
    CategoryStats s = kv.second;
    s.event_rate = s.total() > 0 ? static_cast<double>(s.count_1) / static_cast<double>(s.total()) : 0.0;
    stats_vector.push_back(s);
  }
  
  std::sort(stats_vector.begin(), stats_vector.end(),
            [](const CategoryStats& a, const CategoryStats& b) {
              return a.event_rate < b.event_rate;
            });
  
  CHIMERGE_LOG_DEBUG("Calculated statistics for ", stats_vector.size(), " categories");
  return stats_vector;
}

std::vector<CategoricalBin> apply_categorical_binning(
    const std::vector<DataPoint>& data,
    int /*min_bins*/, int max_bins, double /*min_size*/,
    int max_cat, double rare_pct, const std::string& cat_sep) {
  
  CHIMERGE_LOG_INFO("Applying categorical binning");
  CHIMERGE_TIMER("Categorical binning");
  
  auto category_stats = calculate_category_statistics(data);
  
  size_t total_count = 0;
  for (const auto& s : category_stats) total_count += s.total();
  CHIMERGE_LOG_DEBUG("Total count: ", total_count);
  
  std::vector<CategoricalBin> bins;
  CategoricalBin rare_bin;
  
  for (const auto& s : category_stats) {
    double frequency = total_count > 0 ? static_cast<double>(s.total()) / static_cast<double>(total_count) : 0.0;
    if (frequency < rare_pct && (bins.size() + 1) > static_cast<size_t>(max_cat)) {
      rare_bin.categories.push_back(s.category);
      rare_bin.count_0 += s.count_0;
      rare_bin.count_1 += s.count_1;
      rare_bin.weight_sum += s.weight_sum;
      CHIMERGE_LOG_DEBUG("Adding category ", s.category, " to rare bin");
    } else {
      CategoricalBin b;
      b.categories.push_back(s.category);
      b.count_0 = s.count_0; b.count_1 = s.count_1; b.weight_sum = s.weight_sum;
      bins.push_back(b);
      CHIMERGE_LOG_DEBUG("Creating bin for category ", s.category);
    }
  }
  
  if (!rare_bin.categories.empty() && rare_bin.total() > 0) {
    bins.push_back(rare_bin);
    CHIMERGE_LOG_DEBUG("Added rare bin with ", rare_bin.categories.size(), " categories");
  }
  
  // Merge closest adjacent bins by chi-square until <= max_bins
  while (bins.size() > static_cast<size_t>(max_bins)) {
    double min_chi = std::numeric_limits<double>::infinity();
    size_t merge_index = 0;
    for (size_t i = 0; i + 1 < bins.size(); ++i) {
      NumericBin t1, t2;
      t1.count_0 = bins[i].count_0; t1.count_1 = bins[i].count_1;
      t2.count_0 = bins[i + 1].count_0; t2.count_1 = bins[i + 1].count_1;
      double chi = calculate_chi_square(t1, t2);
      if (chi < min_chi) { min_chi = chi; merge_index = i; }
    }
    bins[merge_index].categories.insert(bins[merge_index].categories.end(),
                                        bins[merge_index + 1].categories.begin(),
                                        bins[merge_index + 1].categories.end());
    bins[merge_index].count_0 += bins[merge_index + 1].count_0;
    bins[merge_index].count_1 += bins[merge_index + 1].count_1;
    bins[merge_index].weight_sum += bins[merge_index + 1].weight_sum;
    bins.erase(bins.begin() + merge_index + 1);
  }
  
  for (auto& b : bins) b.label = format_categorical_bin_label(b, cat_sep);
  
  CHIMERGE_LOG_INFO("Categorical binning completed with ", bins.size(), " bins");
  return bins;
}

// =====================================================================
// SECTION 11: SPECIAL VALUE HANDLING
// =====================================================================

class SpecialValueHandler {
private:
  MissingPolicy miss_policy;
  std::vector<double> special_numeric;
  std::vector<std::string> special_string;
  
  static std::string num_to_string(double v) {
    std::ostringstream oss;
    oss << std::setprecision(15) << v;
    return oss.str();
  }
  
public:
  SpecialValueHandler(const std::string& policy,
                      const std::vector<double>& special_num,
                      const std::vector<std::string>& special_str) {
    if (policy == "separate") miss_policy = MissingPolicy::SEPARATE;
    else if (policy == "remove") miss_policy = MissingPolicy::REMOVE;
    else if (policy == "impute") miss_policy = MissingPolicy::IMPUTE;
    else miss_policy = MissingPolicy::MERGE;
    
    special_numeric = special_num;
    special_string = special_str;
    
    CHIMERGE_LOG_DEBUG("SpecialValueHandler initialized with policy: ", policy);
  }
  
  std::vector<SpecialBin> process_special_values(const std::vector<DataPoint>& data) {
    CHIMERGE_LOG_INFO("Processing special values");
    CHIMERGE_TIMER("Special value processing");
    
    std::vector<SpecialBin> special_bins;
    if (miss_policy != MissingPolicy::SEPARATE) {
      CHIMERGE_LOG_DEBUG("Missing policy not 'separate', skipping special bin creation");
      return special_bins;
    }
    
    // Missing
    SpecialBin missing_bin;
    missing_bin.type = "missing"; missing_bin.label = "[Missing]";
    missing_bin.count_0 = missing_bin.count_1 = 0; missing_bin.weight_sum = 0.0;
    
    for (const auto& p : data) {
      if (p.is_missing) {
        if (p.target) missing_bin.count_1++; else missing_bin.count_0++;
        missing_bin.weight_sum += p.weight;
      }
    }
    if (missing_bin.total() > 0) {
      special_bins.push_back(missing_bin);
      CHIMERGE_LOG_DEBUG("Created missing bin with ", missing_bin.total(), " samples");
    }
    
    // Special numeric values (one bin per special value to preserve mapping)
    for (double sval : special_numeric) {
      SpecialBin sb;
      sb.type = "special_numeric";
      sb.label = "[Special:" + num_to_string(sval) + "]";
      sb.count_0 = sb.count_1 = 0; sb.weight_sum = 0.0;
      sb.values.push_back(num_to_string(sval));
      
      for (const auto& p : data) {
        if (p.is_special && std::fabs(p.numeric_value - sval) < 1e-12) {
          if (p.target) sb.count_1++; else sb.count_0++;
          sb.weight_sum += p.weight;
        }
      }
      if (sb.total() > 0) {
        special_bins.push_back(sb);
        CHIMERGE_LOG_DEBUG("Created special bin for value ", sval, " with ", sb.total(), " samples");
      }
    }
    
    CHIMERGE_LOG_INFO("Processed ", special_bins.size(), " special bins");
    return special_bins;
  }
};

// =====================================================================
// SECTION 12: DATA PREPARATION AND EXTRACTION
// =====================================================================

std::vector<DataPoint> prepare_numeric_data(SEXP x, const LogicalVector& target,
                                            const NumericVector& weights,
                                            const std::vector<double>& special_vals) {
  CHIMERGE_LOG_INFO("Preparing numeric data");
  CHIMERGE_TIMER("Numeric data preparation");
  
  std::vector<DataPoint> data;
  bool has_weights = weights.size() > 0;
  
  if (Rf_isInteger(x)) {
    IntegerVector iv = as<IntegerVector>(x);
    for (int i = 0; i < iv.size(); ++i) {
      DataPoint p; p.target = target[i]; p.original_index = i;
      p.weight = has_weights ? weights[i] : 1.0;
      if (IntegerVector::is_na(iv[i])) {
        p.is_missing = true;
      } else {
        p.numeric_value = static_cast<double>(iv[i]);
        for (double s : special_vals) if (std::fabs(p.numeric_value - s) < 1e-12) { p.is_special = true; break; }
      }
      data.push_back(p);
    }
  } else if (Rf_isNumeric(x)) {
    NumericVector nv = as<NumericVector>(x);
    for (int i = 0; i < nv.size(); ++i) {
      DataPoint p; p.target = target[i]; p.original_index = i;
      p.weight = has_weights ? weights[i] : 1.0;
      if (NumericVector::is_na(nv[i])) {
        p.is_missing = true;
      } else {
        p.numeric_value = nv[i];
        for (double s : special_vals) if (std::fabs(p.numeric_value - s) < 1e-12) { p.is_special = true; break; }
      }
      data.push_back(p);
    }
  }
  
  CHIMERGE_LOG_INFO("Prepared ", data.size(), " numeric data points");
  return data;
}

std::vector<DataPoint> prepare_categorical_data(SEXP x, const LogicalVector& target,
                                                const NumericVector& weights) {
  CHIMERGE_LOG_INFO("Preparing categorical data");
  CHIMERGE_TIMER("Categorical data preparation");
  
  std::vector<DataPoint> data;
  bool has_weights = weights.size() > 0;
  
  if (Rf_isFactor(x)) {
    IntegerVector iv = as<IntegerVector>(x);
    CharacterVector levels = iv.attr("levels");
    for (int i = 0; i < iv.size(); ++i) {
      DataPoint p; p.target = target[i]; p.original_index = i;
      p.weight = has_weights ? weights[i] : 1.0;
      if (IntegerVector::is_na(iv[i])) {
        p.is_missing = true;
      } else {
        int level_idx = iv[i] - 1;
        if (level_idx >= 0 && level_idx < levels.size()) {
          p.categorical_value = as<std::string>(levels[level_idx]);
        } else {
          p.is_missing = true;
        }
      }
      data.push_back(p);
    }
  } else if (Rf_isString(x)) {
    CharacterVector cv = as<CharacterVector>(x);
    for (int i = 0; i < cv.size(); ++i) {
      DataPoint p; p.target = target[i]; p.original_index = i;
      p.weight = has_weights ? weights[i] : 1.0;
      if (CharacterVector::is_na(cv[i])) {
        p.is_missing = true;
      } else {
        p.categorical_value = as<std::string>(cv[i]);
      }
      data.push_back(p);
    }
  }
  
  CHIMERGE_LOG_INFO("Prepared ", data.size(), " categorical data points");
  return data;
}

// =====================================================================
// SECTION 13: TRANSFORMATION APPLICATION
// =====================================================================

static inline bool label_matches_special_value(const SpecialBin& sbin, double val) {
  // sbin.label format: [Special:<number>]
  if (sbin.type != "special_numeric") return false;
  if (sbin.values.empty()) return false;
  std::ostringstream oss; oss << std::setprecision(15) << val;
  return sbin.values[0] == oss.str();
}

NumericVector apply_numeric_woe_transform(SEXP x, const std::vector<NumericBin>& bins,
                                          const std::vector<SpecialBin>& special_bins,
                                          const std::vector<double>& special_vals) {
  CHIMERGE_LOG_INFO("Applying numeric WoE transformation");
  CHIMERGE_TIMER("Numeric WoE transform");
  
  NumericVector result;
  
  auto assign_special = [&](int i, double val) {
    // Find matching special bin by exact value string
    for (const auto& sb : special_bins) {
      if (sb.type == "special_numeric" && label_matches_special_value(sb, val)) {
        result[i] = sb.woe; return true;
      }
    }
    // If not found, fall back to any "special_numeric" (rare)
    for (const auto& sb : special_bins) if (sb.type == "special_numeric") { result[i] = sb.woe; return true; }
    return false;
  };
  
  auto assign_missing = [&](int i) {
    for (const auto& sb : special_bins) if (sb.type == "missing") { result[i] = sb.woe; return true; }
    return false;
  };
  
  if (Rf_isInteger(x)) {
    IntegerVector iv = as<IntegerVector>(x);
    result = NumericVector(iv.size(), NA_REAL);
    for (int i = 0; i < iv.size(); ++i) {
      if (IntegerVector::is_na(iv[i])) { assign_missing(i); continue; }
      double val = static_cast<double>(iv[i]);
      bool handled = false;
      for (double s : special_vals) if (std::fabs(val - s) < 1e-12) { handled = assign_special(i, s); break; }
      if (handled) continue;
      for (const auto& b : bins) if (val >= b.lower_bound && val < b.upper_bound) { result[i] = b.woe; break; }
    }
  } else if (Rf_isNumeric(x)) {
    NumericVector nv = as<NumericVector>(x);
    result = NumericVector(nv.size(), NA_REAL);
    for (int i = 0; i < nv.size(); ++i) {
      if (NumericVector::is_na(nv[i])) { assign_missing(i); continue; }
      double val = nv[i];
      bool handled = false;
      for (double s : special_vals) if (std::fabs(val - s) < 1e-12) { handled = assign_special(i, s); break; }
      if (handled) continue;
      for (const auto& b : bins) if (val >= b.lower_bound && val < b.upper_bound) { result[i] = b.woe; break; }
    }
  }
  
  CHIMERGE_LOG_INFO("Applied WoE transformation to ", result.size(), " values");
  return result;
}

CharacterVector apply_numeric_bin_transform(SEXP x, const std::vector<NumericBin>& bins,
                                            const std::vector<SpecialBin>& special_bins,
                                            const std::vector<double>& special_vals) {
  CHIMERGE_LOG_INFO("Applying numeric bin transformation");
  CHIMERGE_TIMER("Numeric bin transform");
  
  CharacterVector result;
  
  auto label_special = [&](double val) -> std::string {
    std::ostringstream oss; oss << "[Special:" << std::setprecision(15) << val << "]";
    return oss.str();
  };
  
  if (Rf_isInteger(x)) {
    IntegerVector iv = as<IntegerVector>(x);
    result = CharacterVector(iv.size());
    for (int i = 0; i < iv.size(); ++i) {
      if (IntegerVector::is_na(iv[i])) { result[i] = "[Missing]"; continue; }
      double val = static_cast<double>(iv[i]);
      bool done = false;
      for (double s : special_vals) if (std::fabs(val - s) < 1e-12) { result[i] = label_special(s); done = true; break; }
      if (done) continue;
      for (const auto& b : bins) if (val >= b.lower_bound && val < b.upper_bound) { result[i] = b.label; break; }
    }
  } else if (Rf_isNumeric(x)) {
    NumericVector nv = as<NumericVector>(x);
    result = CharacterVector(nv.size());
    for (int i = 0; i < nv.size(); ++i) {
      if (NumericVector::is_na(nv[i])) { result[i] = "[Missing]"; continue; }
      double val = nv[i];
      bool done = false;
      for (double s : special_vals) if (std::fabs(val - s) < 1e-12) { result[i] = label_special(s); done = true; break; }
      if (done) continue;
      for (const auto& b : bins) if (val >= b.lower_bound && val < b.upper_bound) { result[i] = b.label; break; }
    }
  }
  
  CHIMERGE_LOG_INFO("Applied bin transformation to ", result.size(), " values");
  return result;
}

// =====================================================================
// SECTION 14: KS AND GINI CALCULATION
// =====================================================================

void calculate_ks_gini_for_bins(std::vector<double>& ks_values,
                                std::vector<double>& gini_values,
                                const std::vector<int>& neg_counts,
                                const std::vector<int>& pos_counts) {
  CHIMERGE_LOG_INFO("Calculating KS and Gini values");
  CHIMERGE_TIMER("KS/Gini calculation");
  
  size_t n_bins = neg_counts.size();
  if (n_bins == 0) {
    CHIMERGE_LOG_WARNING("Empty bins vector, skipping KS/Gini calculation");
    return;
  }
  
  double total_neg = 0.0, total_pos = 0.0;
  for (size_t i = 0; i < n_bins; ++i) {
    total_neg += static_cast<double>(neg_counts[i]);
    total_pos += static_cast<double>(pos_counts[i]);
  }
  CHIMERGE_LOG_DEBUG("Total negatives: ", total_neg, ", positives: ", total_pos);
  
  double cum_neg = 0.0, cum_pos = 0.0;
  for (size_t i = 0; i < n_bins; ++i) {
    cum_neg += static_cast<double>(neg_counts[i]);
    cum_pos += static_cast<double>(pos_counts[i]);
    double cum_neg_rate = total_neg > 0.0 ? cum_neg / total_neg : 0.0;
    double cum_pos_rate = total_pos > 0.0 ? cum_pos / total_pos : 0.0;
    
    double ks = std::fabs(cum_pos_rate - cum_neg_rate);
    ks_values.push_back(ks);
    
    double total_bin = static_cast<double>(neg_counts[i] + pos_counts[i]);
    double gini = 0.0;
    if (total_bin > 0.0) {
      double p1 = static_cast<double>(pos_counts[i]) / total_bin;
      double p0 = static_cast<double>(neg_counts[i]) / total_bin;
      gini = std::fabs(p1 - p0);
    }
    gini_values.push_back(gini);
    
    CHIMERGE_LOG_TRACE("Bin ", i, " - KS: ", ks, ", Gini: ", gini);
  }
  CHIMERGE_LOG_INFO("Calculated KS and Gini for ", ks_values.size(), " bins");
}

// =====================================================================
// SECTION 15: MAIN BINNING ORCHESTRATOR
// =====================================================================

class OptimalBinningOrchestrator {
private:
  // Factory for numeric algorithms
  std::unique_ptr<ChiMergeAlgorithm> make_algo(const std::string& method_str,
                                               int min_bins, int max_bins, double sig_level,
                                               double min_size, bool cache) {
    std::string m = method_str;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m == "chimerge") return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeStandard(min_bins, max_bins, sig_level, min_size, cache));
    if (m == "chimerge_mod") return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeModified(min_bins, max_bins, sig_level, min_size, cache));
    if (m == "chimerge_opt") return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeOptimized(min_bins, max_bins, sig_level, min_size, cache, /*yates=*/true));
    if (m == "d2") return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeD2(min_bins, max_bins, sig_level, min_size, cache));
    if (m == "hybrid") return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeHybrid(min_bins, max_bins, sig_level, min_size, cache));
    // Fallback
    return std::unique_ptr<ChiMergeAlgorithm>(new ChiMergeModified(min_bins, max_bins, sig_level, min_size, cache));
  }
  
public:
  List process_numeric_feature(
      SEXP x,
      const std::string& feature_name,
      const LogicalVector& target,
      const NumericVector& weights,
      int min_bins, int max_bins, double sig_level, double min_size,
      double smooth, bool monotonic, double min_iv, int digits,
      const std::string& miss_policy,
      const std::vector<double>& special_vals,
      const std::string& method_str,
      bool cache) {
    
    CHIMERGE_LOG_INFO("Processing numeric feature: ", feature_name);
    CHIMERGE_TIMER("Numeric feature processing");
    
    std::vector<std::string> messages;
    
    // Prepare & sort data
    auto data = prepare_numeric_data(x, target, weights, special_vals);
    std::sort(data.begin(), data.end(),
              [](const DataPoint& a, const DataPoint& b) {
                if (a.is_missing != b.is_missing) return !a.is_missing;
                if (a.is_special != b.is_special) return !a.is_special;
                return a.numeric_value < b.numeric_value;
              });
    
    // Choose and run algorithm
    CHIMERGE_LOG_INFO("Applying binning algorithm: ", method_str);
    std::unique_ptr<ChiMergeAlgorithm> algo = make_algo(method_str, min_bins, max_bins, sig_level, min_size, cache);
    auto bins = algo->apply_numeric_binning(data);
    
    // Enforce monotonicity by *pooling* adjacent bins (PAVA) BEFORE WoE,
    // so that output counts imply the same WoE you recompute in R.
    if (monotonic && bins.size() > 1) {
      enforce_monotonic_pooling(bins);
    }
    
    if (bins.empty()) {
      messages.push_back("No valid bins created");
      CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
    }
    
    // Special values
    SpecialValueHandler special_handler(miss_policy, special_vals, {});
    auto special_bins = special_handler.process_special_values(data);
    
    // Totals for WoE
    size_t total_pos = 0, total_neg = 0;
    for (const auto& p : data) {
      bool include = (miss_policy == "remove") ? !p.is_missing : true;
      if (!include) continue;
      if (p.target) total_pos++; else total_neg++;
    }
    CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
    
    // WoE / IV for regular bins
    WoECalculator woe_calc(total_pos, total_neg, smooth, false);
    woe_calc.calculate_woe_iv(bins);
    
    // WoE / IV for special bins
    size_t total_bins = bins.size() + special_bins.size();
    for (auto& sb : special_bins) {
      double sm_pos = static_cast<double>(sb.count_1) + smooth;
      double sm_neg = static_cast<double>(sb.count_0) + smooth;
      double sm_tot_pos = static_cast<double>(total_pos) + static_cast<double>(total_bins) * smooth;
      double sm_tot_neg = static_cast<double>(total_neg) + static_cast<double>(total_bins) * smooth;
      if (sm_tot_pos > 0.0 && sm_tot_neg > 0.0) {
        double dpos = sm_pos / sm_tot_pos;
        double dneg = sm_neg / sm_tot_neg;
        if (dpos > 0.0 && dneg > 0.0) {
          sb.woe = std::log(dpos / dneg);
          if (sb.woe > 5.0) sb.woe = 5.0;
          if (sb.woe < -5.0) sb.woe = -5.0;
          sb.iv_contribution = (dpos - dneg) * sb.woe;
        }
      }
    }
    
    // Labels
    for (size_t i = 0; i < bins.size(); ++i) {
      bins[i].label = format_numeric_bin_label(bins[i], i == 0, i + 1 == bins.size(), digits);
    }
    
    // Combine bins for output
    std::vector<int> bin_ids;
    std::vector<std::string> bin_labels;
    std::vector<int> total_counts, neg_counts, pos_counts;
    std::vector<double> woes, ivs;
    
    int bin_id = 1;
    for (const auto& sb : special_bins) {
      bin_ids.push_back(bin_id++);
      bin_labels.push_back(sb.label);
      total_counts.push_back(static_cast<int>(sb.total()));
      neg_counts.push_back(static_cast<int>(sb.count_0));
      pos_counts.push_back(static_cast<int>(sb.count_1));
      woes.push_back(sb.woe);
      ivs.push_back(sb.iv_contribution);
    }
    for (const auto& b : bins) {
      bin_ids.push_back(bin_id++);
      bin_labels.push_back(b.label);
      total_counts.push_back(static_cast<int>(b.total()));
      neg_counts.push_back(static_cast<int>(b.count_0));
      pos_counts.push_back(static_cast<int>(b.count_1));
      woes.push_back(b.woe);
      ivs.push_back(b.iv_contribution);
    }
    
    // KS / Gini
    std::vector<double> ks_values, gini_values;
    calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
    
    DataFrame bins_df = DataFrame::create(
      Named("bin_id") = bin_ids,
      Named("bin_label") = bin_labels,
      Named("total_count") = total_counts,
      Named("neg_count") = neg_counts,
      Named("pos_count") = pos_counts,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("ks") = ks_values,
      Named("gini") = gini_values
    );
    
    // Transformations
    NumericVector woe_transform = apply_numeric_woe_transform(x, bins, special_bins, special_vals);
    CharacterVector bin_transform = apply_numeric_bin_transform(x, bins, special_bins, special_vals);
    
    DataFrame transform_df = DataFrame::create(
      Named(feature_name) = x,
      Named(feature_name + "_bin") = bin_transform,
      Named(feature_name + "_woe") = woe_transform
    );
    
    // Variable type label
    VariableType var_type = detect_r_variable_type(x);
    std::string var_type_str = "numeric";
    if (var_type == VariableType::NUMERIC_CONTINUOUS) var_type_str = "numeric_continuous";
    else if (var_type == VariableType::NUMERIC_DISCRETE) var_type_str = "numeric_discrete";
    else if (var_type == VariableType::BOOLEAN) var_type_str = "boolean";
    
    double total_iv = woe_calc.get_total_iv();
    for (const auto& sb : special_bins) total_iv += sb.iv_contribution;
    
    CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, ", Bins: ", bin_ids.size());
    
    List result = List::create(
      Named("bins") = bins_df,
      Named("total_iv") = total_iv,
      Named("variable_type") = var_type_str,
      Named("method_used") = method_str,
      Named("n_bins") = static_cast<int>(bin_ids.size()),
      Named("total_samples") = static_cast<int>(data.size()),
      Named("event_rate") = (static_cast<double>(std::accumulate(pos_counts.begin(), pos_counts.end(), 0)))
      / std::max(1.0, static_cast<double>(std::accumulate(total_counts.begin(), total_counts.end(), 0))),
      Named("messages") = messages.empty() ? "Success" : messages[0],
                                                                 Named("transform") = transform_df
    );
    return result;
  }
  
  List process_categorical_feature(
      SEXP x,
      const std::string& feature_name,
      const LogicalVector& target,
      const NumericVector& weights,
      int min_bins, int max_bins, double min_size,
      double smooth, bool monotonic, double /*min_iv*/,
      const std::string& miss_policy,
      int max_cat, double rare_pct, const std::string& cat_sep) {
    
    CHIMERGE_LOG_INFO("Processing categorical feature: ", feature_name);
    CHIMERGE_TIMER("Categorical feature processing");
    
    std::vector<std::string> messages;
    
    auto data = prepare_categorical_data(x, target, weights);
    auto bins = apply_categorical_binning(data, min_bins, max_bins, min_size,
                                          max_cat, rare_pct, cat_sep);
    if (bins.empty()) {
      messages.push_back("No valid bins created");
      CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
    }
    
    SpecialValueHandler special_handler(miss_policy, {}, {});
    auto special_bins = special_handler.process_special_values(data);
    
    size_t total_pos = 0, total_neg = 0;
    for (const auto& p : data) {
      bool include = (miss_policy == "remove") ? !p.is_missing : true;
      if (!include) continue;
      if (p.target) total_pos++; else total_neg++;
    }
    CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
    
    WoECalculator woe_calc(total_pos, total_neg, smooth, monotonic);
    woe_calc.calculate_woe_iv(bins);
    
    size_t total_bins = bins.size() + special_bins.size();
    for (auto& sb : special_bins) {
      double sm_pos = static_cast<double>(sb.count_1) + smooth;
      double sm_neg = static_cast<double>(sb.count_0) + smooth;
      double sm_tot_pos = static_cast<double>(total_pos) + static_cast<double>(total_bins) * smooth;
      double sm_tot_neg = static_cast<double>(total_neg) + static_cast<double>(total_bins) * smooth;
      if (sm_tot_pos > 0.0 && sm_tot_neg > 0.0) {
        double dpos = sm_pos / sm_tot_pos;
        double dneg = sm_neg / sm_tot_neg;
        if (dpos > 0.0 && dneg > 0.0) {
          sb.woe = std::log(dpos / dneg);
          if (sb.woe > 5.0) sb.woe = 5.0;
          if (sb.woe < -5.0) sb.woe = -5.0;
          sb.iv_contribution = (dpos - dneg) * sb.woe;
        }
      }
    }
    
    // Combine bins
    std::vector<int> bin_ids;
    std::vector<std::string> bin_labels;
    std::vector<int> total_counts, neg_counts, pos_counts;
    std::vector<double> woes, ivs;
    
    int bin_id = 1;
    for (const auto& sb : special_bins) {
      bin_ids.push_back(bin_id++); bin_labels.push_back(sb.label);
      total_counts.push_back(static_cast<int>(sb.total()));
      neg_counts.push_back(static_cast<int>(sb.count_0));
      pos_counts.push_back(static_cast<int>(sb.count_1));
      woes.push_back(sb.woe); ivs.push_back(sb.iv_contribution);
    }
    for (const auto& b : bins) {
      bin_ids.push_back(bin_id++); bin_labels.push_back(b.label);
      total_counts.push_back(static_cast<int>(b.total()));
      neg_counts.push_back(static_cast<int>(b.count_0));
      pos_counts.push_back(static_cast<int>(b.count_1));
      woes.push_back(b.woe); ivs.push_back(b.iv_contribution);
    }
    
    std::vector<double> ks_values, gini_values;
    calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
    
    DataFrame bins_df = DataFrame::create(
      Named("bin_id") = bin_ids,
      Named("bin_label") = bin_labels,
      Named("total_count") = total_counts,
      Named("neg_count") = neg_counts,
      Named("pos_count") = pos_counts,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("ks") = ks_values,
      Named("gini") = gini_values
    );
    
    // Transform vectors
    int n = Rf_length(x);
    NumericVector woe_transform(n, NA_REAL);
    CharacterVector bin_transform(n);
    
    std::map<std::string, double> cat_woe_map;
    std::map<std::string, std::string> cat_bin_map;
    for (const auto& b : bins) {
      for (const auto& c : b.categories) {
        cat_woe_map[c] = b.woe; cat_bin_map[c] = b.label;
      }
    }
    
    auto missing_woe = [&]() -> double {
      for (const auto& sb : special_bins) if (sb.type == "missing") return sb.woe;
      return NA_REAL;
    };
    
    if (Rf_isFactor(x)) {
      IntegerVector iv = as<IntegerVector>(x);
      CharacterVector levels = iv.attr("levels");
      for (int i = 0; i < n; ++i) {
        if (IntegerVector::is_na(iv[i])) {
          bin_transform[i] = "[Missing]"; woe_transform[i] = missing_woe();
        } else {
          int idx = iv[i] - 1;
          if (idx >= 0 && idx < levels.size()) {
            std::string cat = as<std::string>(levels[idx]);
            auto wit = cat_woe_map.find(cat); auto bit = cat_bin_map.find(cat);
            if (wit != cat_woe_map.end()) { woe_transform[i] = wit->second; bin_transform[i] = bit->second; }
          }
        }
      }
    } else if (Rf_isString(x)) {
      CharacterVector cv = as<CharacterVector>(x);
      for (int i = 0; i < n; ++i) {
        if (CharacterVector::is_na(cv[i])) {
          bin_transform[i] = "[Missing]"; woe_transform[i] = missing_woe();
        } else {
          std::string cat = as<std::string>(cv[i]);
          auto wit = cat_woe_map.find(cat); auto bit = cat_bin_map.find(cat);
          if (wit != cat_woe_map.end()) { woe_transform[i] = wit->second; bin_transform[i] = bit->second; }
        }
      }
    }
    
    DataFrame transform_df = DataFrame::create(
      Named(feature_name) = x,
      Named(feature_name + "_bin") = bin_transform,
      Named(feature_name + "_woe") = woe_transform
    );
    
    double total_iv = woe_calc.get_total_iv();
    for (const auto& sb : special_bins) total_iv += sb.iv_contribution;
    
    CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, ", Bins: ", bin_ids.size());
    
    List result = List::create(
      Named("bins") = bins_df,
      Named("total_iv") = total_iv,
      Named("variable_type") = "categorical",
      Named("method_used") = "categorical_chimerge",
      Named("n_bins") = static_cast<int>(bin_ids.size()),
      Named("total_samples") = static_cast<int>(data.size()),
      Named("event_rate") = (static_cast<double>(std::accumulate(pos_counts.begin(), pos_counts.end(), 0)))
      / std::max(1.0, static_cast<double>(std::accumulate(total_counts.begin(), total_counts.end(), 0))),
      Named("messages") = messages.empty() ? "Success" : messages[0],
                                                                 Named("transform") = transform_df
    );
    return result;
  }
};

// =====================================================================
// SECTION 16: MAIN EXPORTED FUNCTION
// =====================================================================

//' ChiMerge Optimal Binning with Weight of Evidence (WoE)
//'
//' @description
//' Performs optimal binning on numeric and categorical variables using multiple **ChiMerge** algorithm
//' variations, computing **Weight of Evidence (WoE)** and **Information Value (IV)** for each bin.
//' Implements comprehensive binning strategies including:
//' \itemize{
//'   \item \strong{ChiMerge Original} (Kerber, 1992): bottom-up merging based on chi-square independence test
//'   \item \strong{ChiMerge Modified}: handles small expected frequencies with Yates' continuity correction
//'   \item \strong{ChiMerge Optimized}: includes caching and performance optimizations
//'   \item \strong{D2 Algorithm}: entropy-based recursive splitting
//'   \item \strong{Hybrid WoE}: combines multiple strategies for optimal results
//' }
//'
//' @details
//' \strong{1) Chi-Square Test of Independence}
//'
//' For adjacent intervals \eqn{i} and \eqn{j}, the chi-square statistic is:
//' \deqn{\chi^2 = \sum_{k=0}^{1} \sum_{l \in \\{i,j\\}} \frac{(O_{lk} - E_{lk})^2}{E_{lk}}}
//' where:
//' \itemize{
//'    \item \eqn{O_{lk}} = observed frequency in interval \eqn{l}, class \eqn{k}
//'    \item \eqn{E_{lk} = \frac{R_l \times C_k}{N}} = expected frequency under independence
//'    \item \eqn{R_l} = row total for interval \eqn{l}
//'    \item \eqn{C_k} = column total for class \eqn{k}
//'    \item \eqn{N} = total observations
//' }
//' With df = 1, critical values at common significance levels: \eqn{\chi^2_{0.10,1} = 2.706},
//' \eqn{\chi^2_{0.05,1} = 3.841}, \eqn{\chi^2_{0.01,1} = 6.635}.
//'
//' \strong{2) Weight of Evidence (WoE)}
//'
//' For bin \eqn{i}, WoE measures the strength of relationship with the target:
//' \deqn{\text{WoE}_i = \ln\left(\frac{P(X \text{ in bin}_i, Y=1)}{P(X \text{ in bin}_i, Y=0)}\right) = \ln\left(\frac{\text{Dist}_{\text{Good},i}}{\text{Dist}_{\text{Bad},i}}\right)}
//' where:
//' \deqn{\text{Dist}_{\text{Good},i} = \frac{n_{1i}}{n_1}, \quad \text{Dist}_{\text{Bad},i} = \frac{n_{0i}}{n_0}}
//'
//' \strong{Laplace Smoothing}: To handle zero frequencies, applies smoothing factor \eqn{\lambda}:
//' \deqn{\text{WoE}_i = \ln\left(\frac{(n_{1i} + \lambda) / (n_1 + k\lambda)}{(n_{0i} + \lambda) / (n_0 + k\lambda)}\right)}
//' where \eqn{k} is the number of bins.
//'
//' \strong{3) Information Value (IV)}
//'
//' Total IV quantifies the predictive power of the variable:
//' \deqn{\text{IV} = \sum_{i=1}^{k} (\text{Dist}_{\text{Good},i} - \text{Dist}_{\text{Bad},i}) \times \text{WoE}_i}
//'
//' Interpretation guidelines:
//' \itemize{
//'   \item IV \eqn{< 0.02}: Useless predictor
//'   \item \eqn{0.02 \leq} IV \eqn{< 0.1}: Weak predictor
//'   \item \eqn{0.1 \leq} IV \eqn{< 0.3}: Medium predictor
//'   \item \eqn{0.3 \leq} IV \eqn{< 0.5}: Strong predictor
//'   \item IV \eqn{\geq 0.5}: Suspicious (potential overfitting)
//' }
//'
//' \strong{4) Monotonicity Enforcement}
//'
//' When \code{monotonic = TRUE}, applies isotonic regression using the Pool Adjacent Violators
//' Algorithm (PAVA) to ensure monotonic WoE values across bins, preserving the ordinal relationship
//' while minimizing weighted squared deviations.
//'
//' \strong{5) Special Value Handling}
//'
//' Missing values and special codes are handled according to \code{miss_policy}:
//' \itemize{
//'   \item \code{separate}: Creates separate bin(s) for missing/special values
//'   \item \code{remove}: Excludes missing values from binning
//'   \item \code{impute}: Imputes with mode (categorical) or median (numeric)
//'   \item \code{merge}: Merges with nearest bin based on event rate
//' }
//'
//' @param data DataFrame containing features and target variable. Must contain at least the target column
//'   and one feature column.
//'
//' @param target_col String specifying the name of the binary target column. The target must be binary
//'   (0/1, TRUE/FALSE, or two-level factor).
//'
//' @param feature_cols CharacterVector of feature column names to process. These columns can be numeric,
//'   integer, logical, character, or factor types. Date/time columns are automatically skipped.
//'
//' @param min_bins Integer specifying minimum number of bins to create. Must be at least 2.
//'   Default: 2
//'
//' @param max_bins Integer specifying maximum number of bins allowed. Must be greater than or equal to
//'   min_bins. Values above 100 may lead to overfitting.
//'   Default: 10
//'
//' @param sig_level Numeric significance level for chi-square test used in ChiMerge algorithm.
//'   Must be between 0 and 1. Common values: 0.10 (90\% confidence), 0.05 (95\% confidence),
//'   0.01 (99\% confidence). Lower values create fewer bins.
//'   Default: 0.05
//'
//' @param min_size Numeric minimum bin size as proportion of total observations. Must be between 0 and 1.
//'   Prevents creation of very small bins. For example, 0.05 means each bin must contain at least 5\\%
//'   of observations.
//'   Default: 0.05
//'
//' @param smooth Numeric Laplace smoothing factor for WoE calculation. Must be non-negative.
//'   Prevents undefined WoE values when bins have zero counts. Higher values provide more smoothing.
//'   Default: 0.5
//'
//' @param monotonic Logical indicating whether to enforce monotonic WoE values across bins using
//'   isotonic regression. Useful for regulatory compliance and model interpretability.
//'   Default: TRUE
//'
//' @param min_iv Numeric minimum Information Value threshold for feature selection. Features with
//'   IV below this threshold generate a warning. Must be non-negative.
//'   Default: 0.01
//'
//' @param digits Integer number of decimal places for numeric bin boundaries in labels.
//'   Must be between 0 and 10. Affects display only, not calculation precision.
//'   Default: 4
//'
//' @param miss_policy String specifying how to handle missing values. Options:
//'   \itemize{
//'     \item \code{separate}: Create separate bin(s) for missing values (default)
//'     \item \code{remove}: Exclude missing values from analysis
//'     \item \code{impute}: Impute with mode (categorical) or median (numeric)
//'     \item \code{merge}: Merge with nearest bin based on event rate
//'   }
//'   Default: separate
//'
//' @param special_vals NumericVector of special values to handle separately (e.g., -999, -888 for
//'   special codes). These values are isolated in separate bins when encountered.
//'   Default: empty vector (no special values)
//'
//' @param max_cat Integer maximum number of categories before automatic grouping for categorical variables.
//'   Must be at least 2. Categories beyond this limit are grouped based on event rate similarity.
//'   Default: 20
//'
//' @param rare_pct Numeric threshold for rare category grouping as proportion of total.
//'   Categories with frequency below this threshold are combined. Must be between 0 and 0.5.
//'   Default: 0.01 (1\%)
//'
//' @param cat_sep String separator used when combining multiple categories into a single bin label.
//'   Should be a pattern unlikely to appear in actual category names.
//'   Default: "\%;\%"
//'
//' @param method String specifying the binning algorithm to use. Options:
//'   \itemize{
//'     \item \code{chimerge}: Original ChiMerge algorithm
//'     \item \code{chimerge_mod}: Modified ChiMerge with Yates correction (default)
//'     \item \code{chimerge_opt}: Optimized ChiMerge with caching
//'     \item \code{d2}: D2 entropy-based algorithm
//'     \item \code{hybrid}: Hybrid approach combining multiple methods
//'   }
//'   Default: chimerge_mod
//'
//' @param parallel Logical indicating whether to use parallel processing via OpenMP for multiple features.
//'   Requires OpenMP support in compilation.
//'   Default: FALSE
//'
//' @param cache Logical indicating whether to enable chi-square value caching for performance optimization.
//'   Recommended for large datasets.
//'   Default: TRUE
//'
//' @param weights Optional NumericVector of observation weights for weighted binning. Length must equal
//'   number of rows in data. All weights must be non-negative. NULL indicates equal weights.
//'   Default: NULL
//'
//' @param verbose Logical indicating whether to print detailed processing information.
//'   Default: FALSE
//'
//' @return
//' Named list with one element per processed feature, each containing:
//' \itemize{
//'   \item \code{bins}: data.frame with columns:
//'     \itemize{
//'       \item \code{bin_id}: Sequential bin identifier
//'       \item \code{bin_label}: Formatted bin range/categories
//'       \item \code{total_count}: Total observations in bin
//'       \item \code{neg_count}: Count of negative class (Y=0)
//'       \item \code{pos_count}: Count of positive class (Y=1)
//'       \item \code{woe}: Weight of Evidence value
//'       \item \code{iv}: Information Value contribution
//'       \item \code{ks}: Kolmogorov-Smirnov statistic
//'       \item \code{gini}: Gini impurity coefficient
//'     }
//'   \item \code{total_iv}: Total Information Value for the feature
//'   \item \code{variable_type}: Detected type (numeric_continuous, numeric_discrete, categorical, boolean)
//'   \item \code{method_used}: Algorithm actually used for binning
//'   \item \code{n_bins}: Final number of bins created
//'   \item \code{total_samples}: Total valid observations processed
//'   \item \code{event_rate}: Overall positive class rate
//'   \item \code{messages}: Processing messages or warnings
//'   \item \code{transform}: data.frame with original values and transformations:
//'     \itemize{
//'       \item \code{[feature]}: Original feature values
//'       \item \code{[feature]_bin}: Assigned bin labels
//'       \item \code{[feature]_woe}: Assigned WoE values
//'     }
//' }
//'
//' @section Algorithm Details:
//' The ChiMerge algorithm proceeds as follows:
//' \enumerate{
//'   \item \strong{Initialization}: Create initial bins (one per unique value for numeric, one per category for categorical)
//'   \item \strong{Iterative Merging}:
//'     \itemize{
//'       \item Calculate chi-square statistic for all adjacent bin pairs
//'       \item Find pair with minimum chi-square value
//'       \item If min chi-square < threshold and bins > min_bins, merge the pair
//'       \item Repeat until stopping criteria met
//'     }
//'   \item \strong{Post-processing}:
//'     \itemize{
//'       \item Ensure max_bins constraint
//'       \item Calculate WoE and IV for final bins
//'       \item Apply monotonicity constraint if requested
//'     }
//' }
//'
//' @note
//' \itemize{
//'   \item Date/time variables are automatically excluded
//'   \item Factor variables are processed as categorical
//'   \item Integer variables with 20 or fewer unique values are treated as discrete
//'   \item Special values (-999, -888, etc.) can be isolated in separate bins
//'   \item Uses (-inf,x) and [x,+inf) notation for unbounded intervals
//' }
//'
//' @references
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128).
//'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
//'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @examples
//' \dontrun{
//' # Load credit data
//' data <- scorecard::germancredit
//' data$target <- ifelse(data$creditability == "good", 1, 0)
//'
//' # Basic binning with all defaults
//' result <- chimerge_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
//' )
//'
//' # Custom binning with specific parameters
//' result_custom <- chimerge_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = names(data)[1:10],
//'   min_bins = 3,              # Minimum 3 bins
//'   max_bins = 8,               # Maximum 8 bins
//'   sig_level = 0.10,           # 90% confidence level
//'   min_size = 0.03,            # Each bin at least 3% of data
//'   smooth = 1.0,               # Higher smoothing
//'   monotonic = TRUE,           # Enforce monotonic WoE
//'   min_iv = 0.05,              # Warn if IV < 0.05
//'   digits = 2,                 # 2 decimal places in labels
//'   miss_policy = "separate",   # Missing values in separate bin
//'   special_vals = c(-999, -888), # Special codes to isolate
//'   max_cat = 15,               # Group if >15 categories
//'   rare_pct = 0.02,            # Group categories <2% frequency
//'   cat_sep = " | ",            # Use pipe separator
//'   method = "chimerge_mod",    # Modified ChiMerge
//'   parallel = FALSE,           # No parallel processing
//'   cache = TRUE,               # Enable caching
//'   weights = NULL,             # No observation weights
//'   verbose = TRUE              # Show detailed logs
//' )
//'
//' # Extract results for a specific feature
//' duration_bins <- result$duration.in.month$bins
//' duration_iv <- result$duration.in.month$total_iv
//' duration_transform <- result$duration.in.month$transform
//' }
//'
//' @export
// [[Rcpp::export]]
List chimerge_woe(
   DataFrame data,
   std::string target_col,
   CharacterVector feature_cols,
   int min_bins = 2,
   int max_bins = 5,
   double sig_level = 0.05,
   double min_size = 0.05,
   double smooth = 0,
   bool monotonic = false,
   double min_iv = 0.01,
   int digits = 4,
   std::string miss_policy = "separate",
   NumericVector special_vals = NumericVector::create(),
   int max_cat = 20,
   double rare_pct = 0.01,
   std::string cat_sep = "%;%",
   std::string method = "chimerge_mod",
   bool parallel = false,
   bool cache = true,
   Nullable<NumericVector> weights = R_NilValue,
   bool verbose = false) {
 
 try {
   // Initialize logger
   chimerge_logging::global_logger.set_verbose(verbose);
   
   CHIMERGE_LOG_INFO("Starting ChiMerge WoE processing");
   CHIMERGE_TIMER("Total processing");
   
   // Validate parameters
   auto param_validation = InputValidator::validate_binning_parameters(
     min_bins, max_bins, sig_level, min_size, smooth, monotonic, min_iv, digits,
     miss_policy, max_cat, rare_pct, method, parallel, cache
   );
   if (!param_validation.is_valid) {
     std::string error_msg = "Parameter validation failed:\n";
     for (const auto& err : param_validation.errors) error_msg += "  - " + err + "\n";
     CHIMERGE_LOG_ERROR(error_msg);
     Rcpp::stop(error_msg);
   }
   
   // Extract target
   LogicalVector target;
   SEXP target_sexp = data[target_col];
   if (Rf_isLogical(target_sexp)) {
     target = as<LogicalVector>(target_sexp);
   } else if (Rf_isInteger(target_sexp)) {
     IntegerVector iv = as<IntegerVector>(target_sexp);
     target = LogicalVector(iv.size());
     for (int i = 0; i < iv.size(); ++i) target[i] = (iv[i] == 1);
   } else if (Rf_isNumeric(target_sexp)) {
     NumericVector nv = as<NumericVector>(target_sexp);
     target = LogicalVector(nv.size());
     for (int i = 0; i < nv.size(); ++i) target[i] = (nv[i] == 1.0);
   } else {
     Rcpp::stop("Unsupported target type; must be logical/integer/numeric (0/1)");
   }
   
   // Extract weights
   NumericVector weight_vec;
   if (weights.isNotNull()) weight_vec = weights.get();
   else weight_vec = NumericVector(data.nrows(), 1.0);
   
   // Convert special values
   std::vector<double> special_vals_vec;
   special_vals_vec.reserve(special_vals.size());
   for (int i = 0; i < special_vals.size(); ++i) special_vals_vec.push_back(special_vals[i]);
   
   // Process features
   List results;
   OptimalBinningOrchestrator orchestrator;
   std::vector<std::string> features = Rcpp::as<std::vector<std::string>>(feature_cols);
   
   CHIMERGE_LOG_INFO("Processing ", features.size(), " features");
   
   for (const std::string& feature : features) {
     try {
       CHIMERGE_LOG_INFO("Processing feature: ", feature);
       
       if (!data.containsElementNamed(feature.c_str())) {
         Rcpp::warning("Feature '" + feature + "' not found in data");
         CHIMERGE_LOG_WARNING("Feature '", feature, "' not found in data");
         continue;
       }
       
       SEXP feature_sexp = data[feature];
       
       // Check variable type
       VariableType var_type = detect_r_variable_type(feature_sexp);
       
       if (var_type == VariableType::DATE_TIME) {
         Rcpp::warning("Feature '" + feature + "' is a date/time type, skipping");
         CHIMERGE_LOG_WARNING("Feature '", feature, "' is a date/time type, skipping");
         continue;
       }
       if (var_type == VariableType::UNSUPPORTED) {
         Rcpp::warning("Feature '" + feature + "' has unsupported type, skipping");
         CHIMERGE_LOG_WARNING("Feature '", feature, "' has unsupported type, skipping");
         continue;
       }
       
       List feature_result;
       if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN ||
           Rf_isFactor(feature_sexp) || Rf_isString(feature_sexp)) {
         CHIMERGE_LOG_DEBUG("Processing as categorical feature");
         feature_result = orchestrator.process_categorical_feature(
           feature_sexp, feature, target, weight_vec,
           min_bins, max_bins, min_size,
           smooth, monotonic, min_iv,
           miss_policy,
           max_cat, rare_pct, cat_sep
         );
       } else {
         CHIMERGE_LOG_DEBUG("Processing as numeric feature");
         feature_result = orchestrator.process_numeric_feature(
           feature_sexp, feature, target, weight_vec,
           min_bins, max_bins, sig_level, min_size,
           smooth, monotonic, min_iv, digits,
           miss_policy, special_vals_vec,
           method, cache
         );
       }
       
       // Minimum IV note (does not drop the result; just annotates)
       double iv = as<double>(feature_result["total_iv"]);
       if (iv < min_iv) {
         feature_result["messages"] = "IV below threshold (" + std::to_string(iv) + ")";
         CHIMERGE_LOG_WARNING("Feature '", feature, "' has IV below threshold: ", iv);
       }
       
       results[feature] = feature_result;
       CHIMERGE_LOG_INFO("Completed processing feature: ", feature);
       
     } catch (const std::exception& e) {
       Rcpp::warning("Error processing feature '" + feature + "': " + std::string(e.what()));
       CHIMERGE_LOG_ERROR("Error processing feature '", feature, "': ", e.what());
     }
   }
   
   CHIMERGE_LOG_INFO("ChiMerge WoE processing completed");
   return results;
   
 } catch (const std::exception& e) {
   CHIMERGE_LOG_ERROR("Error in chimerge_woe: ", e.what());
   Rcpp::stop(std::string("Error in chimerge_woe: ") + e.what());
 }
}



















// // chimerge_woe.cpp
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <map>
// #include <unordered_map>
// #include <algorithm>
// #include <numeric>
// #include <cmath>
// #include <iostream>
// #include <sstream>
// #include <iomanip>
// #include <set>
// #include <queue>
// #include <chrono>
// #include <exception>
// #include <memory>
// #include <limits>
// #include <functional>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// using namespace std;
// 
// // =====================================================================
// // SECTION 0: LOGGING SYSTEM
// // =====================================================================
// 
// namespace chimerge_logging {
// 
// enum class LogLevel {
//   NONE = 0,
//     ERROR = 1,
//     WARNING = 2,
//     INFO = 3,
//     DEBUG = 4
// };
// 
// class Logger {
// private:
//   bool verbose_;
//   LogLevel log_level_;
//   
// public:
//   Logger(bool verbose = false, LogLevel level = LogLevel::INFO)
//     : verbose_(verbose), log_level_(level) {}
//   
//   template<typename... Args>
//   void error(Args&&... args) {
//     if (verbose_ && log_level_ >= LogLevel::ERROR) {
//       log("ERROR", std::forward<Args>(args)...);
//     }
//   }
//   
//   template<typename... Args>
//   void warning(Args&&... args) {
//     if (verbose_ && log_level_ >= LogLevel::WARNING) {
//       log("WARNING", std::forward<Args>(args)...);
//     }
//   }
//   
//   template<typename... Args>
//   void info(Args&&... args) {
//     if (verbose_ && log_level_ >= LogLevel::INFO) {
//       log("INFO", std::forward<Args>(args)...);
//     }
//   }
//   
//   template<typename... Args>
//   void debug(Args&&... args) {
//     if (verbose_ && log_level_ >= LogLevel::DEBUG) {
//       log("DEBUG", std::forward<Args>(args)...);
//     }
//   }
//   
//   template<typename... Args>
//   void log(const std::string& level, Args&&... args) {
//     if (!verbose_) return;
//     
//     Rcout << "[" << level << "] ";
//     log_impl(std::forward<Args>(args)...);
//     Rcout << std::endl;
//   }
//   
//   class Timer {
//   private:
//     std::chrono::high_resolution_clock::time_point start_time_;
//     std::string description_;
//     Logger* logger_;
//     bool verbose_;
//     
//   public:
//     Timer(Logger* logger, const std::string& description, bool verbose = true)
//       : logger_(logger), description_(description), 
//         verbose_(verbose && logger->is_verbose()) {
//       if (verbose_) {
//         start_time_ = std::chrono::high_resolution_clock::now();
//         logger_->debug("Starting: ", description_);
//       }
//     }
//     
//     ~Timer() {
//       if (verbose_) {
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//           end_time - start_time_);
//         logger_->debug("Completed: ", description_, " (", duration.count(), " ms)");
//       }
//     }
//   };
//   
//   bool is_verbose() const { return verbose_; }
//   LogLevel get_level() const { return log_level_; }
//   
//   void set_verbose(bool verbose) { verbose_ = verbose; }
//   void set_level(LogLevel level) { log_level_ = level; }
//   
// private:
//   void log_impl() {}
//   
//   template<typename T, typename... Args>
//   void log_impl(T&& t, Args&&... args) {
//     Rcout << t;
//     log_impl(std::forward<Args>(args)...);
//   }
// };
// 
// static Logger global_logger;
// 
// } // namespace chimerge_logging
// 
// #define CHIMERGE_LOG_TRACE(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
// #define CHIMERGE_LOG_ERROR(...) chimerge_logging::global_logger.error(__VA_ARGS__)
// #define CHIMERGE_LOG_WARNING(...) chimerge_logging::global_logger.warning(__VA_ARGS__)
// #define CHIMERGE_LOG_INFO(...) chimerge_logging::global_logger.info(__VA_ARGS__)
// #define CHIMERGE_LOG_DEBUG(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
// #define CHIMERGE_TIMER(description) chimerge_logging::Logger::Timer timer(&chimerge_logging::global_logger, description)
// 
// // =====================================================================
// // SECTION 1: ENUMS AND CORE DATA STRUCTURES
// // =====================================================================
// 
// enum class VariableType {
//   NUMERIC_CONTINUOUS,
//   NUMERIC_DISCRETE,
//   CATEGORICAL,
//   BOOLEAN,
//   DATE_TIME,
//   UNSUPPORTED
// };
// 
// enum class MissingPolicy {
//   SEPARATE,
//   REMOVE,
//   IMPUTE,
//   MERGE
// };
// 
// enum class BinningMethod {
//   CHIMERGE,
//   CHIMERGE_MOD,
//   CHIMERGE_OPT,
//   D2,
//   HYBRID
// };
// 
// struct DataPoint {
//   double numeric_value;
//   std::string categorical_value;
//   bool target;
//   bool is_missing;
//   bool is_special;
//   size_t original_index;
//   double weight;
//   
//   DataPoint() : numeric_value(0.0), target(false), is_missing(false),
//   is_special(false), original_index(0), weight(1.0) {}
// };
// 
// struct NumericBin {
//   double lower_bound;
//   double upper_bound;
//   size_t count_0;
//   size_t count_1;
//   double woe;
//   double iv_contribution;
//   std::string label;
//   double weight_sum;
//   
//   size_t total() const { return count_0 + count_1; }
//   double event_rate() const { 
//     return total() > 0 ? static_cast<double>(count_1) / total() : 0.0; 
//   }
// };
// 
// struct CategoricalBin {
//   std::vector<std::string> categories;
//   size_t count_0;
//   size_t count_1;
//   double woe;
//   double iv_contribution;
//   std::string label;
//   double weight_sum;
//   
//   size_t total() const { return count_0 + count_1; }
//   double event_rate() const { 
//     return total() > 0 ? static_cast<double>(count_1) / total() : 0.0; 
//   }
// };
// 
// struct SpecialBin {
//   std::string type;
//   std::vector<std::string> values;
//   size_t count_0;
//   size_t count_1;
//   double woe;
//   double iv_contribution;
//   std::string label;
//   double weight_sum;
//   
//   size_t total() const { return count_0 + count_1; }
//   double event_rate() const { 
//     return total() > 0 ? static_cast<double>(count_1) / total() : 0.0; 
//   }
// };
// 
// struct CategoryStats {
//   std::string category;
//   size_t count_0;
//   size_t count_1;
//   double event_rate;
//   double weight_sum;
//   
//   size_t total() const { return count_0 + count_1; }
// };
// 
// struct ValidationResult {
//   bool is_valid;
//   std::vector<std::string> errors;
//   std::vector<std::string> warnings;
//   std::vector<std::string> messages;
//   
//   void add_error(const std::string& msg) {
//     errors.push_back(msg);
//     is_valid = false;
//   }
//   
//   void add_warning(const std::string& msg) {
//     warnings.push_back(msg);
//   }
//   
//   void add_message(const std::string& msg) {
//     messages.push_back(msg);
//   }
// };
// 
// // =====================================================================
// // SECTION 2: EXCEPTION HANDLING
// // =====================================================================
// 
// class BinningException : public std::exception {
// private:
//   std::string message;
// public:
//   BinningException(const std::string& msg) : message(msg) {}
//   const char* what() const noexcept override { return message.c_str(); }
// };
// 
// // =====================================================================
// // SECTION 3: TYPE DETECTION
// // =====================================================================
// 
// VariableType detect_r_variable_type(SEXP x) {
//   CHIMERGE_LOG_TRACE("Detecting variable type");
//   
//   // Check for Date/DateTime types first
//   if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") ||
//       Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
//     CHIMERGE_LOG_DEBUG("Detected date/time variable");
//     return VariableType::DATE_TIME;
//   }
//   
//   // Check for factor
//   if (Rf_isFactor(x)) {
//     SEXP levels = Rf_getAttrib(x, R_LevelsSymbol);
//     int n_levels = Rf_length(levels);
//     CHIMERGE_LOG_DEBUG("Detected factor with ", n_levels, " levels");
//     if (n_levels <= 2) {
//       return VariableType::BOOLEAN;
//     }
//     return VariableType::CATEGORICAL;
//   }
//   
//   // Check for logical
//   if (Rf_isLogical(x)) {
//     CHIMERGE_LOG_DEBUG("Detected logical variable");
//     return VariableType::BOOLEAN;
//   }
//   
//   // Check for character
//   if (Rf_isString(x)) {
//     CharacterVector cv = as<CharacterVector>(x);
//     std::set<std::string> unique_vals;
//     for (int i = 0; i < cv.size(); ++i) {
//       if (!CharacterVector::is_na(cv[i])) {
//         unique_vals.insert(as<std::string>(cv[i]));
//       }
//     }
//     CHIMERGE_LOG_DEBUG("Character variable with ", unique_vals.size(), " unique values");
//     if (unique_vals.size() <= 2) {
//       return VariableType::BOOLEAN;
//     }
//     return VariableType::CATEGORICAL;
//   }
//   
//   // Check for numeric types
//   if (Rf_isInteger(x)) {
//     if (Rf_inherits(x, "integer64")) {
//       CHIMERGE_LOG_DEBUG("Detected integer64 variable");
//       return VariableType::NUMERIC_CONTINUOUS;
//     }
//     
//     IntegerVector iv = as<IntegerVector>(x);
//     std::set<int> unique_vals;
//     for (int i = 0; i < iv.size(); ++i) {
//       if (!IntegerVector::is_na(iv[i])) {
//         unique_vals.insert(iv[i]);
//         if (unique_vals.size() > 20) break;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Integer variable with ", unique_vals.size(), " unique values");
//     
//     if (unique_vals.size() <= 2) {
//       bool is_binary = true;
//       for (int val : unique_vals) {
//         if (val != 0 && val != 1) {
//           is_binary = false;
//           break;
//         }
//       }
//       if (is_binary) return VariableType::BOOLEAN;
//     }
//     
//     return unique_vals.size() <= 20 ? VariableType::NUMERIC_DISCRETE :
//       VariableType::NUMERIC_CONTINUOUS;
//   }
//   
//   if (Rf_isNumeric(x)) {
//     NumericVector nv = as<NumericVector>(x);
//     std::set<double> unique_vals;
//     bool all_integers = true;
//     
//     for (int i = 0; i < nv.size(); ++i) {
//       if (!NumericVector::is_na(nv[i])) {
//         unique_vals.insert(nv[i]);
//         if (nv[i] != std::floor(nv[i])) {
//           all_integers = false;
//         }
//         if (unique_vals.size() > 20 && !all_integers) break;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Numeric variable with ", unique_vals.size(), 
//                        " unique values, all_integers: ", all_integers);
//     
//     if (all_integers && unique_vals.size() <= 2) {
//       bool is_binary = true;
//       for (double val : unique_vals) {
//         if (val != 0.0 && val != 1.0) {
//           is_binary = false;
//           break;
//         }
//       }
//       if (is_binary) return VariableType::BOOLEAN;
//     }
//     
//     if (all_integers && unique_vals.size() <= 20) {
//       return VariableType::NUMERIC_DISCRETE;
//     }
//     
//     return VariableType::NUMERIC_CONTINUOUS;
//   }
//   
//   CHIMERGE_LOG_WARNING("Unsupported variable type detected");
//   return VariableType::UNSUPPORTED;
// }
// 
// // =====================================================================
// // SECTION 4: INPUT VALIDATION
// // =====================================================================
// 
// class InputValidator {
// public:
//   static ValidationResult validate_binning_parameters(
//       int min_bins, int max_bins, double sig_level, double min_size,
//       double smooth, bool monotonic, double min_iv, int digits,
//       const std::string& miss_policy, int max_cat, double rare_pct,
//       const std::string& method, bool parallel, bool cache) {
//     
//     CHIMERGE_LOG_INFO("Validating binning parameters");
//     
//     ValidationResult result;
//     result.is_valid = true;
//     
//     // Validate bin counts
//     if (min_bins < 2) {
//       result.add_error("min_bins must be at least 2, got " + std::to_string(min_bins));
//     }
//     if (max_bins < min_bins) {
//       result.add_error("max_bins must be >= min_bins");
//     }
//     if (max_bins > 100) {
//       result.add_warning("max_bins > 100 may lead to overfitting");
//     }
//     
//     // Validate digits
//     if (digits < 0 || digits > 10) {
//       result.add_error("digits must be between 0 and 10");
//     }
//     
//     // Validate significance level
//     if (sig_level <= 0.0 || sig_level >= 1.0) {
//       result.add_error("sig_level must be between 0 and 1");
//     }
//     
//     // Validate minimum bin size
//     if (min_size <= 0.0 || min_size >= 1.0) {
//       result.add_error("min_size must be between 0 and 1");
//     }
//     
//     // Validate smoothing
//     if (smooth < 0.0) {
//       result.add_error("smooth must be non-negative");
//     }
//     
//     // Validate minimum IV
//     if (min_iv < 0.0) {
//       result.add_error("min_iv must be non-negative");
//     }
//     
//     // Validate missing policy
//     std::set<std::string> valid_policies = {"separate", "remove", "impute", "merge"};
//     if (valid_policies.find(miss_policy) == valid_policies.end()) {
//       result.add_error("miss_policy must be one of: separate, remove, impute, merge");
//     }
//     
//     // Validate max categories
//     if (max_cat < 2) {
//       result.add_error("max_cat must be at least 2");
//     }
//     
//     // Validate rare percentage
//     if (rare_pct < 0.0 || rare_pct >= 0.5) {
//       result.add_error("rare_pct must be between 0 and 0.5");
//     }
//     
//     // Validate method
//     std::set<std::string> valid_methods = {"chimerge", "chimerge_mod", "chimerge_opt", 
//                                            "d2", "hybrid"};
//     if (valid_methods.find(method) == valid_methods.end()) {
//       result.add_error("invalid method: " + method);
//     }
//     
//     CHIMERGE_LOG_DEBUG("Validation completed - valid: ", result.is_valid);
//     return result;
//   }
// };
// 
// // =====================================================================
// // SECTION 5: CHI-SQUARE CACHE
// // =====================================================================
// 
// struct PairHash {
//   size_t operator()(const std::pair<size_t, size_t>& p) const {
//     return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 1);
//   }
// };
// 
// class ChiSquareCache {
// private:
//   std::unordered_map<std::pair<size_t, size_t>, double, PairHash> cache;
//   std::vector<bool> valid_flags;
//   size_t max_cache_size;
//   bool enabled;
//   
// public:
//   ChiSquareCache(size_t max_size = 100000, bool use_cache = true)
//     : max_cache_size(max_size), enabled(use_cache) {}
//   
//   void initialize(size_t size) {
//     if (!enabled) return;
//     CHIMERGE_LOG_DEBUG("Initializing chi-square cache with size ", size);
//     cache.clear();
//     valid_flags.assign(size, true);
//   }
//   
//   double get_cached_value(size_t i, size_t j) {
//     if (!enabled) return -1.0;
//     auto key = std::make_pair(i, j);
//     auto it = cache.find(key);
//     return (it != cache.end()) ? it->second : -1.0;
//   }
//   
//   void invalidate_range(size_t start, size_t end) {
//     if (!enabled) return;
//     CHIMERGE_LOG_TRACE("Invalidating cache range [", start, ", ", end, "]");
//     for (auto it = cache.begin(); it != cache.end();) {
//       if (it->first.first >= start || it->first.second >= start) {
//         it = cache.erase(it);
//       } else {
//         ++it;
//       }
//     }
//   }
//   
//   void update_cache(size_t i, size_t j, double value) {
//     if (!enabled) return;
//     if (cache.size() < max_cache_size) {
//       cache[std::make_pair(i, j)] = value;
//     }
//   }
// };
// 
// // =====================================================================
// // SECTION 6: STATISTICAL FUNCTIONS
// // =====================================================================
// 
// double get_chi_square_threshold(double significance_level, int df) {
//   CHIMERGE_LOG_TRACE("Getting chi-square threshold for sig_level: ", 
//                      significance_level, ", df: ", df);
//   
//   if (df != 1) {
//     throw BinningException("Only df=1 is supported for chi-square test");
//   }
//   
//   if (significance_level >= 0.25) return 1.323;
//   else if (significance_level >= 0.20) return 1.642;
//   else if (significance_level >= 0.15) return 2.072;
//   else if (significance_level >= 0.10) return 2.706;
//   else if (significance_level >= 0.05) return 3.841;
//   else if (significance_level >= 0.025) return 5.024;
//   else if (significance_level >= 0.01) return 6.635;
//   else if (significance_level >= 0.005) return 7.879;
//   else if (significance_level >= 0.001) return 10.828;
//   else return 15.138;
// }
// 
// double calculate_chi_square(const NumericBin& bin1, const NumericBin& bin2) {
//   double total = bin1.total() + bin2.total();
//   if (total == 0) return 0.0;
//   
//   double total_0 = bin1.count_0 + bin2.count_0;
//   double total_1 = bin1.count_1 + bin2.count_1;
//   
//   if (total_0 == 0 || total_1 == 0) return 0.0;
//   
//   double chi_square = 0.0;
//   
//   std::vector<std::vector<double>> observed = {
//     {static_cast<double>(bin1.count_0), static_cast<double>(bin1.count_1)},
//     {static_cast<double>(bin2.count_0), static_cast<double>(bin2.count_1)}
//   };
//   
//   for (int i = 0; i < 2; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       double row_total = observed[i][0] + observed[i][1];
//       double col_total = (j == 0) ? total_0 : total_1;
//       double expected = (row_total * col_total) / total;
//       
//       if (expected > 0) {
//         chi_square += std::pow(observed[i][j] - expected, 2) / expected;
//       }
//     }
//   }
//   
//   CHIMERGE_LOG_TRACE("Chi-square calculated: ", chi_square);
//   return chi_square;
// }
// 
// double calculate_chi_square_modified(const NumericBin& bin1, const NumericBin& bin2) {
//   double total = bin1.total() + bin2.total();
//   if (total == 0) return 0.0;
//   
//   double total_0 = bin1.count_0 + bin2.count_0;
//   double total_1 = bin1.count_1 + bin2.count_1;
//   
//   if (total_0 == 0 || total_1 == 0) return 0.0;
//   
//   double chi_square = 0.0;
//   
//   std::vector<std::vector<double>> observed = {
//     {static_cast<double>(bin1.count_0), static_cast<double>(bin1.count_1)},
//     {static_cast<double>(bin2.count_0), static_cast<double>(bin2.count_1)}
//   };
//   
//   for (int i = 0; i < 2; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       double row_total = observed[i][0] + observed[i][1];
//       double col_total = (j == 0) ? total_0 : total_1;
//       double expected = (row_total * col_total) / total;
//       
//       // Yates' continuity correction
//       double diff = std::abs(observed[i][j] - expected) - 0.5;
//       if (diff < 0) diff = 0;
//       
//       double denominator = std::max(expected, 0.5);
//       chi_square += (diff * diff) / denominator;
//     }
//   }
//   
//   CHIMERGE_LOG_TRACE("Modified chi-square calculated: ", chi_square);
//   return chi_square;
// }
// 
// double calculate_correlation(const std::vector<double>& x, const std::vector<double>& y) {
//   if (x.size() != y.size() || x.empty()) return 0.0;
//   
//   double n = x.size();
//   double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
//   double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
//   double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
//   
//   for (size_t i = 0; i < x.size(); ++i) {
//     sum_xy += x[i] * y[i];
//     sum_x2 += x[i] * x[i];
//     sum_y2 += y[i] * y[i];
//   }
//   
//   double numerator = n * sum_xy - sum_x * sum_y;
//   double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * 
//                                  (n * sum_y2 - sum_y * sum_y));
//   
//   double correlation = (denominator != 0) ? numerator / denominator : 0.0;
//   CHIMERGE_LOG_TRACE("Correlation calculated: ", correlation);
//   return correlation;
// }
// 
// // =====================================================================
// // SECTION 7: BIN LABEL FORMATTING
// // =====================================================================
// 
// std::string format_numeric_bin_label(const NumericBin& bin, bool is_first, 
//                                      bool is_last, int digits) {
//   std::ostringstream oss;
//   oss << std::fixed << std::setprecision(digits);
//   
//   if (is_first && bin.lower_bound == -std::numeric_limits<double>::infinity()) {
//     oss << "(-inf," << bin.upper_bound << ")";
//   } else if (is_last && bin.upper_bound == std::numeric_limits<double>::infinity()) {
//     oss << "[" << bin.lower_bound << ",+inf)";
//   } else {
//     oss << "[" << bin.lower_bound << "," << bin.upper_bound << ")";
//   }
//   
//   std::string label = oss.str();
//   CHIMERGE_LOG_TRACE("Formatted numeric bin label: ", label);
//   return label;
// }
// 
// std::string format_categorical_bin_label(const CategoricalBin& bin, 
//                                          const std::string& separator) {
//   if (bin.categories.size() == 1) {
//     return bin.categories[0];
//   }
//   
//   std::ostringstream oss;
//   for (size_t i = 0; i < bin.categories.size(); ++i) {
//     if (i > 0) oss << separator;
//     oss << bin.categories[i];
//   }
//   
//   std::string label = oss.str();
//   CHIMERGE_LOG_TRACE("Formatted categorical bin label: ", label);
//   return label;
// }
// 
// // =====================================================================
// // SECTION 8: WOE CALCULATOR (FIXED)
// // =====================================================================
// 
// class WoECalculator {
// private:
//   double total_positives;
//   double total_negatives;
//   double laplace_smoothing;
//   double total_iv;
//   bool enforce_monotonic;
//   
// public:
//   WoECalculator(size_t pos_count, size_t neg_count, double smoothing, bool monotonic)
//     : total_positives(pos_count), total_negatives(neg_count),
//       laplace_smoothing(smoothing), total_iv(0.0), 
//       enforce_monotonic(monotonic) {}
//   
//   template<typename BinType>
//   void calculate_woe_iv(std::vector<BinType>& bins) {
//     CHIMERGE_LOG_INFO("Calculating WoE and IV for ", bins.size(), " bins");
//     CHIMERGE_TIMER("WoE/IV calculation");
//     
//     total_iv = 0.0;
//     size_t num_bins = bins.size();
//     
//     if (num_bins == 0) {
//       CHIMERGE_LOG_WARNING("Empty bins vector, skipping WoE calculation");
//       return;
//     }
//     
//     // Step 1: Calculate basic WoE values
//     std::vector<double> woe_values(num_bins);
//     std::vector<double> dist_pos(num_bins);
//     std::vector<double> dist_neg(num_bins);
//     
//     double smoothed_total_pos = total_positives + num_bins * laplace_smoothing;
//     double smoothed_total_neg = total_negatives + num_bins * laplace_smoothing;
//     
//     if (smoothed_total_pos == 0 || smoothed_total_neg == 0) {
//       CHIMERGE_LOG_WARNING("Zero total positives or negatives");
//       for (auto& bin : bins) {
//         bin.woe = 0;
//         bin.iv_contribution = 0;
//       }
//       return;
//     }
//     
//     for (size_t i = 0; i < num_bins; ++i) {
//       double smoothed_pos = bins[i].count_1 + laplace_smoothing;
//       double smoothed_neg = bins[i].count_0 + laplace_smoothing;
//       
//       dist_pos[i] = smoothed_pos / smoothed_total_pos;
//       dist_neg[i] = smoothed_neg / smoothed_total_neg;
//       
//       if (dist_pos[i] == 0 || dist_neg[i] == 0) {
//         woe_values[i] = 0;
//       } else {
//         woe_values[i] = std::log(dist_pos[i] / dist_neg[i]);
//         // Cap WoE values to prevent extreme values
//         if (woe_values[i] > 5) woe_values[i] = 5;
//         if (woe_values[i] < -5) woe_values[i] = -5;
//       }
//       
//       CHIMERGE_LOG_DEBUG("Bin ", i, " initial WoE: ", woe_values[i]);
//     }
//     
//     // Step 2: Apply monotonicity if requested
//     if (enforce_monotonic && bins.size() > 2) {
//       CHIMERGE_LOG_DEBUG("Enforcing WoE monotonicity");
//       apply_pava_to_woe(woe_values, bins);
//     }
//     
//     // Step 3: Assign final WoE values and calculate IV
//     for (size_t i = 0; i < num_bins; ++i) {
//       bins[i].woe = woe_values[i];
//       bins[i].iv_contribution = (dist_pos[i] - dist_neg[i]) * bins[i].woe;
//       total_iv += bins[i].iv_contribution;
//       
//       CHIMERGE_LOG_DEBUG("Bin ", i, " final WoE: ", bins[i].woe, 
//                          ", IV contribution: ", bins[i].iv_contribution);
//     }
//     
//     CHIMERGE_LOG_INFO("Total IV calculated: ", total_iv);
//   }
//   
//   double get_total_iv() const { return total_iv; }
//   
// private:
//   // Apply PAVA directly to WoE values (not to event rates)
//   template<typename BinType>
//   void apply_pava_to_woe(std::vector<double>& woe_values, 
//                          const std::vector<BinType>& bins) {
//     size_t n = woe_values.size();
//     if (n <= 2) return;
//     
//     // Determine monotonicity direction based on correlation
//     std::vector<double> positions(n);
//     std::vector<double> event_rates(n);
//     
//     for (size_t i = 0; i < n; ++i) {
//       positions[i] = i;
//       event_rates[i] = bins[i].event_rate();
//     }
//     
//     double correlation = calculate_correlation(positions, event_rates);
//     bool increasing = (correlation >= 0);
//     
//     CHIMERGE_LOG_DEBUG("Applying PAVA to WoE with direction: ", 
//                        (increasing ? "increasing" : "decreasing"));
//     
//     // Weights based on bin sizes
//     std::vector<double> weights(n);
//     for (size_t i = 0; i < n; ++i) {
//       weights[i] = bins[i].total();
//     }
//     
//     // Apply PAVA algorithm to WoE values
//     if (increasing) {
//       // WoE should increase (bad rate decreases)
//       apply_pava_increasing(woe_values, weights);
//     } else {
//       // WoE should decrease (bad rate increases)
//       apply_pava_decreasing(woe_values, weights);
//     }
//   }
//   
//   // PAVA for increasing constraint
//   void apply_pava_increasing(std::vector<double>& values, 
//                              std::vector<double>& weights) {
//     size_t n = values.size();
//     
//     // Create working arrays
//     std::vector<double> work_values = values;
//     std::vector<double> work_weights = weights;
//     std::vector<size_t> blocks(n);
//     for (size_t i = 0; i < n; ++i) blocks[i] = i;
//     
//     bool changed = true;
//     while (changed) {
//       changed = false;
//       
//       for (size_t i = 0; i < n - 1; ++i) {
//         if (blocks[i] == blocks[i + 1]) continue;
//         
//         if (work_values[blocks[i]] > work_values[blocks[i + 1]]) {
//           // Pool adjacent violators
//           double pooled_weight = work_weights[blocks[i]] + 
//             work_weights[blocks[i + 1]];
//           double pooled_value = (work_weights[blocks[i]] * work_values[blocks[i]] + 
//                                  work_weights[blocks[i + 1]] * work_values[blocks[i + 1]]) / 
//                                  pooled_weight;
//           
//           size_t block_id = blocks[i];
//           size_t next_block_id = blocks[i + 1];
//           
//           work_values[block_id] = pooled_value;
//           work_weights[block_id] = pooled_weight;
//           
//           // Update block assignments
//           for (size_t j = 0; j < n; ++j) {
//             if (blocks[j] == next_block_id) {
//               blocks[j] = block_id;
//             }
//           }
//           
//           changed = true;
//           break;
//         }
//       }
//     }
//     
//     // Copy back pooled values
//     for (size_t i = 0; i < n; ++i) {
//       values[i] = work_values[blocks[i]];
//     }
//   }
//   
//   // PAVA for decreasing constraint
//   void apply_pava_decreasing(std::vector<double>& values, 
//                              std::vector<double>& weights) {
//     size_t n = values.size();
//     
//     // Create working arrays
//     std::vector<double> work_values = values;
//     std::vector<double> work_weights = weights;
//     std::vector<size_t> blocks(n);
//     for (size_t i = 0; i < n; ++i) blocks[i] = i;
//     
//     bool changed = true;
//     while (changed) {
//       changed = false;
//       
//       for (size_t i = 0; i < n - 1; ++i) {
//         if (blocks[i] == blocks[i + 1]) continue;
//         
//         if (work_values[blocks[i]] < work_values[blocks[i + 1]]) {
//           // Pool adjacent violators
//           double pooled_weight = work_weights[blocks[i]] + 
//             work_weights[blocks[i + 1]];
//           double pooled_value = (work_weights[blocks[i]] * work_values[blocks[i]] + 
//                                  work_weights[blocks[i + 1]] * work_values[blocks[i + 1]]) / 
//                                  pooled_weight;
//           
//           size_t block_id = blocks[i];
//           size_t next_block_id = blocks[i + 1];
//           
//           work_values[block_id] = pooled_value;
//           work_weights[block_id] = pooled_weight;
//           
//           // Update block assignments
//           for (size_t j = 0; j < n; ++j) {
//             if (blocks[j] == next_block_id) {
//               blocks[j] = block_id;
//             }
//           }
//           
//           changed = true;
//           break;
//         }
//       }
//     }
//     
//     // Copy back pooled values
//     for (size_t i = 0; i < n; ++i) {
//       values[i] = work_values[blocks[i]];
//     }
//   }
// };
// 
// // =====================================================================
// // SECTION 9: CHIMERGE IMPLEMENTATIONS
// // =====================================================================
// 
// class ChiMergeAlgorithm {
// protected:
//   int min_bins;
//   int max_bins;
//   double sig_level;
//   double min_size;
//   ChiSquareCache cache;
//   size_t total_samples;
//   
//   void merge_numeric_bins(std::vector<NumericBin>& bins, size_t index) {
//     if (index >= bins.size() - 1) return;
//     
//     CHIMERGE_LOG_DEBUG("Merging bins ", index, " and ", (index + 1));
//     
//     bins[index].upper_bound = bins[index + 1].upper_bound;
//     bins[index].count_0 += bins[index + 1].count_0;
//     bins[index].count_1 += bins[index + 1].count_1;
//     bins[index].weight_sum += bins[index + 1].weight_sum;
//     
//     bins.erase(bins.begin() + index + 1);
//     cache.invalidate_range(index, bins.size());
//   }
//   
//   bool should_stop_merging(const std::vector<NumericBin>& bins, double min_chi) {
//     if (bins.size() <= static_cast<size_t>(min_bins)) {
//       CHIMERGE_LOG_DEBUG("Stopping merge: bins.size() <= min_bins");
//       return true;
//     }
//     
//     double threshold = get_chi_square_threshold(sig_level, 1);
//     if (min_chi >= threshold) {
//       CHIMERGE_LOG_DEBUG("Stopping merge: min_chi >= threshold (", 
//                          min_chi, " >= ", threshold, ")");
//       return true;
//     }
//     
//     for (const auto& bin : bins) {
//       if (bin.total() < min_size * total_samples) {
//         CHIMERGE_LOG_DEBUG("Continuing merge: bin size < min_size threshold");
//         return false;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Stopping merge: all bins meet min_size requirement");
//     return false;
//   }
//   
//   std::vector<NumericBin> create_initial_numeric_bins(const std::vector<DataPoint>& data) {
//     CHIMERGE_LOG_INFO("Creating initial numeric bins from ", data.size(), " data points");
//     CHIMERGE_TIMER("Initial bin creation");
//     
//     std::map<double, NumericBin> unique_bins;
//     
//     for (const auto& point : data) {
//       if (point.is_missing || point.is_special) continue;
//       
//       auto& bin = unique_bins[point.numeric_value];
//       bin.lower_bound = point.numeric_value;
//       bin.upper_bound = point.numeric_value;
//       
//       if (point.target) {
//         bin.count_1++;
//       } else {
//         bin.count_0++;
//       }
//       bin.weight_sum += point.weight;
//     }
//     
//     std::vector<NumericBin> bins;
//     for (const auto& pair : unique_bins) {
//       bins.push_back(pair.second);
//     }
//     
//     CHIMERGE_LOG_DEBUG("Created ", bins.size(), " initial bins");
//     
//     // Adjust boundaries
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (i == 0) {
//         bins[i].lower_bound = -std::numeric_limits<double>::infinity();
//       }
//       if (i == bins.size() - 1) {
//         bins[i].upper_bound = std::numeric_limits<double>::infinity();
//       } else {
//         bins[i].upper_bound = bins[i + 1].lower_bound;
//       }
//     }
//     
//     return bins;
//   }
//   
// public:
//   ChiMergeAlgorithm(int min_b, int max_b, double sig, double min_s, bool use_cache)
//     : min_bins(min_b), max_bins(max_b), sig_level(sig), min_size(min_s),
//       cache(100000, use_cache), total_samples(0) {}
//   
//   virtual std::vector<NumericBin> apply_numeric_binning(
//       const std::vector<DataPoint>& data) = 0;
// };
// 
// class ChiMergeModified : public ChiMergeAlgorithm {
// public:
//   ChiMergeModified(int min_b, int max_b, double sig, double min_s, bool use_cache)
//     : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache) {}
//   
//   std::vector<NumericBin> apply_numeric_binning(
//       const std::vector<DataPoint>& data) override {
//         CHIMERGE_LOG_INFO("Applying ChiMerge Modified algorithm");
//         CHIMERGE_TIMER("ChiMerge Modified");
//         
//         auto bins = create_initial_numeric_bins(data);
//         
//         total_samples = 0;
//         for (const auto& bin : bins) {
//           total_samples += bin.total();
//         }
//         
//         CHIMERGE_LOG_DEBUG("Total samples: ", total_samples);
//         
//         if (bins.size() <= static_cast<size_t>(min_bins)) {
//           CHIMERGE_LOG_DEBUG("Not enough bins for merging, returning initial bins");
//           return bins;
//         }
//         
//         cache.initialize(bins.size());
//         
//         // First pass: merge bins with zero counts
//         CHIMERGE_LOG_DEBUG("First pass: merging bins with zero counts");
//         for (size_t i = 0; i < bins.size() - 1;) {
//           if (bins[i].count_0 == 0 || bins[i].count_1 == 0 ||
//               bins[i+1].count_0 == 0 || bins[i+1].count_1 == 0) {
//             CHIMERGE_LOG_DEBUG("Merging zero-count bins at index ", i);
//             merge_numeric_bins(bins, i);
//           } else {
//             i++;
//           }
//         }
//         
//         // Main merging loop
//         CHIMERGE_LOG_DEBUG("Starting main merging loop");
//         while (bins.size() > static_cast<size_t>(min_bins) && bins.size() > 2) {
//           size_t min_index = 0;
//           double min_chi = std::numeric_limits<double>::max();
//           
//           // Find minimum chi-square pair
//           for (size_t i = 0; i < bins.size() - 1; ++i) {
//             double chi = cache.get_cached_value(i, i + 1);
//             if (chi < 0) {
//               chi = calculate_chi_square_modified(bins[i], bins[i + 1]);
//               cache.update_cache(i, i + 1, chi);
//             }
//             
//             if (chi < min_chi) {
//               min_chi = chi;
//               min_index = i;
//             }
//           }
//           
//           CHIMERGE_LOG_DEBUG("Min chi-square: ", min_chi, " at index ", min_index);
//           
//           if (should_stop_merging(bins, min_chi)) {
//             CHIMERGE_LOG_DEBUG("Stopping criteria met, breaking merge loop");
//             break;
//           }
//           
//           merge_numeric_bins(bins, min_index);
//           
//           if (bins.size() <= static_cast<size_t>(max_bins)) {
//             CHIMERGE_LOG_DEBUG("Reached max_bins limit, breaking merge loop");
//             break;
//           }
//         }
//         
//         // Force reduction if still above max_bins
//         while (bins.size() > static_cast<size_t>(max_bins)) {
//           CHIMERGE_LOG_DEBUG("Forcing bin reduction, current bins: ", bins.size());
//           size_t min_index = 0;
//           double min_chi = std::numeric_limits<double>::max();
//           
//           for (size_t i = 0; i < bins.size() - 1; ++i) {
//             double chi = calculate_chi_square_modified(bins[i], bins[i + 1]);
//             if (chi < min_chi) {
//               min_chi = chi;
//               min_index = i;
//             }
//           }
//           
//           merge_numeric_bins(bins, min_index);
//         }
//         
//         CHIMERGE_LOG_INFO("ChiMerge Modified completed with ", bins.size(), " bins");
//         return bins;
//       }
// };
// 
// // =====================================================================
// // SECTION 10: CATEGORICAL BINNING
// // =====================================================================
// 
// std::vector<CategoryStats> calculate_category_statistics(
//     const std::vector<DataPoint>& data) {
//   CHIMERGE_LOG_INFO("Calculating category statistics for ", data.size(), " data points");
//   CHIMERGE_TIMER("Category statistics");
//   
//   std::map<std::string, CategoryStats> stats_map;
//   
//   for (const auto& point : data) {
//     if (point.is_missing || point.is_special) continue;
//     
//     auto& stats = stats_map[point.categorical_value];
//     stats.category = point.categorical_value;
//     if (point.target) {
//       stats.count_1++;
//     } else {
//       stats.count_0++;
//     }
//     stats.weight_sum += point.weight;
//   }
//   
//   std::vector<CategoryStats> stats_vector;
//   for (const auto& pair : stats_map) {
//     CategoryStats stats = pair.second;
//     stats.event_rate = stats.total() > 0 ? 
//     static_cast<double>(stats.count_1) / stats.total() : 0.0;
//     stats_vector.push_back(stats);
//   }
//   
//   // Sort by event rate
//   std::sort(stats_vector.begin(), stats_vector.end(),
//             [](const CategoryStats& a, const CategoryStats& b) {
//               return a.event_rate < b.event_rate;
//             });
//   
//   CHIMERGE_LOG_DEBUG("Calculated statistics for ", stats_vector.size(), " categories");
//   return stats_vector;
// }
// 
// std::vector<CategoricalBin> apply_categorical_binning(
//     const std::vector<DataPoint>& data,
//     int min_bins, int max_bins, double min_size,
//     int max_cat, double rare_pct, const std::string& cat_sep) {
//   
//   CHIMERGE_LOG_INFO("Applying categorical binning");
//   CHIMERGE_TIMER("Categorical binning");
//   
//   auto category_stats = calculate_category_statistics(data);
//   
//   size_t total_count = 0;
//   for (const auto& stats : category_stats) {
//     total_count += stats.total();
//   }
//   
//   CHIMERGE_LOG_DEBUG("Total count: ", total_count);
//   
//   std::vector<CategoricalBin> bins;
//   CategoricalBin rare_bin;
//   rare_bin.count_0 = 0;
//   rare_bin.count_1 = 0;
//   rare_bin.weight_sum = 0;
//   
//   for (const auto& stats : category_stats) {
//     double frequency = static_cast<double>(stats.total()) / total_count;
//     
//     if (frequency < rare_pct && bins.size() + 1 > static_cast<size_t>(max_cat)) {
//       rare_bin.categories.push_back(stats.category);
//       rare_bin.count_0 += stats.count_0;
//       rare_bin.count_1 += stats.count_1;
//       rare_bin.weight_sum += stats.weight_sum;
//       CHIMERGE_LOG_DEBUG("Adding category ", stats.category, " to rare bin");
//     } else {
//       CategoricalBin bin;
//       bin.categories.push_back(stats.category);
//       bin.count_0 = stats.count_0;
//       bin.count_1 = stats.count_1;
//       bin.weight_sum = stats.weight_sum;
//       bins.push_back(bin);
//       CHIMERGE_LOG_DEBUG("Creating bin for category ", stats.category);
//     }
//   }
//   
//   if (!rare_bin.categories.empty() && rare_bin.total() > 0) {
//     bins.push_back(rare_bin);
//     CHIMERGE_LOG_DEBUG("Added rare bin with ", rare_bin.categories.size(), " categories");
//   }
//   
//   // Merge bins to meet max_bins constraint
//   while (bins.size() > static_cast<size_t>(max_bins)) {
//     double min_chi = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
//     
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       NumericBin temp1, temp2;
//       temp1.count_0 = bins[i].count_0;
//       temp1.count_1 = bins[i].count_1;
//       temp2.count_0 = bins[i + 1].count_0;
//       temp2.count_1 = bins[i + 1].count_1;
//       
//       double chi = calculate_chi_square(temp1, temp2);
//       if (chi < min_chi) {
//         min_chi = chi;
//         merge_index = i;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Merging bins ", merge_index, " and ", (merge_index + 1), 
//                        " with chi: ", min_chi);
//     
//     bins[merge_index].categories.insert(bins[merge_index].categories.end(),
//                                         bins[merge_index + 1].categories.begin(),
//                                         bins[merge_index + 1].categories.end());
//     bins[merge_index].count_0 += bins[merge_index + 1].count_0;
//     bins[merge_index].count_1 += bins[merge_index + 1].count_1;
//     bins[merge_index].weight_sum += bins[merge_index + 1].weight_sum;
//     bins.erase(bins.begin() + merge_index + 1);
//   }
//   
//   // Format labels
//   for (auto& bin : bins) {
//     bin.label = format_categorical_bin_label(bin, cat_sep);
//   }
//   
//   CHIMERGE_LOG_INFO("Categorical binning completed with ", bins.size(), " bins");
//   return bins;
// }
// 
// // =====================================================================
// // SECTION 11: SPECIAL VALUE HANDLING
// // =====================================================================
// 
// class SpecialValueHandler {
// private:
//   MissingPolicy miss_policy;
//   std::vector<double> special_numeric;
//   std::vector<std::string> special_string;
//   
// public:
//   SpecialValueHandler(const std::string& policy,
//                       const std::vector<double>& special_num,
//                       const std::vector<std::string>& special_str) {
//     if (policy == "separate") miss_policy = MissingPolicy::SEPARATE;
//     else if (policy == "remove") miss_policy = MissingPolicy::REMOVE;
//     else if (policy == "impute") miss_policy = MissingPolicy::IMPUTE;
//     else miss_policy = MissingPolicy::MERGE;
//     
//     special_numeric = special_num;
//     special_string = special_str;
//     
//     CHIMERGE_LOG_DEBUG("SpecialValueHandler initialized with policy: ", policy);
//   }
//   
//   std::vector<SpecialBin> process_special_values(const std::vector<DataPoint>& data) {
//     CHIMERGE_LOG_INFO("Processing special values");
//     CHIMERGE_TIMER("Special value processing");
//     
//     std::vector<SpecialBin> special_bins;
//     
//     if (miss_policy != MissingPolicy::SEPARATE) {
//       CHIMERGE_LOG_DEBUG("Missing policy not 'separate', skipping special bin creation");
//       return special_bins;
//     }
//     
//     // Handle missing values
//     SpecialBin missing_bin;
//     missing_bin.type = "missing";
//     missing_bin.label = "[Missing]";
//     missing_bin.count_0 = 0;
//     missing_bin.count_1 = 0;
//     missing_bin.weight_sum = 0;
//     
//     for (const auto& point : data) {
//       if (point.is_missing) {
//         if (point.target) {
//           missing_bin.count_1++;
//         } else {
//           missing_bin.count_0++;
//         }
//         missing_bin.weight_sum += point.weight;
//       }
//     }
//     
//     if (missing_bin.total() > 0) {
//       special_bins.push_back(missing_bin);
//       CHIMERGE_LOG_DEBUG("Created missing bin with ", missing_bin.total(), " samples");
//     }
//     
//     // Handle special numeric values
//     for (double special_val : special_numeric) {
//       SpecialBin special_bin;
//       special_bin.type = "special_numeric";
//       special_bin.label = "[Special:" + std::to_string(static_cast<int>(special_val)) + "]";
//       special_bin.count_0 = 0;
//       special_bin.count_1 = 0;
//       special_bin.weight_sum = 0;
//       
//       for (const auto& point : data) {
//         if (point.is_special && std::abs(point.numeric_value - special_val) < 1e-9) {
//           if (point.target) {
//             special_bin.count_1++;
//           } else {
//             special_bin.count_0++;
//           }
//           special_bin.weight_sum += point.weight;
//         }
//       }
//       
//       if (special_bin.total() > 0) {
//         special_bins.push_back(special_bin);
//         CHIMERGE_LOG_DEBUG("Created special bin for value ", special_val, 
//                            " with ", special_bin.total(), " samples");
//       }
//     }
//     
//     CHIMERGE_LOG_INFO("Processed ", special_bins.size(), " special bins");
//     return special_bins;
//   }
// };
// 
// // =====================================================================
// // SECTION 12: DATA PREPARATION AND EXTRACTION
// // =====================================================================
// 
// std::vector<DataPoint> prepare_numeric_data(SEXP x, const LogicalVector& target,
//                                             const NumericVector& weights,
//                                             const std::vector<double>& special_vals) {
//   CHIMERGE_LOG_INFO("Preparing numeric data");
//   CHIMERGE_TIMER("Numeric data preparation");
//   
//   std::vector<DataPoint> data;
//   bool has_weights = weights.size() > 0;
//   
//   if (Rf_isInteger(x)) {
//     IntegerVector iv = as<IntegerVector>(x);
//     CHIMERGE_LOG_DEBUG("Processing integer vector with ", iv.size(), " elements");
//     for (int i = 0; i < iv.size(); ++i) {
//       DataPoint point;
//       point.target = target[i];
//       point.original_index = i;
//       point.weight = has_weights ? weights[i] : 1.0;
//       
//       if (IntegerVector::is_na(iv[i])) {
//         point.is_missing = true;
//         CHIMERGE_LOG_TRACE("Missing value at index ", i);
//       } else {
//         point.numeric_value = iv[i];
//         for (double special : special_vals) {
//           if (std::abs(point.numeric_value - special) < 1e-9) {
//             point.is_special = true;
//             CHIMERGE_LOG_TRACE("Special value ", special, " at index ", i);
//             break;
//           }
//         }
//       }
//       data.push_back(point);
//     }
//   } else if (Rf_isNumeric(x)) {
//     NumericVector nv = as<NumericVector>(x);
//     CHIMERGE_LOG_DEBUG("Processing numeric vector with ", nv.size(), " elements");
//     for (int i = 0; i < nv.size(); ++i) {
//       DataPoint point;
//       point.target = target[i];
//       point.original_index = i;
//       point.weight = has_weights ? weights[i] : 1.0;
//       
//       if (NumericVector::is_na(nv[i])) {
//         point.is_missing = true;
//         CHIMERGE_LOG_TRACE("Missing value at index ", i);
//       } else {
//         point.numeric_value = nv[i];
//         for (double special : special_vals) {
//           if (std::abs(point.numeric_value - special) < 1e-9) {
//             point.is_special = true;
//             CHIMERGE_LOG_TRACE("Special value ", special, " at index ", i);
//             break;
//           }
//         }
//       }
//       data.push_back(point);
//     }
//   }
//   
//   CHIMERGE_LOG_INFO("Prepared ", data.size(), " numeric data points");
//   return data;
// }
// 
// std::vector<DataPoint> prepare_categorical_data(SEXP x, const LogicalVector& target,
//                                                 const NumericVector& weights) {
//   CHIMERGE_LOG_INFO("Preparing categorical data");
//   CHIMERGE_TIMER("Categorical data preparation");
//   
//   std::vector<DataPoint> data;
//   bool has_weights = weights.size() > 0;
//   
//   if (Rf_isFactor(x)) {
//     IntegerVector iv = as<IntegerVector>(x);
//     CharacterVector levels = iv.attr("levels");
//     CHIMERGE_LOG_DEBUG("Processing factor with ", iv.size(), 
//                        " elements and ", levels.size(), " levels");
//     
//     for (int i = 0; i < iv.size(); ++i) {
//       DataPoint point;
//       point.target = target[i];
//       point.original_index = i;
//       point.weight = has_weights ? weights[i] : 1.0;
//       
//       if (IntegerVector::is_na(iv[i])) {
//         point.is_missing = true;
//         CHIMERGE_LOG_TRACE("Missing factor value at index ", i);
//       } else {
//         int level_idx = iv[i] - 1;  // R uses 1-based indexing
//         if (level_idx >= 0 && level_idx < levels.size()) {
//           point.categorical_value = as<std::string>(levels[level_idx]);
//           CHIMERGE_LOG_TRACE("Factor level ", point.categorical_value, 
//                              " at index ", i);
//         } else {
//           point.is_missing = true;
//           CHIMERGE_LOG_TRACE("Invalid factor level at index ", i);
//         }
//       }
//       data.push_back(point);
//     }
//   } else if (Rf_isString(x)) {
//     CharacterVector cv = as<CharacterVector>(x);
//     CHIMERGE_LOG_DEBUG("Processing character vector with ", cv.size(), " elements");
//     for (int i = 0; i < cv.size(); ++i) {
//       DataPoint point;
//       point.target = target[i];
//       point.original_index = i;
//       point.weight = has_weights ? weights[i] : 1.0;
//       
//       if (CharacterVector::is_na(cv[i])) {
//         point.is_missing = true;
//         CHIMERGE_LOG_TRACE("Missing character value at index ", i);
//       } else {
//         point.categorical_value = as<std::string>(cv[i]);
//         CHIMERGE_LOG_TRACE("Character value ", point.categorical_value, 
//                            " at index ", i);
//       }
//       data.push_back(point);
//     }
//   }
//   
//   CHIMERGE_LOG_INFO("Prepared ", data.size(), " categorical data points");
//   return data;
// }
// 
// // =====================================================================
// // SECTION 13: TRANSFORMATION APPLICATION
// // =====================================================================
// 
// NumericVector apply_numeric_woe_transform(SEXP x, const std::vector<NumericBin>& bins,
//                                           const std::vector<SpecialBin>& special_bins,
//                                           const std::vector<double>& special_vals) {
//   CHIMERGE_LOG_INFO("Applying numeric WoE transformation");
//   CHIMERGE_TIMER("Numeric WoE transform");
//   
//   NumericVector result;
//   
//   if (Rf_isInteger(x)) {
//     IntegerVector iv = as<IntegerVector>(x);
//     result = NumericVector(iv.size(), NA_REAL);
//     
//     for (int i = 0; i < iv.size(); ++i) {
//       if (IntegerVector::is_na(iv[i])) {
//         for (const auto& sbin : special_bins) {
//           if (sbin.type == "missing") {
//             result[i] = sbin.woe;
//             CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i);
//             break;
//           }
//         }
//       } else {
//         double val = iv[i];
//         bool found = false;
//         
//         // Check special values
//         for (double special : special_vals) {
//           if (std::abs(val - special) < 1e-9) {
//             for (const auto& sbin : special_bins) {
//               if (sbin.type == "special_numeric") {
//                 result[i] = sbin.woe;
//                 found = true;
//                 CHIMERGE_LOG_TRACE("Applied special WoE at index ", i);
//                 break;
//               }
//             }
//             break;
//           }
//         }
//         
//         // Find appropriate bin
//         if (!found) {
//           for (const auto& bin : bins) {
//             if (val >= bin.lower_bound && val < bin.upper_bound) {
//               result[i] = bin.woe;
//               CHIMERGE_LOG_TRACE("Applied regular WoE at index ", i);
//               break;
//             }
//           }
//         }
//       }
//     }
//   } else if (Rf_isNumeric(x)) {
//     NumericVector nv = as<NumericVector>(x);
//     result = NumericVector(nv.size(), NA_REAL);
//     
//     for (int i = 0; i < nv.size(); ++i) {
//       if (NumericVector::is_na(nv[i])) {
//         for (const auto& sbin : special_bins) {
//           if (sbin.type == "missing") {
//             result[i] = sbin.woe;
//             CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i);
//             break;
//           }
//         }
//       } else {
//         double val = nv[i];
//         bool found = false;
//         
//         // Check special values
//         for (double special : special_vals) {
//           if (std::abs(val - special) < 1e-9) {
//             for (const auto& sbin : special_bins) {
//               if (sbin.type == "special_numeric") {
//                 result[i] = sbin.woe;
//                 found = true;
//                 CHIMERGE_LOG_TRACE("Applied special WoE at index ", i);
//                 break;
//               }
//             }
//             break;
//           }
//         }
//         
//         // Find appropriate bin
//         if (!found) {
//           for (const auto& bin : bins) {
//             if (val >= bin.lower_bound && val < bin.upper_bound) {
//               result[i] = bin.woe;
//               CHIMERGE_LOG_TRACE("Applied regular WoE at index ", i);
//               break;
//             }
//           }
//         }
//       }
//     }
//   }
//   
//   CHIMERGE_LOG_INFO("Applied WoE transformation to ", result.size(), " values");
//   return result;
// }
// 
// CharacterVector apply_numeric_bin_transform(SEXP x, const std::vector<NumericBin>& bins,
//                                             const std::vector<SpecialBin>& special_bins,
//                                             const std::vector<double>& special_vals) {
//   CHIMERGE_LOG_INFO("Applying numeric bin transformation");
//   CHIMERGE_TIMER("Numeric bin transform");
//   
//   CharacterVector result;
//   
//   if (Rf_isInteger(x)) {
//     IntegerVector iv = as<IntegerVector>(x);
//     result = CharacterVector(iv.size());
//     
//     for (int i = 0; i < iv.size(); ++i) {
//       if (IntegerVector::is_na(iv[i])) {
//         result[i] = "[Missing]";
//         CHIMERGE_LOG_TRACE("Applied missing bin label at index ", i);
//       } else {
//         double val = iv[i];
//         bool found = false;
//         
//         // Check special values
//         for (double special : special_vals) {
//           if (std::abs(val - special) < 1e-9) {
//             result[i] = "[Special:" + std::to_string(static_cast<int>(special)) + "]";
//             found = true;
//             CHIMERGE_LOG_TRACE("Applied special bin label at index ", i);
//             break;
//           }
//         }
//         
//         // Find appropriate bin
//         if (!found) {
//           for (const auto& bin : bins) {
//             if (val >= bin.lower_bound && val < bin.upper_bound) {
//               result[i] = bin.label;
//               CHIMERGE_LOG_TRACE("Applied regular bin label at index ", i);
//               break;
//             }
//           }
//         }
//       }
//     }
//   } else if (Rf_isNumeric(x)) {
//     NumericVector nv = as<NumericVector>(x);
//     result = CharacterVector(nv.size());
//     
//     for (int i = 0; i < nv.size(); ++i) {
//       if (NumericVector::is_na(nv[i])) {
//         result[i] = "[Missing]";
//         CHIMERGE_LOG_TRACE("Applied missing bin label at index ", i);
//       } else {
//         double val = nv[i];
//         bool found = false;
//         
//         // Check special values
//         for (double special : special_vals) {
//           if (std::abs(val - special) < 1e-9) {
//             result[i] = "[Special:" + std::to_string(static_cast<int>(special)) + "]";
//             found = true;
//             CHIMERGE_LOG_TRACE("Applied special bin label at index ", i);
//             break;
//           }
//         }
//         
//         // Find appropriate bin
//         if (!found) {
//           for (const auto& bin : bins) {
//             if (val >= bin.lower_bound && val < bin.upper_bound) {
//               result[i] = bin.label;
//               CHIMERGE_LOG_TRACE("Applied regular bin label at index ", i);
//               break;
//             }
//           }
//         }
//       }
//     }
//   }
//   
//   CHIMERGE_LOG_INFO("Applied bin transformation to ", result.size(), " values");
//   return result;
// }
// 
// // =====================================================================
// // SECTION 14: KS AND GINI CALCULATION
// // =====================================================================
// 
// void calculate_ks_gini_for_bins(std::vector<double>& ks_values,
//                                 std::vector<double>& gini_values,
//                                 const std::vector<int>& neg_counts,
//                                 const std::vector<int>& pos_counts) {
//   CHIMERGE_LOG_INFO("Calculating KS and Gini values");
//   CHIMERGE_TIMER("KS/Gini calculation");
//   
//   size_t n_bins = neg_counts.size();
//   if (n_bins == 0) {
//     CHIMERGE_LOG_WARNING("Empty bins vector, skipping KS/Gini calculation");
//     return;
//   }
//   
//   // Calculate totals
//   double total_neg = 0, total_pos = 0;
//   for (size_t i = 0; i < n_bins; ++i) {
//     total_neg += neg_counts[i];
//     total_pos += pos_counts[i];
//   }
//   
//   CHIMERGE_LOG_DEBUG("Total negatives: ", total_neg, ", positives: ", total_pos);
//   
//   // Calculate cumulative distributions
//   double cum_neg = 0, cum_pos = 0;
//   for (size_t i = 0; i < n_bins; ++i) {
//     cum_neg += neg_counts[i];
//     cum_pos += pos_counts[i];
//     
//     double cum_neg_rate = total_neg > 0 ? cum_neg / total_neg : 0;
//     double cum_pos_rate = total_pos > 0 ? cum_pos / total_pos : 0;
//     
//     // KS for this bin
//     double ks = std::abs(cum_pos_rate - cum_neg_rate);
//     ks_values.push_back(ks);
//     
//     // Gini coefficient for this bin
//     double total_bin = neg_counts[i] + pos_counts[i];
//     double gini = 0.0;
//     if (total_bin > 0) {
//       double p1 = pos_counts[i] / total_bin;
//       double p0 = neg_counts[i] / total_bin;
//       gini = std::abs(p1 - p0);
//     }
//     gini_values.push_back(gini);
//     
//     CHIMERGE_LOG_TRACE("Bin ", i, " - KS: ", ks, ", Gini: ", gini);
//   }
//   
//   CHIMERGE_LOG_INFO("Calculated KS and Gini for ", ks_values.size(), " bins");
// }
// 
// // =====================================================================
// // SECTION 15: MAIN BINNING ORCHESTRATOR
// // =====================================================================
// 
// class OptimalBinningOrchestrator {
// public:
//   List process_numeric_feature(
//       SEXP x,
//       const std::string& feature_name,
//       const LogicalVector& target,
//       const NumericVector& weights,
//       int min_bins, int max_bins, double sig_level, double min_size,
//       double smooth, bool monotonic, double min_iv, int digits,
//       const std::string& miss_policy,
//       const std::vector<double>& special_vals,
//       const std::string& method_str,
//       bool cache) {
//     
//     CHIMERGE_LOG_INFO("Processing numeric feature: ", feature_name);
//     CHIMERGE_TIMER("Numeric feature processing");
//     
//     std::vector<std::string> messages;
//     
//     // Prepare data
//     auto data = prepare_numeric_data(x, target, weights, special_vals);
//     
//     // Sort data
//     CHIMERGE_LOG_DEBUG("Sorting data");
//     std::sort(data.begin(), data.end(),
//               [](const DataPoint& a, const DataPoint& b) {
//                 if (a.is_missing != b.is_missing) return !a.is_missing;
//                 if (a.is_special != b.is_special) return !a.is_special;
//                 return a.numeric_value < b.numeric_value;
//               });
//     
//     // Apply binning
//     CHIMERGE_LOG_INFO("Applying binning algorithm: ", method_str);
//     ChiMergeModified algo(min_bins, max_bins, sig_level, min_size, cache);
//     auto bins = algo.apply_numeric_binning(data);
//     
//     if (bins.empty()) {
//       messages.push_back("No valid bins created");
//       CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
//     }
//     
//     // Process special values
//     CHIMERGE_LOG_DEBUG("Processing special values");
//     SpecialValueHandler special_handler(miss_policy, special_vals, {});
//     auto special_bins = special_handler.process_special_values(data);
//     
//     // Calculate WoE
//     size_t total_pos = 0, total_neg = 0;
//     for (const auto& point : data) {
//       if (!point.is_missing || miss_policy != "remove") {
//         if (point.target) total_pos++;
//         else total_neg++;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
//     
//     // Calculate WoE for regular bins
//     WoECalculator woe_calc(total_pos, total_neg, smooth, monotonic);
//     woe_calc.calculate_woe_iv(bins);
//     
//     // Calculate WoE for special bins
//     size_t total_bins = bins.size() + special_bins.size();
//     for (auto& sbin : special_bins) {
//       double smoothed_pos = sbin.count_1 + smooth;
//       double smoothed_neg = sbin.count_0 + smooth;
//       double smoothed_total_pos = total_pos + total_bins * smooth;
//       double smoothed_total_neg = total_neg + total_bins * smooth;
//       
//       if (smoothed_total_pos > 0 && smoothed_total_neg > 0) {
//         double dist_pos = smoothed_pos / smoothed_total_pos;
//         double dist_neg = smoothed_neg / smoothed_total_neg;
//         
//         if (dist_pos > 0 && dist_neg > 0) {
//           sbin.woe = std::log(dist_pos / dist_neg);
//           if (sbin.woe > 5) sbin.woe = 5;
//           if (sbin.woe < -5) sbin.woe = -5;
//           sbin.iv_contribution = (dist_pos - dist_neg) * sbin.woe;
//         }
//       }
//     }
//     
//     // Format labels
//     CHIMERGE_LOG_DEBUG("Formatting bin labels");
//     for (size_t i = 0; i < bins.size(); ++i) {
//       bins[i].label = format_numeric_bin_label(bins[i], i == 0, 
//                                                i == bins.size() - 1, digits);
//     }
//     
//     // Combine regular bins and special bins for output
//     std::vector<int> bin_ids;
//     std::vector<std::string> bin_labels;
//     std::vector<int> total_counts, neg_counts, pos_counts;
//     std::vector<double> woes, ivs;
//     
//     // Add special bins first
//     int bin_id = 1;
//     for (const auto& sbin : special_bins) {
//       bin_ids.push_back(bin_id++);
//       bin_labels.push_back(sbin.label);
//       total_counts.push_back(sbin.total());
//       neg_counts.push_back(sbin.count_0);
//       pos_counts.push_back(sbin.count_1);
//       woes.push_back(sbin.woe);
//       ivs.push_back(sbin.iv_contribution);
//     }
//     
//     // Add regular bins
//     for (const auto& bin : bins) {
//       bin_ids.push_back(bin_id++);
//       bin_labels.push_back(bin.label);
//       total_counts.push_back(bin.total());
//       neg_counts.push_back(bin.count_0);
//       pos_counts.push_back(bin.count_1);
//       woes.push_back(bin.woe);
//       ivs.push_back(bin.iv_contribution);
//     }
//     
//     // Calculate KS and Gini
//     std::vector<double> ks_values, gini_values;
//     calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
//     
//     DataFrame bins_df = DataFrame::create(
//       Named("bin_id") = bin_ids,
//       Named("bin_label") = bin_labels,
//       Named("total_count") = total_counts,
//       Named("neg_count") = neg_counts,
//       Named("pos_count") = pos_counts,
//       Named("woe") = woes,
//       Named("iv") = ivs,
//       Named("ks") = ks_values,
//       Named("gini") = gini_values
//     );
//     
//     // Create transformation DataFrame
//     NumericVector woe_transform = apply_numeric_woe_transform(x, bins, 
//                                                               special_bins, special_vals);
//     CharacterVector bin_transform = apply_numeric_bin_transform(x, bins, 
//                                                                 special_bins, special_vals);
//     
//     DataFrame transform_df = DataFrame::create(
//       Named(feature_name) = x,
//       Named(feature_name + "_bin") = bin_transform,
//       Named(feature_name + "_woe") = woe_transform
//     );
//     
//     // Detect variable type
//     VariableType var_type = detect_r_variable_type(x);
//     std::string var_type_str;
//     switch (var_type) {
//     case VariableType::NUMERIC_CONTINUOUS: 
//       var_type_str = "numeric_continuous"; 
//       break;
//     case VariableType::NUMERIC_DISCRETE: 
//       var_type_str = "numeric_discrete"; 
//       break;
//     case VariableType::BOOLEAN: 
//       var_type_str = "boolean"; 
//       break;
//     default: 
//       var_type_str = "numeric";
//     }
//     
//     // Calculate total IV including special bins
//     double total_iv = woe_calc.get_total_iv();
//     for (const auto& sbin : special_bins) {
//       total_iv += sbin.iv_contribution;
//     }
//     
//     CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, 
//                       ", Bins: ", bin_ids.size());
//     
//     // Return results
//     List result = List::create(
//       Named("bins") = bins_df,
//       Named("total_iv") = total_iv,
//       Named("variable_type") = var_type_str,
//       Named("method_used") = method_str,
//       Named("n_bins") = bin_ids.size(),
//       Named("total_samples") = static_cast<int>(data.size()),
//       Named("event_rate") = static_cast<double>(total_pos) / (total_pos + total_neg),
//       Named("messages") = messages.empty() ? "Success" : messages[0],
//                                                                  Named("transform") = transform_df
//     );
//     
//     return result;
//   }
//   
//   List process_categorical_feature(
//       SEXP x,
//       const std::string& feature_name,
//       const LogicalVector& target,
//       const NumericVector& weights,
//       int min_bins, int max_bins, double min_size,
//       double smooth, bool monotonic, double min_iv,
//       const std::string& miss_policy,
//       int max_cat, double rare_pct, const std::string& cat_sep) {
//     
//     CHIMERGE_LOG_INFO("Processing categorical feature: ", feature_name);
//     CHIMERGE_TIMER("Categorical feature processing");
//     
//     std::vector<std::string> messages;
//     
//     // Prepare data
//     auto data = prepare_categorical_data(x, target, weights);
//     
//     // Apply binning
//     CHIMERGE_LOG_DEBUG("Applying categorical binning");
//     auto bins = apply_categorical_binning(data, min_bins, max_bins, min_size,
//                                           max_cat, rare_pct, cat_sep);
//     
//     if (bins.empty()) {
//       messages.push_back("No valid bins created");
//       CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
//     }
//     
//     // Process special values
//     CHIMERGE_LOG_DEBUG("Processing special values");
//     SpecialValueHandler special_handler(miss_policy, {}, {});
//     auto special_bins = special_handler.process_special_values(data);
//     
//     // Calculate WoE
//     size_t total_pos = 0, total_neg = 0;
//     for (const auto& point : data) {
//       if (!point.is_missing || miss_policy != "remove") {
//         if (point.target) total_pos++;
//         else total_neg++;
//       }
//     }
//     
//     CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
//     
//     // Calculate WoE for regular bins
//     WoECalculator woe_calc(total_pos, total_neg, smooth, monotonic);
//     woe_calc.calculate_woe_iv(bins);
//     
//     // Calculate WoE for special bins
//     size_t total_bins = bins.size() + special_bins.size();
//     for (auto& sbin : special_bins) {
//       double smoothed_pos = sbin.count_1 + smooth;
//       double smoothed_neg = sbin.count_0 + smooth;
//       double smoothed_total_pos = total_pos + total_bins * smooth;
//       double smoothed_total_neg = total_neg + total_bins * smooth;
//       
//       if (smoothed_total_pos > 0 && smoothed_total_neg > 0) {
//         double dist_pos = smoothed_pos / smoothed_total_pos;
//         double dist_neg = smoothed_neg / smoothed_total_neg;
//         
//         if (dist_pos > 0 && dist_neg > 0) {
//           sbin.woe = std::log(dist_pos / dist_neg);
//           if (sbin.woe > 5) sbin.woe = 5;
//           if (sbin.woe < -5) sbin.woe = -5;
//           sbin.iv_contribution = (dist_pos - dist_neg) * sbin.woe;
//         }
//       }
//     }
//     
//     // Combine special bins and regular bins
//     std::vector<int> bin_ids;
//     std::vector<std::string> bin_labels;
//     std::vector<int> total_counts, neg_counts, pos_counts;
//     std::vector<double> woes, ivs;
//     
//     // Add special bins first
//     int bin_id = 1;
//     for (const auto& sbin : special_bins) {
//       bin_ids.push_back(bin_id++);
//       bin_labels.push_back(sbin.label);
//       total_counts.push_back(sbin.total());
//       neg_counts.push_back(sbin.count_0);
//       pos_counts.push_back(sbin.count_1);
//       woes.push_back(sbin.woe);
//       ivs.push_back(sbin.iv_contribution);
//     }
//     
//     // Add regular bins
//     for (const auto& bin : bins) {
//       bin_ids.push_back(bin_id++);
//       bin_labels.push_back(bin.label);
//       total_counts.push_back(bin.total());
//       neg_counts.push_back(bin.count_0);
//       pos_counts.push_back(bin.count_1);
//       woes.push_back(bin.woe);
//       ivs.push_back(bin.iv_contribution);
//     }
//     
//     // Calculate KS and Gini
//     std::vector<double> ks_values, gini_values;
//     calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
//     
//     DataFrame bins_df = DataFrame::create(
//       Named("bin_id") = bin_ids,
//       Named("bin_label") = bin_labels,
//       Named("total_count") = total_counts,
//       Named("neg_count") = neg_counts,
//       Named("pos_count") = pos_counts,
//       Named("woe") = woes,
//       Named("iv") = ivs,
//       Named("ks") = ks_values,
//       Named("gini") = gini_values
//     );
//     
//     // Create transformation vectors
//     int n = Rf_length(x);
//     NumericVector woe_transform(n, NA_REAL);
//     CharacterVector bin_transform(n);
//     
//     // Create category to WoE mapping
//     std::map<std::string, double> cat_woe_map;
//     std::map<std::string, std::string> cat_bin_map;
//     
//     // Map regular categories
//     for (const auto& bin : bins) {
//       for (const auto& cat : bin.categories) {
//         cat_woe_map[cat] = bin.woe;
//         cat_bin_map[cat] = bin.label;
//       }
//     }
//     
//     // Apply transformations
//     if (Rf_isFactor(x)) {
//       IntegerVector iv = as<IntegerVector>(x);
//       CharacterVector levels = iv.attr("levels");
//       
//       for (int i = 0; i < n; ++i) {
//         if (IntegerVector::is_na(iv[i])) {
//           bin_transform[i] = "[Missing]";
//           // Find missing bin WoE
//           for (const auto& sbin : special_bins) {
//             if (sbin.type == "missing") {
//               woe_transform[i] = sbin.woe;
//               CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i, " for factor");
//               break;
//             }
//           }
//         } else {
//           int level_idx = iv[i] - 1;
//           if (level_idx >= 0 && level_idx < levels.size()) {
//             std::string cat = as<std::string>(levels[level_idx]);
//             auto woe_it = cat_woe_map.find(cat);
//             auto bin_it = cat_bin_map.find(cat);
//             
//             if (woe_it != cat_woe_map.end()) {
//               woe_transform[i] = woe_it->second;
//               bin_transform[i] = bin_it->second;
//               CHIMERGE_LOG_TRACE("Applied WoE for category ", cat, " at index ", i);
//             }
//           }
//         }
//       }
//     } else if (Rf_isString(x)) {
//       CharacterVector cv = as<CharacterVector>(x);
//       
//       for (int i = 0; i < n; ++i) {
//         if (CharacterVector::is_na(cv[i])) {
//           bin_transform[i] = "[Missing]";
//           // Find missing bin WoE
//           for (const auto& sbin : special_bins) {
//             if (sbin.type == "missing") {
//               woe_transform[i] = sbin.woe;
//               CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i, " for character");
//               break;
//             }
//           }
//         } else {
//           std::string cat = as<std::string>(cv[i]);
//           auto woe_it = cat_woe_map.find(cat);
//           auto bin_it = cat_bin_map.find(cat);
//           
//           if (woe_it != cat_woe_map.end()) {
//             woe_transform[i] = woe_it->second;
//             bin_transform[i] = bin_it->second;
//             CHIMERGE_LOG_TRACE("Applied WoE for category ", cat, " at index ", i);
//           }
//         }
//       }
//     }
//     
//     // Create transformation DataFrame
//     DataFrame transform_df = DataFrame::create(
//       Named(feature_name) = x,
//       Named(feature_name + "_bin") = bin_transform,
//       Named(feature_name + "_woe") = woe_transform
//     );
//     
//     // Calculate total IV including special bins
//     double total_iv = woe_calc.get_total_iv();
//     for (const auto& sbin : special_bins) {
//       total_iv += sbin.iv_contribution;
//     }
//     
//     CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, 
//                       ", Bins: ", bin_ids.size());
//     
//     // Return results
//     List result = List::create(
//       Named("bins") = bins_df,
//       Named("total_iv") = total_iv,
//       Named("variable_type") = "categorical",
//       Named("method_used") = "categorical_chimerge",
//       Named("n_bins") = bin_ids.size(),
//       Named("total_samples") = static_cast<int>(data.size()),
//       Named("event_rate") = static_cast<double>(total_pos) / (total_pos + total_neg),
//       Named("messages") = messages.empty() ? "Success" : messages[0],
//                                                                  Named("transform") = transform_df
//     );
//     
//     return result;
//   }
// };
// 
// // =====================================================================
// // SECTION 16: MAIN EXPORTED FUNCTION
// // =====================================================================
// 
// //' ChiMerge Optimal Binning with Weight of Evidence (WoE)
// //'
// //' @description
// //' Performs optimal binning on numeric and categorical variables using ChiMerge algorithm
// //' variations, computing Weight of Evidence (WoE) and Information Value (IV) for each bin.
// //'
// //' @param data DataFrame containing features and target variable
// //' @param target_col String specifying the name of the binary target column
// //' @param feature_cols CharacterVector of feature column names to process
// //' @param min_bins Integer minimum number of bins (default: 2)
// //' @param max_bins Integer maximum number of bins (default: 5)
// //' @param sig_level Numeric significance level for chi-square test (default: 0.05)
// //' @param min_size Numeric minimum bin size as proportion (default: 0.05)
// //' @param smooth Numeric Laplace smoothing factor (default: 0)
// //' @param monotonic Logical enforce monotonic WoE (default: FALSE)
// //' @param min_iv Numeric minimum IV threshold (default: 0.01)
// //' @param digits Integer decimal places for labels (default: 4)
// //' @param miss_policy String missing value policy (default: "separate")
// //' @param special_vals NumericVector special values to isolate (default: empty)
// //' @param max_cat Integer max categories before grouping (default: 20)
// //' @param rare_pct Numeric rare category threshold (default: 0.01)
// //' @param cat_sep String category separator (default: "%;%")
// //' @param method String binning algorithm (default: "chimerge_mod")
// //' @param parallel Logical use parallel processing (default: FALSE)
// //' @param cache Logical enable chi-square caching (default: TRUE)
// //' @param weights Optional observation weights (default: NULL)
// //' @param verbose Logical print detailed logs (default: FALSE)
// //'
// //' @return Named list with binning results for each feature
// //'
// //' @export
// // [[Rcpp::export]]
// List chimerge_woe(
//    DataFrame data,
//    std::string target_col,
//    CharacterVector feature_cols,
//    int min_bins = 2,
//    int max_bins = 5,
//    double sig_level = 0.05,
//    double min_size = 0.05,
//    double smooth = 0,
//    bool monotonic = false,
//    double min_iv = 0.01,
//    int digits = 4,
//    std::string miss_policy = "separate",
//    NumericVector special_vals = NumericVector::create(),
//    int max_cat = 20,
//    double rare_pct = 0.01,
//    std::string cat_sep = "%;%",
//    std::string method = "chimerge_mod",
//    bool parallel = false,
//    bool cache = true,
//    Nullable<NumericVector> weights = R_NilValue,
//    bool verbose = false) {
//  
//  try {
//    // Initialize logger
//    chimerge_logging::global_logger.set_verbose(verbose);
//    
//    CHIMERGE_LOG_INFO("Starting ChiMerge WoE processing");
//    CHIMERGE_TIMER("Total processing");
//    
//    // Validate parameters
//    auto param_validation = InputValidator::validate_binning_parameters(
//      min_bins, max_bins, sig_level, min_size, smooth, monotonic, min_iv, digits,
//      miss_policy, max_cat, rare_pct, method, parallel, cache
//    );
//    
//    if (!param_validation.is_valid) {
//      std::string error_msg = "Parameter validation failed:\n";
//      for (const auto& err : param_validation.errors) {
//        error_msg += "  - " + err + "\n";
//      }
//      CHIMERGE_LOG_ERROR(error_msg);
//      Rcpp::stop(error_msg);
//    }
//    
//    // Extract target
//    LogicalVector target;
//    SEXP target_sexp = data[target_col];
//    
//    if (Rf_isLogical(target_sexp)) {
//      target = as<LogicalVector>(target_sexp);
//    } else if (Rf_isInteger(target_sexp)) {
//      IntegerVector iv = as<IntegerVector>(target_sexp);
//      target = LogicalVector(iv.size());
//      for (int i = 0; i < iv.size(); ++i) {
//        target[i] = (iv[i] == 1);
//      }
//    } else if (Rf_isNumeric(target_sexp)) {
//      NumericVector nv = as<NumericVector>(target_sexp);
//      target = LogicalVector(nv.size());
//      for (int i = 0; i < nv.size(); ++i) {
//        target[i] = (nv[i] == 1.0);
//      }
//    }
//    
//    // Extract weights
//    NumericVector weight_vec;
//    if (weights.isNotNull()) {
//      weight_vec = weights.get();
//    } else {
//      weight_vec = NumericVector(data.nrows(), 1.0);
//    }
//    
//    // Convert special values
//    std::vector<double> special_vals_vec;
//    for (int i = 0; i < special_vals.size(); ++i) {
//      special_vals_vec.push_back(special_vals[i]);
//    }
//    
//    // Process features
//    List results;
//    OptimalBinningOrchestrator orchestrator;
//    std::vector<std::string> features = Rcpp::as<std::vector<std::string>>(feature_cols);
//    
//    CHIMERGE_LOG_INFO("Processing ", features.size(), " features");
//    
//    for (const std::string& feature : features) {
//      try {
//        CHIMERGE_LOG_INFO("Processing feature: ", feature);
//        
//        if (!data.containsElementNamed(feature.c_str())) {
//          Rcpp::warning("Feature '" + feature + "' not found in data");
//          CHIMERGE_LOG_WARNING("Feature '", feature, "' not found in data");
//          continue;
//        }
//        
//        SEXP feature_sexp = data[feature];
//        
//        // Check variable type
//        VariableType var_type = detect_r_variable_type(feature_sexp);
//        
//        if (var_type == VariableType::DATE_TIME) {
//          Rcpp::warning("Feature '" + feature + "' is a date/time type, skipping");
//          CHIMERGE_LOG_WARNING("Feature '", feature, "' is a date/time type, skipping");
//          continue;
//        }
//        
//        if (var_type == VariableType::UNSUPPORTED) {
//          Rcpp::warning("Feature '" + feature + "' has unsupported type, skipping");
//          CHIMERGE_LOG_WARNING("Feature '", feature, "' has unsupported type, skipping");
//          continue;
//        }
//        
//        List feature_result;
//        
//        // Process based on type
//        if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN ||
//            Rf_isFactor(feature_sexp) || Rf_isString(feature_sexp)) {
//          
//          CHIMERGE_LOG_DEBUG("Processing as categorical feature");
//          feature_result = orchestrator.process_categorical_feature(
//            feature_sexp, feature, target, weight_vec,
//            min_bins, max_bins, min_size,
//            smooth, monotonic, min_iv,
//            miss_policy,
//            max_cat, rare_pct, cat_sep
//          );
//        } else {
//          CHIMERGE_LOG_DEBUG("Processing as numeric feature");
//          feature_result = orchestrator.process_numeric_feature(
//            feature_sexp, feature, target, weight_vec,
//            min_bins, max_bins, sig_level, min_size,
//            smooth, monotonic, min_iv, digits,
//            miss_policy, special_vals_vec,
//            method, cache
//          );
//        }
//        
//        // Check IV threshold
//        double iv = as<double>(feature_result["total_iv"]);
//        if (iv < min_iv) {
//          feature_result["messages"] = "IV below threshold (" + 
//            std::to_string(iv) + ")";
//          CHIMERGE_LOG_WARNING("Feature '", feature, "' has IV below threshold: ", iv);
//        }
//        
//        results[feature] = feature_result;
//        CHIMERGE_LOG_INFO("Completed processing feature: ", feature);
//        
//      } catch (const std::exception& e) {
//        Rcpp::warning("Error processing feature '" + feature + "': " + e.what());
//        CHIMERGE_LOG_ERROR("Error processing feature '", feature, "': ", e.what());
//      }
//    }
//    
//    CHIMERGE_LOG_INFO("ChiMerge WoE processing completed");
//    return results;
//    
//  } catch (const std::exception& e) {
//    CHIMERGE_LOG_ERROR("Error in chimerge_woe: ", e.what());
//    Rcpp::stop(std::string("Error in chimerge_woe: ") + e.what());
//  }
// }

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 // // chimerge_woe.cpp
 // #include <Rcpp.h>
 // #include <vector>
 // #include <string>
 // #include <map>
 // #include <unordered_map>
 // #include <algorithm>
 // #include <numeric>
 // #include <cmath>
 // #include <iostream>
 // #include <sstream>
 // #include <iomanip>
 // #include <set>
 // #include <queue>
 // #include <chrono>
 // #include <exception>
 // #include <memory>
 // #include <limits>
 // #include <functional>
 // 
 // #ifdef _OPENMP
 // #include <omp.h>
 // #endif
 // 
 // using namespace Rcpp;
 // using namespace std;
 // 
 // // =====================================================================
 // // SECTION 0: LOGGING SYSTEM
 // // =====================================================================
 // 
 // namespace chimerge_logging {
 // 
 // // Níveis de log
 // enum class LogLevel {
 //   NONE = 0,
 //     ERROR = 1,
 //     WARNING = 2,
 //     INFO = 3,
 //     DEBUG = 4
 // };
 // 
 // class Logger {
 // private:
 //   bool verbose_;
 //   LogLevel log_level_;
 //   
 // public:
 //   Logger(bool verbose = false, LogLevel level = LogLevel::INFO)
 //     : verbose_(verbose), log_level_(level) {}
 //   
 //   // Métodos para diferentes níveis de log
 //   template<typename... Args>
 //   void error(Args&&... args) {
 //     if (verbose_ && log_level_ >= LogLevel::ERROR) {
 //       log("ERROR", std::forward<Args>(args)...);
 //     }
 //   }
 //   
 //   template<typename... Args>
 //   void warning(Args&&... args) {
 //     if (verbose_ && log_level_ >= LogLevel::WARNING) {
 //       log("WARNING", std::forward<Args>(args)...);
 //     }
 //   }
 //   
 //   template<typename... Args>
 //   void info(Args&&... args) {
 //     if (verbose_ && log_level_ >= LogLevel::INFO) {
 //       log("INFO", std::forward<Args>(args)...);
 //     }
 //   }
 //   
 //   template<typename... Args>
 //   void debug(Args&&... args) {
 //     if (verbose_ && log_level_ >= LogLevel::DEBUG) {
 //       log("DEBUG", std::forward<Args>(args)...);
 //     }
 //   }
 //   
 //   // Método genérico de log
 //   template<typename... Args>
 //   void log(const std::string& level, Args&&... args) {
 //     if (!verbose_) return;
 //     
 //     Rcout << "[" << level << "] ";
 //     log_impl(std::forward<Args>(args)...);
 //     Rcout << std::endl;
 //   }
 //   
 //   // Métodos para medir tempo de execução
 //   class Timer {
 //   private:
 //     std::chrono::high_resolution_clock::time_point start_time_;
 //     std::string description_;
 //     Logger* logger_;
 //     bool verbose_;
 //     
 //   public:
 //     Timer(Logger* logger, const std::string& description, bool verbose = true)
 //       : logger_(logger), description_(description), verbose_(verbose && logger->is_verbose()) {
 //       if (verbose_) {
 //         start_time_ = std::chrono::high_resolution_clock::now();
 //         logger_->debug("Starting: ", description_);
 //       }
 //     }
 //     
 //     ~Timer() {
 //       if (verbose_) {
 //         auto end_time = std::chrono::high_resolution_clock::now();
 //         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
 //         logger_->debug("Completed: ", description_, " (", duration.count(), " ms)");
 //       }
 //     }
 //   };
 //   
 //   bool is_verbose() const { return verbose_; }
 //   LogLevel get_level() const { return log_level_; }
 //   
 //   void set_verbose(bool verbose) { verbose_ = verbose; }
 //   void set_level(LogLevel level) { log_level_ = level; }
 //   
 // private:
 //   // Implementação recursiva para concatenar argumentos
 //   void log_impl() {}
 //   
 //   template<typename T, typename... Args>
 //   void log_impl(T&& t, Args&&... args) {
 //     Rcout << t;
 //     log_impl(std::forward<Args>(args)...);
 //   }
 // };
 // 
 // // Instância global do logger
 // static Logger global_logger;
 // 
 // } // namespace chimerge_logging
 // 
 // // Macros para facilitar o uso do logger
 // #define CHIMERGE_LOG_TRACE(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
 // #define CHIMERGE_LOG_ERROR(...) chimerge_logging::global_logger.error(__VA_ARGS__)
 // #define CHIMERGE_LOG_WARNING(...) chimerge_logging::global_logger.warning(__VA_ARGS__)
 // #define CHIMERGE_LOG_INFO(...) chimerge_logging::global_logger.info(__VA_ARGS__)
 // #define CHIMERGE_LOG_DEBUG(...) chimerge_logging::global_logger.debug(__VA_ARGS__)
 // #define CHIMERGE_TIMER(description) chimerge_logging::Logger::Timer timer(&chimerge_logging::global_logger, description)
 // 
 // // =====================================================================
 // // SECTION 1: ENUMS AND CORE DATA STRUCTURES
 // // =====================================================================
 // 
 // enum class VariableType {
 //   NUMERIC_CONTINUOUS,
 //   NUMERIC_DISCRETE,
 //   CATEGORICAL,
 //   BOOLEAN,
 //   DATE_TIME,  // Para identificar e pular
 //   UNSUPPORTED
 // };
 // 
 // enum class MissingPolicy {
 //   SEPARATE,
 //   REMOVE,
 //   IMPUTE,
 //   MERGE
 // };
 // 
 // enum class BinningMethod {
 //   CHIMERGE,
 //   CHIMERGE_MOD,
 //   CHIMERGE_OPT,
 //   D2,
 //   HYBRID
 // };
 // 
 // struct DataPoint {
 //   double numeric_value;
 //   std::string categorical_value;
 //   bool target;
 //   bool is_missing;
 //   bool is_special;
 //   size_t original_index;
 //   double weight;
 //   
 //   DataPoint() : numeric_value(0.0), target(false), is_missing(false),
 //   is_special(false), original_index(0), weight(1.0) {}
 // };
 // 
 // struct NumericBin {
 //   double lower_bound;
 //   double upper_bound;
 //   size_t count_0;
 //   size_t count_1;
 //   double woe;
 //   double iv_contribution;
 //   std::string label;
 //   double weight_sum;
 //   
 //   size_t total() const { return count_0 + count_1; }
 //   double event_rate() const { return total() > 0 ? (double)count_1 / total() : 0.0; }
 // };
 // 
 // struct CategoricalBin {
 //   std::vector<std::string> categories;
 //   size_t count_0;
 //   size_t count_1;
 //   double woe;
 //   double iv_contribution;
 //   std::string label;
 //   double weight_sum;
 //   
 //   size_t total() const { return count_0 + count_1; }
 //   double event_rate() const { return total() > 0 ? (double)count_1 / total() : 0.0; }
 // };
 // 
 // struct SpecialBin {
 //   std::string type;
 //   std::vector<std::string> values;
 //   size_t count_0;
 //   size_t count_1;
 //   double woe;
 //   double iv_contribution;
 //   std::string label;
 //   double weight_sum;
 //   
 //   size_t total() const { return count_0 + count_1; }
 //   double event_rate() const { return total() > 0 ? (double)count_1 / total() : 0.0; }
 // };
 // 
 // struct CategoryStats {
 //   std::string category;
 //   size_t count_0;
 //   size_t count_1;
 //   double event_rate;
 //   double weight_sum;
 //   
 //   size_t total() const { return count_0 + count_1; }
 // };
 // 
 // struct ValidationResult {
 //   bool is_valid;
 //   std::vector<std::string> errors;
 //   std::vector<std::string> warnings;
 //   std::vector<std::string> messages;
 //   
 //   void add_error(const std::string& msg) {
 //     errors.push_back(msg);
 //     is_valid = false;
 //   }
 //   
 //   void add_warning(const std::string& msg) {
 //     warnings.push_back(msg);
 //   }
 //   
 //   void add_message(const std::string& msg) {
 //     messages.push_back(msg);
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 2: EXCEPTION HANDLING
 // // =====================================================================
 // 
 // class BinningException : public std::exception {
 // private:
 //   std::string message;
 // public:
 //   BinningException(const std::string& msg) : message(msg) {}
 //   const char* what() const noexcept override { return message.c_str(); }
 // };
 // 
 // // =====================================================================
 // // SECTION 3: TYPE DETECTION
 // // =====================================================================
 // 
 // VariableType detect_r_variable_type(SEXP x) {
 //   CHIMERGE_LOG_TRACE("Detecting variable type");
 //   
 //   // Check for Date/DateTime types first to exclude them
 //   if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") ||
 //       Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
 //     CHIMERGE_LOG_DEBUG("Detected date/time variable");
 //     return VariableType::DATE_TIME;
 //   }
 //   
 //   // Check for factor
 //   if (Rf_isFactor(x)) {
 //     SEXP levels = Rf_getAttrib(x, R_LevelsSymbol);
 //     int n_levels = Rf_length(levels);
 //     CHIMERGE_LOG_DEBUG("Detected factor with ", n_levels, " levels");
 //     if (n_levels <= 2) {
 //       return VariableType::BOOLEAN;
 //     }
 //     return VariableType::CATEGORICAL;
 //   }
 //   
 //   // Check for logical (boolean)
 //   if (Rf_isLogical(x)) {
 //     CHIMERGE_LOG_DEBUG("Detected logical variable");
 //     return VariableType::BOOLEAN;
 //   }
 //   
 //   // Check for character
 //   if (Rf_isString(x)) {
 //     // Count unique values to determine if boolean
 //     CharacterVector cv = as<CharacterVector>(x);
 //     std::set<std::string> unique_vals;
 //     for (int i = 0; i < cv.size(); ++i) {
 //       if (!CharacterVector::is_na(cv[i])) {
 //         unique_vals.insert(as<std::string>(cv[i]));
 //       }
 //     }
 //     CHIMERGE_LOG_DEBUG("Character variable with ", unique_vals.size(), " unique values");
 //     if (unique_vals.size() <= 2) {
 //       return VariableType::BOOLEAN;
 //     }
 //     return VariableType::CATEGORICAL;
 //   }
 //   
 //   // Check for numeric types
 //   if (Rf_isInteger(x)) {
 //     // Check if it's integer64
 //     if (Rf_inherits(x, "integer64")) {
 //       CHIMERGE_LOG_DEBUG("Detected integer64 variable");
 //       return VariableType::NUMERIC_CONTINUOUS;
 //     }
 //     
 //     // Regular integer - check unique values
 //     IntegerVector iv = as<IntegerVector>(x);
 //     std::set<int> unique_vals;
 //     for (int i = 0; i < iv.size(); ++i) {
 //       if (!IntegerVector::is_na(iv[i])) {
 //         unique_vals.insert(iv[i]);
 //         if (unique_vals.size() > 20) break;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Integer variable with ", unique_vals.size(), " unique values");
 //     
 //     if (unique_vals.size() <= 2) {
 //       bool is_binary = true;
 //       for (int val : unique_vals) {
 //         if (val != 0 && val != 1) {
 //           is_binary = false;
 //           break;
 //         }
 //       }
 //       if (is_binary) return VariableType::BOOLEAN;
 //     }
 //     
 //     return unique_vals.size() <= 20 ? VariableType::NUMERIC_DISCRETE :
 //       VariableType::NUMERIC_CONTINUOUS;
 //   }
 //   
 //   if (Rf_isNumeric(x)) {
 //     NumericVector nv = as<NumericVector>(x);
 //     std::set<double> unique_vals;
 //     bool all_integers = true;
 //     
 //     for (int i = 0; i < nv.size(); ++i) {
 //       if (!NumericVector::is_na(nv[i])) {
 //         unique_vals.insert(nv[i]);
 //         if (nv[i] != std::floor(nv[i])) {
 //           all_integers = false;
 //         }
 //         if (unique_vals.size() > 20 && !all_integers) break;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Numeric variable with ", unique_vals.size(), " unique values, all_integers: ", all_integers);
 //     
 //     if (all_integers && unique_vals.size() <= 2) {
 //       bool is_binary = true;
 //       for (double val : unique_vals) {
 //         if (val != 0.0 && val != 1.0) {
 //           is_binary = false;
 //           break;
 //         }
 //       }
 //       if (is_binary) return VariableType::BOOLEAN;
 //     }
 //     
 //     if (all_integers && unique_vals.size() <= 20) {
 //       return VariableType::NUMERIC_DISCRETE;
 //     }
 //     
 //     return VariableType::NUMERIC_CONTINUOUS;
 //   }
 //   
 //   CHIMERGE_LOG_WARNING("Unsupported variable type detected");
 //   return VariableType::UNSUPPORTED;
 // }
 // 
 // // =====================================================================
 // // SECTION 4: INPUT VALIDATION
 // // =====================================================================
 // 
 // class InputValidator {
 // public:
 //   static ValidationResult validate_binning_parameters(
 //       int min_bins, int max_bins, double sig_level, double min_size,
 //       double smooth, bool monotonic, double min_iv, int digits,
 //       const std::string& miss_policy, int max_cat, double rare_pct,
 //       const std::string& method, bool parallel, bool cache) {
 //     
 //     CHIMERGE_LOG_INFO("Validating binning parameters");
 //     
 //     ValidationResult result;
 //     result.is_valid = true;
 //     
 //     // Validate bin counts
 //     if (min_bins < 2) {
 //       result.add_error("min_bins must be at least 2, got " + std::to_string(min_bins));
 //     }
 //     if (max_bins < min_bins) {
 //       result.add_error("max_bins must be >= min_bins");
 //     }
 //     if (max_bins > 100) {
 //       result.add_warning("max_bins > 100 may lead to overfitting");
 //     }
 //     
 //     // Validate digits
 //     if (digits < 0 || digits > 10) {
 //       result.add_error("digits must be between 0 and 10");
 //     }
 //     
 //     // Validate significance level
 //     if (sig_level <= 0.0 || sig_level >= 1.0) {
 //       result.add_error("sig_level must be between 0 and 1");
 //     }
 //     
 //     // Validate minimum bin size
 //     if (min_size <= 0.0 || min_size >= 1.0) {
 //       result.add_error("min_size must be between 0 and 1");
 //     }
 //     
 //     // Validate smoothing
 //     if (smooth < 0.0) {
 //       result.add_error("smooth must be non-negative");
 //     }
 //     
 //     // Validate minimum IV
 //     if (min_iv < 0.0) {
 //       result.add_error("min_iv must be non-negative");
 //     }
 //     
 //     // Validate missing policy
 //     std::set<std::string> valid_policies = {"separate", "remove", "impute", "merge"};
 //     if (valid_policies.find(miss_policy) == valid_policies.end()) {
 //       result.add_error("miss_policy must be one of: separate, remove, impute, merge");
 //     }
 //     
 //     // Validate max categories
 //     if (max_cat < 2) {
 //       result.add_error("max_cat must be at least 2");
 //     }
 //     
 //     // Validate rare percentage
 //     if (rare_pct < 0.0 || rare_pct >= 0.5) {
 //       result.add_error("rare_pct must be between 0 and 0.5");
 //     }
 //     
 //     // Validate method
 //     std::set<std::string> valid_methods = {"chimerge", "chimerge_mod", "chimerge_opt", "d2", "hybrid"};
 //                                            if (valid_methods.find(method) == valid_methods.end()) {
 //                                              result.add_error("invalid method: " + method);
 //                                            }
 //                                            
 //                                            CHIMERGE_LOG_DEBUG("Validation completed - valid: ", result.is_valid);
 //                                            return result;
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 5: CHI-SQUARE CACHE
 // // =====================================================================
 // 
 // struct PairHash {
 //   size_t operator()(const std::pair<size_t, size_t>& p) const {
 //     return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 1);
 //   }
 // };
 // 
 // class ChiSquareCache {
 // private:
 //   std::unordered_map<std::pair<size_t, size_t>, double, PairHash> cache;
 //   std::vector<bool> valid_flags;
 //   size_t max_cache_size;
 //   bool enabled;
 //   
 // public:
 //   ChiSquareCache(size_t max_size = 100000, bool use_cache = true)
 //     : max_cache_size(max_size), enabled(use_cache) {}
 //   
 //   void initialize(size_t size) {
 //     if (!enabled) return;
 //     CHIMERGE_LOG_DEBUG("Initializing chi-square cache with size ", size);
 //     cache.clear();
 //     valid_flags.assign(size, true);
 //   }
 //   
 //   double get_cached_value(size_t i, size_t j) {
 //     if (!enabled) return -1.0;
 //     auto key = std::make_pair(i, j);
 //     auto it = cache.find(key);
 //     return (it != cache.end()) ? it->second : -1.0;
 //   }
 //   
 //   void invalidate_range(size_t start, size_t end) {
 //     if (!enabled) return;
 //     CHIMERGE_LOG_TRACE("Invalidating cache range [", start, ", ", end, "]");
 //     for (auto it = cache.begin(); it != cache.end();) {
 //       if (it->first.first >= start || it->first.second >= start) {
 //         it = cache.erase(it);
 //       } else {
 //         ++it;
 //       }
 //     }
 //   }
 //   
 //   void update_cache(size_t i, size_t j, double value) {
 //     if (!enabled) return;
 //     if (cache.size() < max_cache_size) {
 //       cache[std::make_pair(i, j)] = value;
 //     }
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 6: STATISTICAL FUNCTIONS
 // // =====================================================================
 // 
 // double get_chi_square_threshold(double significance_level, int df) {
 //   CHIMERGE_LOG_TRACE("Getting chi-square threshold for sig_level: ", significance_level, ", df: ", df);
 //   
 //   if (df != 1) {
 //     throw BinningException("Only df=1 is supported for chi-square test");
 //   }
 //   
 //   if (significance_level >= 0.25) return 1.323;
 //   else if (significance_level >= 0.20) return 1.642;
 //   else if (significance_level >= 0.15) return 2.072;
 //   else if (significance_level >= 0.10) return 2.706;
 //   else if (significance_level >= 0.05) return 3.841;
 //   else if (significance_level >= 0.025) return 5.024;
 //   else if (significance_level >= 0.01) return 6.635;
 //   else if (significance_level >= 0.005) return 7.879;
 //   else if (significance_level >= 0.001) return 10.828;
 //   else return 15.138;
 // }
 // 
 // double calculate_chi_square(const NumericBin& bin1, const NumericBin& bin2) {
 //   double total = bin1.total() + bin2.total();
 //   if (total == 0) return 0.0;
 //   
 //   double total_0 = bin1.count_0 + bin2.count_0;
 //   double total_1 = bin1.count_1 + bin2.count_1;
 //   
 //   if (total_0 == 0 || total_1 == 0) return 0.0;
 //   
 //   double chi_square = 0.0;
 //   
 //   std::vector<std::vector<double>> observed = {{(double)bin1.count_0, (double)bin1.count_1},
 //   {(double)bin2.count_0, (double)bin2.count_1}};
 //   
 //   for (int i = 0; i < 2; ++i) {
 //     for (int j = 0; j < 2; ++j) {
 //       double row_total = observed[i][0] + observed[i][1];
 //       double col_total = (j == 0) ? total_0 : total_1;
 //       double expected = (row_total * col_total) / total;
 //       
 //       if (expected > 0) {
 //         chi_square += std::pow(observed[i][j] - expected, 2) / expected;
 //       }
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_TRACE("Chi-square calculated: ", chi_square);
 //   return chi_square;
 // }
 // 
 // double calculate_chi_square_modified(const NumericBin& bin1, const NumericBin& bin2) {
 //   double total = bin1.total() + bin2.total();
 //   if (total == 0) return 0.0;
 //   
 //   double total_0 = bin1.count_0 + bin2.count_0;
 //   double total_1 = bin1.count_1 + bin2.count_1;
 //   
 //   if (total_0 == 0 || total_1 == 0) return 0.0;
 //   
 //   double chi_square = 0.0;
 //   
 //   std::vector<std::vector<double>> observed = {{(double)bin1.count_0, (double)bin1.count_1},
 //   {(double)bin2.count_0, (double)bin2.count_1}};
 //   
 //   for (int i = 0; i < 2; ++i) {
 //     for (int j = 0; j < 2; ++j) {
 //       double row_total = observed[i][0] + observed[i][1];
 //       double col_total = (j == 0) ? total_0 : total_1;
 //       double expected = (row_total * col_total) / total;
 //       
 //       double diff = std::abs(observed[i][j] - expected) - 0.5;
 //       if (diff < 0) diff = 0;
 //       
 //       double denominator = std::max(expected, 0.5);
 //       chi_square += (diff * diff) / denominator;
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_TRACE("Modified chi-square calculated: ", chi_square);
 //   return chi_square;
 // }
 // 
 // double calculate_correlation(const std::vector<double>& x, const std::vector<double>& y) {
 //   if (x.size() != y.size() || x.empty()) return 0.0;
 //   
 //   double n = x.size();
 //   double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
 //   double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
 //   double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
 //   
 //   for (size_t i = 0; i < x.size(); ++i) {
 //     sum_xy += x[i] * y[i];
 //     sum_x2 += x[i] * x[i];
 //     sum_y2 += y[i] * y[i];
 //   }
 //   
 //   double numerator = n * sum_xy - sum_x * sum_y;
 //   double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
 //   
 //   double correlation = (denominator != 0) ? numerator / denominator : 0.0;
 //   CHIMERGE_LOG_TRACE("Correlation calculated: ", correlation);
 //   return correlation;
 // }
 // 
 // // =====================================================================
 // // SECTION 7: BIN LABEL FORMATTING
 // // =====================================================================
 // 
 // std::string format_numeric_bin_label(const NumericBin& bin, bool is_first, bool is_last, int digits) {
 //   std::ostringstream oss;
 //   oss << std::fixed << std::setprecision(digits);
 //   
 //   if (is_first && bin.lower_bound == -std::numeric_limits<double>::infinity()) {
 //     oss << "(-inf," << bin.upper_bound << ")";
 //   } else if (is_last && bin.upper_bound == std::numeric_limits<double>::infinity()) {
 //     oss << "[" << bin.lower_bound << ",+inf)";
 //   } else {
 //     oss << "[" << bin.lower_bound << "," << bin.upper_bound << ")";
 //   }
 //   
 //   std::string label = oss.str();
 //   CHIMERGE_LOG_TRACE("Formatted numeric bin label: ", label);
 //   return label;
 // }
 // 
 // std::string format_categorical_bin_label(const CategoricalBin& bin, const std::string& separator) {
 //   if (bin.categories.size() == 1) {
 //     return bin.categories[0];
 //   }
 //   
 //   std::ostringstream oss;
 //   for (size_t i = 0; i < bin.categories.size(); ++i) {
 //     if (i > 0) oss << separator;
 //     oss << bin.categories[i];
 //   }
 //   
 //   std::string label = oss.str();
 //   CHIMERGE_LOG_TRACE("Formatted categorical bin label: ", label);
 //   return label;
 // }
 // 
 // // =====================================================================
 // // SECTION 8: WOE CALCULATOR (CORRIGIDO)
 // // =====================================================================
 // 
 // class WoECalculator {
 // private:
 //   double total_positives;
 //   double total_negatives;
 //   double laplace_smoothing;
 //   double total_iv;
 //   bool enforce_monotonic;
 //   
 // public:
 //   WoECalculator(size_t pos_count, size_t neg_count, double smoothing, bool monotonic)
 //     : total_positives(pos_count), total_negatives(neg_count),
 //       laplace_smoothing(smoothing), total_iv(0.0), enforce_monotonic(monotonic) {}
 //   
 //   template<typename BinType>
 //   void calculate_woe_iv(std::vector<BinType>& bins) {
 //     CHIMERGE_LOG_INFO("Calculating WoE and IV for ", bins.size(), " bins");
 //     CHIMERGE_TIMER("WoE/IV calculation");
 //     
 //     total_iv = 0.0;
 //     size_t num_bins = bins.size();
 //     
 //     if (num_bins == 0) {
 //       CHIMERGE_LOG_WARNING("Empty bins vector, skipping WoE calculation");
 //       return;
 //     }
 //     
 //     // Calculate basic WoE values
 //     for (auto& bin : bins) {
 //       double smoothed_pos = bin.count_1 + laplace_smoothing;
 //       double smoothed_neg = bin.count_0 + laplace_smoothing;
 //       double smoothed_total_pos = total_positives + num_bins * laplace_smoothing;
 //       double smoothed_total_neg = total_negatives + num_bins * laplace_smoothing;
 //       
 //       if (smoothed_total_pos == 0 || smoothed_total_neg == 0) {
 //         bin.woe = 0;
 //         bin.iv_contribution = 0;
 //         CHIMERGE_LOG_DEBUG("Zero totals for bin, setting WoE=0");
 //         continue;
 //       }
 //       
 //       double dist_pos = smoothed_pos / smoothed_total_pos;
 //       double dist_neg = smoothed_neg / smoothed_total_neg;
 //       
 //       if (dist_pos == 0 || dist_neg == 0) {
 //         bin.woe = 0;
 //         bin.iv_contribution = 0;
 //         CHIMERGE_LOG_DEBUG("Zero distribution for bin, setting WoE=0");
 //         continue;
 //       }
 //       
 //       bin.woe = std::log(dist_pos / dist_neg);
 //       
 //       // Cap WoE values to prevent extreme values
 //       if (bin.woe > 5) bin.woe = 5;
 //       if (bin.woe < -5) bin.woe = -5;
 //       
 //       bin.iv_contribution = (dist_pos - dist_neg) * bin.woe;
 //       total_iv += bin.iv_contribution;
 //       
 //       CHIMERGE_LOG_DEBUG("Bin WoE: ", bin.woe, ", IV contribution: ", bin.iv_contribution);
 //     }
 //     
 //     // Only apply monotonicity if explicitly requested and we have enough bins
 //     if (enforce_monotonic && bins.size() > 2) {
 //       CHIMERGE_LOG_DEBUG("Enforcing monotonicity via PAVA");
 //       apply_monotonic_constraint(bins);
 //     }
 //     
 //     CHIMERGE_LOG_INFO("Total IV calculated: ", total_iv);
 //   }
 //   
 //   double get_total_iv() const { return total_iv; }
 //   
 // private:
 //   // Implementação correta do PAVA (Pool Adjacent Violators Algorithm)
 //   template<typename BinType>
 //   void apply_monotonic_constraint(std::vector<BinType>& bins) {
 //     size_t n = bins.size();
 //     if (n <= 2) return;
 //     
 //     // Determine the direction (increasing or decreasing)
 //     // Based on the correlation between bin position and event rate
 //     std::vector<double> positions(n);
 //     std::vector<double> event_rates(n);
 //     
 //     for (size_t i = 0; i < n; ++i) {
 //       positions[i] = i;
 //       event_rates[i] = bins[i].event_rate();
 //     }
 //     
 //     double correlation = calculate_correlation(positions, event_rates);
 //     bool increasing = (correlation >= 0);
 //     
 //     CHIMERGE_LOG_DEBUG("Applying PAVA with direction: ", (increasing ? "increasing" : "decreasing"));
 //     
 //     // Apply PAVA to event rates
 //     std::vector<double> weights(n);
 //     std::vector<double> values(n);
 //     
 //     for (size_t i = 0; i < n; ++i) {
 //       weights[i] = bins[i].total();
 //       values[i] = bins[i].event_rate();
 //     }
 //     
 //     // Apply PAVA algorithm
 //     if (increasing) {
 //       apply_pava_increasing(values, weights);
 //     } else {
 //       apply_pava_decreasing(values, weights);
 //     }
 //     
 //     // Recalculate WoE based on monotonic event rates
 //     total_iv = 0.0;
 //     for (size_t i = 0; i < n; ++i) {
 //       double event_rate = values[i];
 //       
 //       // Calculate new count_1 and count_0 based on monotonic event rate
 //       double total = bins[i].total();
 //       bins[i].count_1 = std::round(event_rate * total);
 //       bins[i].count_0 = total - bins[i].count_1;
 //       
 //       // Recalculate WoE
 //       double smoothed_pos = bins[i].count_1 + laplace_smoothing;
 //       double smoothed_neg = bins[i].count_0 + laplace_smoothing;
 //       double smoothed_total_pos = total_positives + n * laplace_smoothing;
 //       double smoothed_total_neg = total_negatives + n * laplace_smoothing;
 //       
 //       if (smoothed_total_pos > 0 && smoothed_total_neg > 0) {
 //         double dist_pos = smoothed_pos / smoothed_total_pos;
 //         double dist_neg = smoothed_neg / smoothed_total_neg;
 //         
 //         if (dist_pos > 0 && dist_neg > 0) {
 //           bins[i].woe = std::log(dist_pos / dist_neg);
 //           
 //           // Cap WoE values
 //           if (bins[i].woe > 5) bins[i].woe = 5;
 //           if (bins[i].woe < -5) bins[i].woe = -5;
 //           
 //           bins[i].iv_contribution = (dist_pos - dist_neg) * bins[i].woe;
 //           total_iv += bins[i].iv_contribution;
 //         }
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Monotonicity enforced, new total IV: ", total_iv);
 //   }
 //   
 //   // PAVA for increasing constraint
 //   void apply_pava_increasing(std::vector<double>& values, std::vector<double>& weights) {
 //     size_t n = values.size();
 //     
 //     bool changed = true;
 //     while (changed) {
 //       changed = false;
 //       
 //       for (size_t i = 0; i < n - 1; ++i) {
 //         if (values[i] > values[i + 1]) {
 //           // Pool adjacent violators
 //           double pooled_weight = weights[i] + weights[i + 1];
 //           double pooled_value = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / pooled_weight;
 //           
 //           values[i] = pooled_value;
 //           values[i + 1] = pooled_value;
 //           
 //           // Update weights for next iteration
 //           weights[i] = pooled_weight;
 //           weights[i + 1] = pooled_weight;
 //           
 //           changed = true;
 //         }
 //       }
 //     }
 //   }
 //   
 //   // PAVA for decreasing constraint
 //   void apply_pava_decreasing(std::vector<double>& values, std::vector<double>& weights) {
 //     size_t n = values.size();
 //     
 //     bool changed = true;
 //     while (changed) {
 //       changed = false;
 //       
 //       for (size_t i = 0; i < n - 1; ++i) {
 //         if (values[i] < values[i + 1]) {
 //           // Pool adjacent violators
 //           double pooled_weight = weights[i] + weights[i + 1];
 //           double pooled_value = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / pooled_weight;
 //           
 //           values[i] = pooled_value;
 //           values[i + 1] = pooled_value;
 //           
 //           // Update weights for next iteration
 //           weights[i] = pooled_weight;
 //           weights[i + 1] = pooled_weight;
 //           
 //           changed = true;
 //         }
 //       }
 //     }
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 9: CHIMERGE IMPLEMENTATIONS
 // // =====================================================================
 // 
 // class ChiMergeAlgorithm {
 // protected:
 //   int min_bins;
 //   int max_bins;
 //   double sig_level;
 //   double min_size;
 //   ChiSquareCache cache;
 //   size_t total_samples;
 //   
 //   void merge_numeric_bins(std::vector<NumericBin>& bins, size_t index) {
 //     if (index >= bins.size() - 1) return;
 //     
 //     CHIMERGE_LOG_DEBUG("Merging bins ", index, " and ", (index + 1));
 //     
 //     bins[index].upper_bound = bins[index + 1].upper_bound;
 //     bins[index].count_0 += bins[index + 1].count_0;
 //     bins[index].count_1 += bins[index + 1].count_1;
 //     bins[index].weight_sum += bins[index + 1].weight_sum;
 //     
 //     bins.erase(bins.begin() + index + 1);
 //     cache.invalidate_range(index, bins.size());
 //   }
 //   
 //   bool should_stop_merging(const std::vector<NumericBin>& bins, double min_chi) {
 //     if (bins.size() <= (size_t)min_bins) {
 //       CHIMERGE_LOG_DEBUG("Stopping merge: bins.size() <= min_bins");
 //       return true;
 //     }
 //     
 //     double threshold = get_chi_square_threshold(sig_level, 1);
 //     if (min_chi >= threshold) {
 //       CHIMERGE_LOG_DEBUG("Stopping merge: min_chi >= threshold (", min_chi, " >= ", threshold, ")");
 //       return true;
 //     }
 //     
 //     for (const auto& bin : bins) {
 //       if (bin.total() < min_size * total_samples) {
 //         CHIMERGE_LOG_DEBUG("Continuing merge: bin size < min_size threshold");
 //         return false;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Stopping merge: all bins meet min_size requirement");
 //     return false;
 //   }
 //   
 //   std::vector<NumericBin> create_initial_numeric_bins(const std::vector<DataPoint>& data) {
 //     CHIMERGE_LOG_INFO("Creating initial numeric bins from ", data.size(), " data points");
 //     CHIMERGE_TIMER("Initial bin creation");
 //     
 //     std::map<double, NumericBin> unique_bins;
 //     
 //     for (const auto& point : data) {
 //       if (point.is_missing || point.is_special) continue;
 //       
 //       auto& bin = unique_bins[point.numeric_value];
 //       bin.lower_bound = point.numeric_value;
 //       bin.upper_bound = point.numeric_value;
 //       
 //       if (point.target) {
 //         bin.count_1++;
 //       } else {
 //         bin.count_0++;
 //       }
 //       bin.weight_sum += point.weight;
 //     }
 //     
 //     std::vector<NumericBin> bins;
 //     for (const auto& pair : unique_bins) {
 //       bins.push_back(pair.second);
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Created ", bins.size(), " initial bins");
 //     
 //     for (size_t i = 0; i < bins.size(); ++i) {
 //       if (i == 0) {
 //         bins[i].lower_bound = -std::numeric_limits<double>::infinity();
 //       }
 //       if (i == bins.size() - 1) {
 //         bins[i].upper_bound = std::numeric_limits<double>::infinity();
 //       } else {
 //         bins[i].upper_bound = bins[i + 1].lower_bound;
 //       }
 //     }
 //     
 //     return bins;
 //   }
 //   
 // public:
 //   ChiMergeAlgorithm(int min_b, int max_b, double sig, double min_s, bool use_cache)
 //     : min_bins(min_b), max_bins(max_b), sig_level(sig), min_size(min_s),
 //       cache(100000, use_cache), total_samples(0) {}
 //   
 //   virtual std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) = 0;
 // };
 // 
 // class ChiMergeModified : public ChiMergeAlgorithm {
 // public:
 //   ChiMergeModified(int min_b, int max_b, double sig, double min_s, bool use_cache)
 //     : ChiMergeAlgorithm(min_b, max_b, sig, min_s, use_cache) {}
 //   
 //   std::vector<NumericBin> apply_numeric_binning(const std::vector<DataPoint>& data) override {
 //     CHIMERGE_LOG_INFO("Applying ChiMerge Modified algorithm");
 //     CHIMERGE_TIMER("ChiMerge Modified");
 //     
 //     auto bins = create_initial_numeric_bins(data);
 //     
 //     total_samples = 0;
 //     for (const auto& bin : bins) {
 //       total_samples += bin.total();
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Total samples: ", total_samples);
 //     
 //     if (bins.size() <= (size_t)min_bins) {
 //       CHIMERGE_LOG_DEBUG("Not enough bins for merging, returning initial bins");
 //       return bins;
 //     }
 //     
 //     cache.initialize(bins.size());
 //     
 //     // First pass: merge bins with zero counts
 //     CHIMERGE_LOG_DEBUG("First pass: merging bins with zero counts");
 //     for (size_t i = 0; i < bins.size() - 1;) {
 //       if (bins[i].count_0 == 0 || bins[i].count_1 == 0 ||
 //           bins[i+1].count_0 == 0 || bins[i+1].count_1 == 0) {
 //         CHIMERGE_LOG_DEBUG("Merging zero-count bins at index ", i);
 //         merge_numeric_bins(bins, i);
 //       } else {
 //         i++;
 //       }
 //     }
 //     
 //     // Main merging loop
 //     CHIMERGE_LOG_DEBUG("Starting main merging loop");
 //     while (bins.size() > (size_t)min_bins && bins.size() > 2) {
 //       size_t min_index = 0;
 //       double min_chi = std::numeric_limits<double>::max();
 //       
 //       // Find minimum chi-square pair
 //       for (size_t i = 0; i < bins.size() - 1; ++i) {
 //         double chi = cache.get_cached_value(i, i + 1);
 //         if (chi < 0) {
 //           chi = calculate_chi_square_modified(bins[i], bins[i + 1]);
 //           cache.update_cache(i, i + 1, chi);
 //         }
 //         
 //         if (chi < min_chi) {
 //           min_chi = chi;
 //           min_index = i;
 //         }
 //       }
 //       
 //       CHIMERGE_LOG_DEBUG("Min chi-square: ", min_chi, " at index ", min_index);
 //       
 //       if (should_stop_merging(bins, min_chi)) {
 //         CHIMERGE_LOG_DEBUG("Stopping criteria met, breaking merge loop");
 //         break;
 //       }
 //       
 //       merge_numeric_bins(bins, min_index);
 //       
 //       if (bins.size() <= (size_t)max_bins) {
 //         CHIMERGE_LOG_DEBUG("Reached max_bins limit, breaking merge loop");
 //         break;
 //       }
 //     }
 //     
 //     // Force reduction if still above max_bins
 //     while (bins.size() > (size_t)max_bins) {
 //       CHIMERGE_LOG_DEBUG("Forcing bin reduction, current bins: ", bins.size());
 //       size_t min_index = 0;
 //       double min_chi = std::numeric_limits<double>::max();
 //       
 //       for (size_t i = 0; i < bins.size() - 1; ++i) {
 //         double chi = calculate_chi_square_modified(bins[i], bins[i + 1]);
 //         if (chi < min_chi) {
 //           min_chi = chi;
 //           min_index = i;
 //         }
 //       }
 //       
 //       merge_numeric_bins(bins, min_index);
 //     }
 //     
 //     CHIMERGE_LOG_INFO("ChiMerge Modified completed with ", bins.size(), " bins");
 //     return bins;
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 10: CATEGORICAL BINNING
 // // =====================================================================
 // 
 // std::vector<CategoryStats> calculate_category_statistics(const std::vector<DataPoint>& data) {
 //   CHIMERGE_LOG_INFO("Calculating category statistics for ", data.size(), " data points");
 //   CHIMERGE_TIMER("Category statistics");
 //   
 //   std::map<std::string, CategoryStats> stats_map;
 //   
 //   for (const auto& point : data) {
 //     if (point.is_missing || point.is_special) continue;
 //     
 //     auto& stats = stats_map[point.categorical_value];
 //     stats.category = point.categorical_value;
 //     if (point.target) {
 //       stats.count_1++;
 //     } else {
 //       stats.count_0++;
 //     }
 //     stats.weight_sum += point.weight;
 //   }
 //   
 //   std::vector<CategoryStats> stats_vector;
 //   for (const auto& pair : stats_map) {
 //     CategoryStats stats = pair.second;
 //     stats.event_rate = stats.total() > 0 ? (double)stats.count_1 / stats.total() : 0.0;
 //     stats_vector.push_back(stats);
 //   }
 //   
 //   std::sort(stats_vector.begin(), stats_vector.end(),
 //             [](const CategoryStats& a, const CategoryStats& b) {
 //               return a.event_rate < b.event_rate;
 //             });
 //   
 //   CHIMERGE_LOG_DEBUG("Calculated statistics for ", stats_vector.size(), " categories");
 //   return stats_vector;
 // }
 // 
 // std::vector<CategoricalBin> apply_categorical_binning(
 //     const std::vector<DataPoint>& data,
 //     int min_bins, int max_bins, double min_size,
 //     int max_cat, double rare_pct, const std::string& cat_sep) {
 //   
 //   CHIMERGE_LOG_INFO("Applying categorical binning");
 //   CHIMERGE_TIMER("Categorical binning");
 //   
 //   auto category_stats = calculate_category_statistics(data);
 //   
 //   size_t total_count = 0;
 //   for (const auto& stats : category_stats) {
 //     total_count += stats.total();
 //   }
 //   
 //   CHIMERGE_LOG_DEBUG("Total count: ", total_count);
 //   
 //   std::vector<CategoricalBin> bins;
 //   CategoricalBin rare_bin;
 //   rare_bin.count_0 = 0;
 //   rare_bin.count_1 = 0;
 //   rare_bin.weight_sum = 0;
 //   
 //   for (const auto& stats : category_stats) {
 //     double frequency = (double)stats.total() / total_count;
 //     
 //     if (frequency < rare_pct && bins.size() + 1 > (size_t)max_cat) {
 //       rare_bin.categories.push_back(stats.category);
 //       rare_bin.count_0 += stats.count_0;
 //       rare_bin.count_1 += stats.count_1;
 //       rare_bin.weight_sum += stats.weight_sum;
 //       CHIMERGE_LOG_DEBUG("Adding category ", stats.category, " to rare bin");
 //     } else {
 //       CategoricalBin bin;
 //       bin.categories.push_back(stats.category);
 //       bin.count_0 = stats.count_0;
 //       bin.count_1 = stats.count_1;
 //       bin.weight_sum = stats.weight_sum;
 //       bins.push_back(bin);
 //       CHIMERGE_LOG_DEBUG("Creating bin for category ", stats.category);
 //     }
 //   }
 //   
 //   if (!rare_bin.categories.empty() && rare_bin.total() > 0) {
 //     bins.push_back(rare_bin);
 //     CHIMERGE_LOG_DEBUG("Added rare bin with ", rare_bin.categories.size(), " categories");
 //   }
 //   
 //   // Merge bins to meet max_bins constraint
 //   while (bins.size() > (size_t)max_bins) {
 //     double min_chi = std::numeric_limits<double>::max();
 //     size_t merge_index = 0;
 //     
 //     for (size_t i = 0; i < bins.size() - 1; ++i) {
 //       NumericBin temp1, temp2;
 //       temp1.count_0 = bins[i].count_0;
 //       temp1.count_1 = bins[i].count_1;
 //       temp2.count_0 = bins[i + 1].count_0;
 //       temp2.count_1 = bins[i + 1].count_1;
 //       
 //       double chi = calculate_chi_square(temp1, temp2);
 //       if (chi < min_chi) {
 //         min_chi = chi;
 //         merge_index = i;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Merging bins ", merge_index, " and ", (merge_index + 1), " with chi: ", min_chi);
 //     
 //     bins[merge_index].categories.insert(bins[merge_index].categories.end(),
 //                                         bins[merge_index + 1].categories.begin(),
 //                                         bins[merge_index + 1].categories.end());
 //     bins[merge_index].count_0 += bins[merge_index + 1].count_0;
 //     bins[merge_index].count_1 += bins[merge_index + 1].count_1;
 //     bins[merge_index].weight_sum += bins[merge_index + 1].weight_sum;
 //     bins.erase(bins.begin() + merge_index + 1);
 //   }
 //   
 //   // Format labels
 //   for (auto& bin : bins) {
 //     bin.label = format_categorical_bin_label(bin, cat_sep);
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Categorical binning completed with ", bins.size(), " bins");
 //   return bins;
 // }
 // 
 // // =====================================================================
 // // SECTION 11: SPECIAL VALUE HANDLING
 // // =====================================================================
 // 
 // class SpecialValueHandler {
 // private:
 //   MissingPolicy miss_policy;
 //   std::vector<double> special_numeric;
 //   std::vector<std::string> special_string;
 //   
 // public:
 //   SpecialValueHandler(const std::string& policy,
 //                       const std::vector<double>& special_num,
 //                       const std::vector<std::string>& special_str) {
 //     if (policy == "separate") miss_policy = MissingPolicy::SEPARATE;
 //     else if (policy == "remove") miss_policy = MissingPolicy::REMOVE;
 //     else if (policy == "impute") miss_policy = MissingPolicy::IMPUTE;
 //     else miss_policy = MissingPolicy::MERGE;
 //     
 //     special_numeric = special_num;
 //     special_string = special_str;
 //     
 //     CHIMERGE_LOG_DEBUG("SpecialValueHandler initialized with policy: ", policy);
 //   }
 //   
 //   std::vector<SpecialBin> process_special_values(const std::vector<DataPoint>& data) {
 //     CHIMERGE_LOG_INFO("Processing special values");
 //     CHIMERGE_TIMER("Special value processing");
 //     
 //     std::vector<SpecialBin> special_bins;
 //     
 //     if (miss_policy != MissingPolicy::SEPARATE) {
 //       CHIMERGE_LOG_DEBUG("Missing policy not 'separate', skipping special bin creation");
 //       return special_bins;
 //     }
 //     
 //     // Handle missing values
 //     SpecialBin missing_bin;
 //     missing_bin.type = "missing";
 //     missing_bin.label = "[Missing]";
 //     missing_bin.count_0 = 0;
 //     missing_bin.count_1 = 0;
 //     missing_bin.weight_sum = 0;
 //     
 //     for (const auto& point : data) {
 //       if (point.is_missing) {
 //         if (point.target) {
 //           missing_bin.count_1++;
 //         } else {
 //           missing_bin.count_0++;
 //         }
 //         missing_bin.weight_sum += point.weight;
 //       }
 //     }
 //     
 //     if (missing_bin.total() > 0) {
 //       special_bins.push_back(missing_bin);
 //       CHIMERGE_LOG_DEBUG("Created missing bin with ", missing_bin.total(), " samples");
 //     }
 //     
 //     // Handle special numeric values
 //     for (double special_val : special_numeric) {
 //       SpecialBin special_bin;
 //       special_bin.type = "special_numeric";
 //       special_bin.label = "[Special:" + std::to_string(static_cast<int>(special_val)) + "]";
 //       special_bin.count_0 = 0;
 //       special_bin.count_1 = 0;
 //       special_bin.weight_sum = 0;
 //       
 //       for (const auto& point : data) {
 //         if (point.is_special && std::abs(point.numeric_value - special_val) < 1e-9) {
 //           if (point.target) {
 //             special_bin.count_1++;
 //           } else {
 //             special_bin.count_0++;
 //           }
 //           special_bin.weight_sum += point.weight;
 //         }
 //       }
 //       
 //       if (special_bin.total() > 0) {
 //         special_bins.push_back(special_bin);
 //         CHIMERGE_LOG_DEBUG("Created special bin for value ", special_val, " with ", special_bin.total(), " samples");
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_INFO("Processed ", special_bins.size(), " special bins");
 //     return special_bins;
 //   }
 // };
 // 
 // // =====================================================================
 // // SECTION 12: DATA PREPARATION AND EXTRACTION
 // // =====================================================================
 // 
 // std::vector<DataPoint> prepare_numeric_data(SEXP x, const LogicalVector& target,
 //                                             const NumericVector& weights,
 //                                             const std::vector<double>& special_vals) {
 //   CHIMERGE_LOG_INFO("Preparing numeric data");
 //   CHIMERGE_TIMER("Numeric data preparation");
 //   
 //   std::vector<DataPoint> data;
 //   bool has_weights = weights.size() > 0;
 //   
 //   if (Rf_isInteger(x)) {
 //     IntegerVector iv = as<IntegerVector>(x);
 //     CHIMERGE_LOG_DEBUG("Processing integer vector with ", iv.size(), " elements");
 //     for (int i = 0; i < iv.size(); ++i) {
 //       DataPoint point;
 //       point.target = target[i];
 //       point.original_index = i;
 //       point.weight = has_weights ? weights[i] : 1.0;
 //       
 //       if (IntegerVector::is_na(iv[i])) {
 //         point.is_missing = true;
 //         CHIMERGE_LOG_TRACE("Missing value at index ", i);
 //       } else {
 //         point.numeric_value = iv[i];
 //         for (double special : special_vals) {
 //           if (std::abs(point.numeric_value - special) < 1e-9) {
 //             point.is_special = true;
 //             CHIMERGE_LOG_TRACE("Special value ", special, " at index ", i);
 //             break;
 //           }
 //         }
 //       }
 //       data.push_back(point);
 //     }
 //   } else if (Rf_isNumeric(x)) {
 //     NumericVector nv = as<NumericVector>(x);
 //     CHIMERGE_LOG_DEBUG("Processing numeric vector with ", nv.size(), " elements");
 //     for (int i = 0; i < nv.size(); ++i) {
 //       DataPoint point;
 //       point.target = target[i];
 //       point.original_index = i;
 //       point.weight = has_weights ? weights[i] : 1.0;
 //       
 //       if (NumericVector::is_na(nv[i])) {
 //         point.is_missing = true;
 //         CHIMERGE_LOG_TRACE("Missing value at index ", i);
 //       } else {
 //         point.numeric_value = nv[i];
 //         for (double special : special_vals) {
 //           if (std::abs(point.numeric_value - special) < 1e-9) {
 //             point.is_special = true;
 //             CHIMERGE_LOG_TRACE("Special value ", special, " at index ", i);
 //             break;
 //           }
 //         }
 //       }
 //       data.push_back(point);
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Prepared ", data.size(), " numeric data points");
 //   return data;
 // }
 // 
 // std::vector<DataPoint> prepare_categorical_data(SEXP x, const LogicalVector& target,
 //                                                 const NumericVector& weights) {
 //   CHIMERGE_LOG_INFO("Preparing categorical data");
 //   CHIMERGE_TIMER("Categorical data preparation");
 //   
 //   std::vector<DataPoint> data;
 //   bool has_weights = weights.size() > 0;
 //   
 //   if (Rf_isFactor(x)) {
 //     IntegerVector iv = as<IntegerVector>(x);
 //     CharacterVector levels = iv.attr("levels");
 //     CHIMERGE_LOG_DEBUG("Processing factor with ", iv.size(), " elements and ", levels.size(), " levels");
 //     
 //     for (int i = 0; i < iv.size(); ++i) {
 //       DataPoint point;
 //       point.target = target[i];
 //       point.original_index = i;
 //       point.weight = has_weights ? weights[i] : 1.0;
 //       
 //       if (IntegerVector::is_na(iv[i])) {
 //         point.is_missing = true;
 //         CHIMERGE_LOG_TRACE("Missing factor value at index ", i);
 //       } else {
 //         int level_idx = iv[i] - 1;  // R uses 1-based indexing
 //         if (level_idx >= 0 && level_idx < levels.size()) {
 //           point.categorical_value = as<std::string>(levels[level_idx]);
 //           CHIMERGE_LOG_TRACE("Factor level ", point.categorical_value, " at index ", i);
 //         } else {
 //           point.is_missing = true;
 //           CHIMERGE_LOG_TRACE("Invalid factor level at index ", i);
 //         }
 //       }
 //       data.push_back(point);
 //     }
 //   } else if (Rf_isString(x)) {
 //     CharacterVector cv = as<CharacterVector>(x);
 //     CHIMERGE_LOG_DEBUG("Processing character vector with ", cv.size(), " elements");
 //     for (int i = 0; i < cv.size(); ++i) {
 //       DataPoint point;
 //       point.target = target[i];
 //       point.original_index = i;
 //       point.weight = has_weights ? weights[i] : 1.0;
 //       
 //       if (CharacterVector::is_na(cv[i])) {
 //         point.is_missing = true;
 //         CHIMERGE_LOG_TRACE("Missing character value at index ", i);
 //       } else {
 //         point.categorical_value = as<std::string>(cv[i]);
 //         CHIMERGE_LOG_TRACE("Character value ", point.categorical_value, " at index ", i);
 //       }
 //       data.push_back(point);
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Prepared ", data.size(), " categorical data points");
 //   return data;
 // }
 // 
 // // =====================================================================
 // // SECTION 13: TRANSFORMATION APPLICATION
 // // =====================================================================
 // 
 // NumericVector apply_numeric_woe_transform(SEXP x, const std::vector<NumericBin>& bins,
 //                                           const std::vector<SpecialBin>& special_bins,
 //                                           const std::vector<double>& special_vals) {
 //   CHIMERGE_LOG_INFO("Applying numeric WoE transformation");
 //   CHIMERGE_TIMER("Numeric WoE transform");
 //   
 //   NumericVector result;
 //   
 //   if (Rf_isInteger(x)) {
 //     IntegerVector iv = as<IntegerVector>(x);
 //     result = NumericVector(iv.size(), NA_REAL);
 //     
 //     for (int i = 0; i < iv.size(); ++i) {
 //       if (IntegerVector::is_na(iv[i])) {
 //         for (const auto& sbin : special_bins) {
 //           if (sbin.type == "missing") {
 //             result[i] = sbin.woe;
 //             CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i);
 //             break;
 //           }
 //         }
 //       } else {
 //         double val = iv[i];
 //         bool found = false;
 //         
 //         // Check special values
 //         for (double special : special_vals) {
 //           if (std::abs(val - special) < 1e-9) {
 //             for (const auto& sbin : special_bins) {
 //               if (sbin.type == "special_numeric") {
 //                 result[i] = sbin.woe;
 //                 found = true;
 //                 CHIMERGE_LOG_TRACE("Applied special WoE at index ", i);
 //                 break;
 //               }
 //             }
 //             break;
 //           }
 //         }
 //         
 //         // Find appropriate bin using binary search
 //         if (!found) {
 //           auto it = std::lower_bound(bins.begin(), bins.end(), val,
 //                                      [](const NumericBin& bin, double value) {
 //                                        return bin.upper_bound <= value;
 //                                      });
 //           if (it != bins.end() && val >= it->lower_bound && val < it->upper_bound) {
 //             result[i] = it->woe;
 //             CHIMERGE_LOG_TRACE("Applied regular WoE at index ", i);
 //           }
 //         }
 //       }
 //     }
 //   } else if (Rf_isNumeric(x)) {
 //     NumericVector nv = as<NumericVector>(x);
 //     result = NumericVector(nv.size(), NA_REAL);
 //     
 //     for (int i = 0; i < nv.size(); ++i) {
 //       if (NumericVector::is_na(nv[i])) {
 //         for (const auto& sbin : special_bins) {
 //           if (sbin.type == "missing") {
 //             result[i] = sbin.woe;
 //             CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i);
 //             break;
 //           }
 //         }
 //       } else {
 //         double val = nv[i];
 //         bool found = false;
 //         
 //         for (double special : special_vals) {
 //           if (std::abs(val - special) < 1e-9) {
 //             for (const auto& sbin : special_bins) {
 //               if (sbin.type == "special_numeric") {
 //                 result[i] = sbin.woe;
 //                 found = true;
 //                 CHIMERGE_LOG_TRACE("Applied special WoE at index ", i);
 //                 break;
 //               }
 //             }
 //             break;
 //           }
 //         }
 //         
 //         // Find appropriate bin using binary search
 //         if (!found) {
 //           auto it = std::lower_bound(bins.begin(), bins.end(), val,
 //                                      [](const NumericBin& bin, double value) {
 //                                        return bin.upper_bound <= value;
 //                                      });
 //           if (it != bins.end() && val >= it->lower_bound && val < it->upper_bound) {
 //             result[i] = it->woe;
 //             CHIMERGE_LOG_TRACE("Applied regular WoE at index ", i);
 //           }
 //         }
 //       }
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Applied WoE transformation to ", result.size(), " values");
 //   return result;
 // }
 // 
 // CharacterVector apply_numeric_bin_transform(SEXP x, const std::vector<NumericBin>& bins,
 //                                             const std::vector<SpecialBin>& special_bins,
 //                                             const std::vector<double>& special_vals) {
 //   CHIMERGE_LOG_INFO("Applying numeric bin transformation");
 //   CHIMERGE_TIMER("Numeric bin transform");
 //   
 //   CharacterVector result;
 //   
 //   if (Rf_isInteger(x)) {
 //     IntegerVector iv = as<IntegerVector>(x);
 //     result = CharacterVector(iv.size());
 //     
 //     for (int i = 0; i < iv.size(); ++i) {
 //       if (IntegerVector::is_na(iv[i])) {
 //         result[i] = "[Missing]";
 //         CHIMERGE_LOG_TRACE("Applied missing bin label at index ", i);
 //       } else {
 //         double val = iv[i];
 //         bool found = false;
 //         
 //         for (double special : special_vals) {
 //           if (std::abs(val - special) < 1e-9) {
 //             result[i] = "[Special:" + std::to_string(static_cast<int>(special)) + "]";
 //             found = true;
 //             CHIMERGE_LOG_TRACE("Applied special bin label at index ", i);
 //             break;
 //           }
 //         }
 //         
 //         // Find appropriate bin using binary search
 //         if (!found) {
 //           auto it = std::lower_bound(bins.begin(), bins.end(), val,
 //                                      [](const NumericBin& bin, double value) {
 //                                        return bin.upper_bound <= value;
 //                                      });
 //           if (it != bins.end() && val >= it->lower_bound && val < it->upper_bound) {
 //             result[i] = it->label;
 //             CHIMERGE_LOG_TRACE("Applied regular bin label at index ", i);
 //           }
 //         }
 //       }
 //     }
 //   } else if (Rf_isNumeric(x)) {
 //     NumericVector nv = as<NumericVector>(x);
 //     result = CharacterVector(nv.size());
 //     
 //     for (int i = 0; i < nv.size(); ++i) {
 //       if (NumericVector::is_na(nv[i])) {
 //         result[i] = "[Missing]";
 //         CHIMERGE_LOG_TRACE("Applied missing bin label at index ", i);
 //       } else {
 //         double val = nv[i];
 //         bool found = false;
 //         
 //         for (double special : special_vals) {
 //           if (std::abs(val - special) < 1e-9) {
 //             result[i] = "[Special:" + std::to_string(static_cast<int>(special)) + "]";
 //             found = true;
 //             CHIMERGE_LOG_TRACE("Applied special bin label at index ", i);
 //             break;
 //           }
 //         }
 //         
 //         // Find appropriate bin using binary search
 //         if (!found) {
 //           auto it = std::lower_bound(bins.begin(), bins.end(), val,
 //                                      [](const NumericBin& bin, double value) {
 //                                        return bin.upper_bound <= value;
 //                                      });
 //           if (it != bins.end() && val >= it->lower_bound && val < it->upper_bound) {
 //             result[i] = it->label;
 //             CHIMERGE_LOG_TRACE("Applied regular bin label at index ", i);
 //           }
 //         }
 //       }
 //     }
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Applied bin transformation to ", result.size(), " values");
 //   return result;
 // }
 // 
 // // SECTION 14: KS AND GINI CALCULATION
 // void calculate_ks_gini_for_bins(std::vector<double>& ks_values,
 //                                 std::vector<double>& gini_values,
 //                                 const std::vector<int>& neg_counts,
 //                                 const std::vector<int>& pos_counts) {
 //   CHIMERGE_LOG_INFO("Calculating KS and Gini values");
 //   CHIMERGE_TIMER("KS/Gini calculation");
 //   
 //   size_t n_bins = neg_counts.size();
 //   if (n_bins == 0) {
 //     CHIMERGE_LOG_WARNING("Empty bins vector, skipping KS/Gini calculation");
 //     return;
 //   }
 //   
 //   // Calculate totals
 //   double total_neg = 0, total_pos = 0;
 //   for (size_t i = 0; i < n_bins; ++i) {
 //     total_neg += neg_counts[i];
 //     total_pos += pos_counts[i];
 //   }
 //   
 //   CHIMERGE_LOG_DEBUG("Total negatives: ", total_neg, ", positives: ", total_pos);
 //   
 //   // Calculate cumulative distributions
 //   double cum_neg = 0, cum_pos = 0;
 //   for (size_t i = 0; i < n_bins; ++i) {
 //     cum_neg += neg_counts[i];
 //     cum_pos += pos_counts[i];
 //     
 //     double cum_neg_rate = total_neg > 0 ? cum_neg / total_neg : 0;
 //     double cum_pos_rate = total_pos > 0 ? cum_pos / total_pos : 0;
 //     
 //     // KS for this bin
 //     double ks = std::abs(cum_pos_rate - cum_neg_rate);
 //     ks_values.push_back(ks);
 //     
 //     // Gini coefficient for this bin
 //     double total_bin = neg_counts[i] + pos_counts[i];
 //     double gini = 0.0;
 //     if (total_bin > 0) {
 //       double p1 = pos_counts[i] / total_bin;
 //       double p0 = neg_counts[i] / total_bin;
 //       gini = std::abs(p1 - p0);
 //     }
 //     gini_values.push_back(gini);
 //     
 //     CHIMERGE_LOG_TRACE("Bin ", i, " - KS: ", ks, ", Gini: ", gini);
 //   }
 //   
 //   CHIMERGE_LOG_INFO("Calculated KS and Gini for ", ks_values.size(), " bins");
 // }
 // 
 // // SECTION 15: MAIN BINNING ORCHESTRATOR
 // class OptimalBinningOrchestrator {
 // public:
 //   List process_numeric_feature(
 //       SEXP x,
 //       const std::string& feature_name,
 //       const LogicalVector& target,
 //       const NumericVector& weights,
 //       int min_bins, int max_bins, double sig_level, double min_size,
 //       double smooth, bool monotonic, double min_iv, int digits,
 //       const std::string& miss_policy,
 //       const std::vector<double>& special_vals,
 //       const std::string& method_str,
 //       bool cache) {
 //     
 //     CHIMERGE_LOG_INFO("Processing numeric feature: ", feature_name);
 //     CHIMERGE_TIMER("Numeric feature processing");
 //     
 //     std::vector<std::string> messages;
 //     
 //     // Prepare data
 //     auto data = prepare_numeric_data(x, target, weights, special_vals);
 //     
 //     // Sort data
 //     CHIMERGE_LOG_DEBUG("Sorting data");
 //     std::sort(data.begin(), data.end(),
 //               [](const DataPoint& a, const DataPoint& b) {
 //                 if (a.is_missing != b.is_missing) return !a.is_missing;
 //                 if (a.is_special != b.is_special) return !a.is_special;
 //                 return a.numeric_value < b.numeric_value;
 //               });
 //     
 //     // Apply binning
 //     CHIMERGE_LOG_INFO("Applying binning algorithm: ", method_str);
 //     ChiMergeModified algo(min_bins, max_bins, sig_level, min_size, cache);
 //     auto bins = algo.apply_numeric_binning(data);
 //     
 //     if (bins.empty()) {
 //       messages.push_back("No valid bins created");
 //       CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
 //     }
 //     
 //     // Process special values
 //     CHIMERGE_LOG_DEBUG("Processing special values");
 //     SpecialValueHandler special_handler(miss_policy, special_vals, {});
 //     auto special_bins = special_handler.process_special_values(data);
 //     
 //     // Calculate WoE
 //     size_t total_pos = 0, total_neg = 0;
 //     for (const auto& point : data) {
 //       if (!point.is_missing || miss_policy != "remove") {
 //         if (point.target) total_pos++;
 //         else total_neg++;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
 //     
 //     // Calculate WoE for all bins together
 //     size_t total_bins = bins.size() + special_bins.size();
 //     WoECalculator woe_calc(total_pos, total_neg, smooth, monotonic);
 //     woe_calc.calculate_woe_iv(bins);
 //     
 //     // Calculate WoE for special bins with the same smoothing parameters
 //     for (auto& sbin : special_bins) {
 //       double smoothed_pos = sbin.count_1 + smooth;
 //       double smoothed_neg = sbin.count_0 + smooth;
 //       double smoothed_total_pos = total_pos + total_bins * smooth;
 //       double smoothed_total_neg = total_neg + total_bins * smooth;
 //       
 //       if (smoothed_total_pos > 0 && smoothed_total_neg > 0) {
 //         double dist_pos = smoothed_pos / smoothed_total_pos;
 //         double dist_neg = smoothed_neg / smoothed_total_neg;
 //         
 //         if (dist_pos > 0 && dist_neg > 0) {
 //           sbin.woe = std::log(dist_pos / dist_neg);
 //           if (sbin.woe > 5) sbin.woe = 5;
 //           if (sbin.woe < -5) sbin.woe = -5;
 //           sbin.iv_contribution = (dist_pos - dist_neg) * sbin.woe;
 //         }
 //       }
 //     }
 //     
 //     // Format labels
 //     CHIMERGE_LOG_DEBUG("Formatting bin labels");
 //     for (size_t i = 0; i < bins.size(); ++i) {
 //       bins[i].label = format_numeric_bin_label(bins[i], i == 0, i == bins.size() - 1, digits);
 //     }
 //     
 //     // Combine regular bins and special bins for output
 //     std::vector<int> bin_ids;
 //     std::vector<std::string> bin_labels;
 //     std::vector<int> total_counts, neg_counts, pos_counts;
 //     std::vector<double> woes, ivs;
 //     
 //     // Add special bins first
 //     int bin_id = 1;
 //     for (const auto& sbin : special_bins) {
 //       bin_ids.push_back(bin_id++);
 //       bin_labels.push_back(sbin.label);
 //       total_counts.push_back(sbin.total());
 //       neg_counts.push_back(sbin.count_0);
 //       pos_counts.push_back(sbin.count_1);
 //       woes.push_back(sbin.woe);
 //       ivs.push_back(sbin.iv_contribution);
 //     }
 //     
 //     // Add regular bins
 //     for (const auto& bin : bins) {
 //       bin_ids.push_back(bin_id++);
 //       bin_labels.push_back(bin.label);
 //       total_counts.push_back(bin.total());
 //       neg_counts.push_back(bin.count_0);
 //       pos_counts.push_back(bin.count_1);
 //       woes.push_back(bin.woe);
 //       ivs.push_back(bin.iv_contribution);
 //     }
 //     
 //     // Calculate KS and Gini
 //     std::vector<double> ks_values, gini_values;
 //     calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
 //     
 //     DataFrame bins_df = DataFrame::create(
 //       Named("bin_id") = bin_ids,
 //       Named("bin_label") = bin_labels,
 //       Named("total_count") = total_counts,
 //       Named("neg_count") = neg_counts,
 //       Named("pos_count") = pos_counts,
 //       Named("woe") = woes,
 //       Named("iv") = ivs,
 //       Named("ks") = ks_values,
 //       Named("gini") = gini_values
 //     );
 //     
 //     // Create transformation DataFrame
 //     NumericVector woe_transform = apply_numeric_woe_transform(x, bins, special_bins, special_vals);
 //     CharacterVector bin_transform = apply_numeric_bin_transform(x, bins, special_bins, special_vals);
 //     
 //     DataFrame transform_df = DataFrame::create(
 //       Named(feature_name) = x,
 //       Named(feature_name + "_bin") = bin_transform,
 //       Named(feature_name + "_woe") = woe_transform
 //     );
 //     
 //     // Detect variable type
 //     VariableType var_type = detect_r_variable_type(x);
 //     std::string var_type_str;
 //     switch (var_type) {
 //     case VariableType::NUMERIC_CONTINUOUS: var_type_str = "numeric_continuous"; break;
 //     case VariableType::NUMERIC_DISCRETE: var_type_str = "numeric_discrete"; break;
 //     case VariableType::BOOLEAN: var_type_str = "boolean"; break;
 //     default: var_type_str = "numeric";
 //     }
 //     
 //     // Calculate total IV including special bins
 //     double total_iv = woe_calc.get_total_iv();
 //     for (const auto& sbin : special_bins) {
 //       total_iv += sbin.iv_contribution;
 //     }
 //     
 //     CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, ", Bins: ", bin_ids.size());
 //     
 //     // Return results
 //     List result = List::create(
 //       Named("bins") = bins_df,
 //       Named("total_iv") = total_iv,
 //       Named("variable_type") = var_type_str,
 //       Named("method_used") = method_str,
 //       Named("n_bins") = bin_ids.size(),
 //       Named("total_samples") = (int)data.size(),
 //       Named("event_rate") = (double)total_pos / (total_pos + total_neg),
 //       Named("messages") = messages.empty() ? "Success" : messages[0],
 //                                                                  Named("transform") = transform_df
 //     );
 //     
 //     return result;
 //   }
 //   
 //   List process_categorical_feature(
 //       SEXP x,
 //       const std::string& feature_name,
 //       const LogicalVector& target,
 //       const NumericVector& weights,
 //       int min_bins, int max_bins, double min_size,
 //       double smooth, bool monotonic, double min_iv,
 //       const std::string& miss_policy,
 //       int max_cat, double rare_pct, const std::string& cat_sep) {
 //     
 //     CHIMERGE_LOG_INFO("Processing categorical feature: ", feature_name);
 //     CHIMERGE_TIMER("Categorical feature processing");
 //     
 //     std::vector<std::string> messages;
 //     
 //     // Prepare data
 //     auto data = prepare_categorical_data(x, target, weights);
 //     
 //     // Apply binning
 //     CHIMERGE_LOG_DEBUG("Applying categorical binning");
 //     auto bins = apply_categorical_binning(data, min_bins, max_bins, min_size,
 //                                           max_cat, rare_pct, cat_sep);
 //     
 //     if (bins.empty()) {
 //       messages.push_back("No valid bins created");
 //       CHIMERGE_LOG_WARNING("No valid bins created for feature: ", feature_name);
 //     }
 //     
 //     // Process special values (missing categories)
 //     CHIMERGE_LOG_DEBUG("Processing special values");
 //     SpecialValueHandler special_handler(miss_policy, {}, {});
 //     auto special_bins = special_handler.process_special_values(data);
 //     
 //     // Calculate WoE
 //     size_t total_pos = 0, total_neg = 0;
 //     for (const auto& point : data) {
 //       if (!point.is_missing || miss_policy != "remove") {
 //         if (point.target) total_pos++;
 //         else total_neg++;
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_DEBUG("Total positives: ", total_pos, ", negatives: ", total_neg);
 //     
 //     // Calculate WoE for all bins together
 //     size_t total_bins = bins.size() + special_bins.size();
 //     WoECalculator woe_calc(total_pos, total_neg, smooth, monotonic);
 //     woe_calc.calculate_woe_iv(bins);
 //     
 //     // Calculate WoE for special bins with the same smoothing parameters
 //     for (auto& sbin : special_bins) {
 //       double smoothed_pos = sbin.count_1 + smooth;
 //       double smoothed_neg = sbin.count_0 + smooth;
 //       double smoothed_total_pos = total_pos + total_bins * smooth;
 //       double smoothed_total_neg = total_neg + total_bins * smooth;
 //       
 //       if (smoothed_total_pos > 0 && smoothed_total_neg > 0) {
 //         double dist_pos = smoothed_pos / smoothed_total_pos;
 //         double dist_neg = smoothed_neg / smoothed_total_neg;
 //         
 //         if (dist_pos > 0 && dist_neg > 0) {
 //           sbin.woe = std::log(dist_pos / dist_neg);
 //           if (sbin.woe > 5) sbin.woe = 5;
 //           if (sbin.woe < -5) sbin.woe = -5;
 //           sbin.iv_contribution = (dist_pos - dist_neg) * sbin.woe;
 //         }
 //       }
 //     }
 //     
 //     // Combine special bins and regular bins
 //     std::vector<int> bin_ids;
 //     std::vector<std::string> bin_labels;
 //     std::vector<int> total_counts, neg_counts, pos_counts;
 //     std::vector<double> woes, ivs;
 //     
 //     // Add special bins first
 //     int bin_id = 1;
 //     for (const auto& sbin : special_bins) {
 //       bin_ids.push_back(bin_id++);
 //       bin_labels.push_back(sbin.label);
 //       total_counts.push_back(sbin.total());
 //       neg_counts.push_back(sbin.count_0);
 //       pos_counts.push_back(sbin.count_1);
 //       woes.push_back(sbin.woe);
 //       ivs.push_back(sbin.iv_contribution);
 //     }
 //     
 //     // Add regular bins
 //     for (const auto& bin : bins) {
 //       bin_ids.push_back(bin_id++);
 //       bin_labels.push_back(bin.label);
 //       total_counts.push_back(bin.total());
 //       neg_counts.push_back(bin.count_0);
 //       pos_counts.push_back(bin.count_1);
 //       woes.push_back(bin.woe);
 //       ivs.push_back(bin.iv_contribution);
 //     }
 //     
 //     // Calculate KS and Gini
 //     std::vector<double> ks_values, gini_values;
 //     calculate_ks_gini_for_bins(ks_values, gini_values, neg_counts, pos_counts);
 //     
 //     DataFrame bins_df = DataFrame::create(
 //       Named("bin_id") = bin_ids,
 //       Named("bin_label") = bin_labels,
 //       Named("total_count") = total_counts,
 //       Named("neg_count") = neg_counts,
 //       Named("pos_count") = pos_counts,
 //       Named("woe") = woes,
 //       Named("iv") = ivs,
 //       Named("ks") = ks_values,
 //       Named("gini") = gini_values
 //     );
 //     
 //     // Create transformation vectors
 //     int n = Rf_length(x);
 //     NumericVector woe_transform(n, NA_REAL);
 //     CharacterVector bin_transform(n);
 //     
 //     // Create category to WoE mapping (including special bins)
 //     std::map<std::string, double> cat_woe_map;
 //     std::map<std::string, std::string> cat_bin_map;
 //     
 //     // Map regular categories
 //     for (const auto& bin : bins) {
 //       for (const auto& cat : bin.categories) {
 //         cat_woe_map[cat] = bin.woe;
 //         cat_bin_map[cat] = bin.label;
 //       }
 //     }
 //     
 //     // Apply transformations
 //     if (Rf_isFactor(x)) {
 //       IntegerVector iv = as<IntegerVector>(x);
 //       CharacterVector levels = iv.attr("levels");
 //       
 //       for (int i = 0; i < n; ++i) {
 //         if (IntegerVector::is_na(iv[i])) {
 //           bin_transform[i] = "[Missing]";
 //           // Find missing bin WoE
 //           for (const auto& sbin : special_bins) {
 //             if (sbin.type == "missing") {
 //               woe_transform[i] = sbin.woe;
 //               CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i, " for factor");
 //               break;
 //             }
 //           }
 //         } else {
 //           int level_idx = iv[i] - 1;
 //           if (level_idx >= 0 && level_idx < levels.size()) {
 //             std::string cat = as<std::string>(levels[level_idx]);
 //             auto woe_it = cat_woe_map.find(cat);
 //             auto bin_it = cat_bin_map.find(cat);
 //             
 //             if (woe_it != cat_woe_map.end()) {
 //               woe_transform[i] = woe_it->second;
 //               bin_transform[i] = bin_it->second;
 //               CHIMERGE_LOG_TRACE("Applied WoE for category ", cat, " at index ", i);
 //             }
 //           }
 //         }
 //       }
 //     } else if (Rf_isString(x)) {
 //       CharacterVector cv = as<CharacterVector>(x);
 //       
 //       for (int i = 0; i < n; ++i) {
 //         if (CharacterVector::is_na(cv[i])) {
 //           bin_transform[i] = "[Missing]";
 //           // Find missing bin WoE
 //           for (const auto& sbin : special_bins) {
 //             if (sbin.type == "missing") {
 //               woe_transform[i] = sbin.woe;
 //               CHIMERGE_LOG_TRACE("Applied missing WoE at index ", i, " for character");
 //               break;
 //             }
 //           }
 //         } else {
 //           std::string cat = as<std::string>(cv[i]);
 //           auto woe_it = cat_woe_map.find(cat);
 //           auto bin_it = cat_bin_map.find(cat);
 //           
 //           if (woe_it != cat_woe_map.end()) {
 //             woe_transform[i] = woe_it->second;
 //             bin_transform[i] = bin_it->second;
 //             CHIMERGE_LOG_TRACE("Applied WoE for category ", cat, " at index ", i);
 //           }
 //         }
 //       }
 //     }
 //     
 //     // Create transformation DataFrame
 //     DataFrame transform_df = DataFrame::create(
 //       Named(feature_name) = x,
 //       Named(feature_name + "_bin") = bin_transform,
 //       Named(feature_name + "_woe") = woe_transform
 //     );
 //     
 //     // Calculate total IV including special bins
 //     double total_iv = woe_calc.get_total_iv();
 //     for (const auto& sbin : special_bins) {
 //       total_iv += sbin.iv_contribution;
 //     }
 //     
 //     CHIMERGE_LOG_INFO("Feature processing completed - Total IV: ", total_iv, ", Bins: ", bin_ids.size());
 //     
 //     // Return results
 //     List result = List::create(
 //       Named("bins") = bins_df,
 //       Named("total_iv") = total_iv,
 //       Named("variable_type") = "categorical",
 //       Named("method_used") = "categorical_chimerge",
 //       Named("n_bins") = bin_ids.size(),
 //       Named("total_samples") = (int)data.size(),
 //       Named("event_rate") = (double)total_pos / (total_pos + total_neg),
 //       Named("messages") = messages.empty() ? "Success" : messages[0],
 //                                                                  Named("transform") = transform_df
 //     );
 //     
 //     return result;
 //   }
 // };
 // 
 // // SECTION 16: MAIN EXPORTED FUNCTION
 // 
 // //' ChiMerge Optimal Binning with Weight of Evidence (WoE)
 // //'
 // //' @description
 // //' Performs optimal binning on numeric and categorical variables using multiple **ChiMerge** algorithm
 // //' variations, computing **Weight of Evidence (WoE)** and **Information Value (IV)** for each bin.
 // //' Implements comprehensive binning strategies including:
 // //' \itemize{
 // //'   \item \strong{ChiMerge Original} (Kerber, 1992): bottom-up merging based on chi-square independence test
 // //'   \item \strong{ChiMerge Modified}: handles small expected frequencies with Yates' continuity correction
 // //'   \item \strong{ChiMerge Optimized}: includes caching and performance optimizations
 // //'   \item \strong{D2 Algorithm}: entropy-based recursive splitting
 // //'   \item \strong{Hybrid WoE}: combines multiple strategies for optimal results
 // //' }
 // //'
 // //' @details
 // //' \strong{1) Chi-Square Test of Independence}
 // //'
 // //' For adjacent intervals \eqn{i} and \eqn{j}, the chi-square statistic is:
 // //' \deqn{\chi^2 = \sum_{k=0}^{1} \sum_{l \in \\{i,j\\}} \frac{(O_{lk} - E_{lk})^2}{E_{lk}}}
 // //' where:
 // //' \itemize{
 // //'    \item \eqn{O_{lk}} = observed frequency in interval \eqn{l}, class \eqn{k}
 // //'    \item \eqn{E_{lk} = \frac{R_l \times C_k}{N}} = expected frequency under independence
 // //'    \item \eqn{R_l} = row total for interval \eqn{l}
 // //'    \item \eqn{C_k} = column total for class \eqn{k}
 // //'    \item \eqn{N} = total observations
 // //' }
 // //' With df = 1, critical values at common significance levels: \eqn{\chi^2_{0.10,1} = 2.706},
 // //' \eqn{\chi^2_{0.05,1} = 3.841}, \eqn{\chi^2_{0.01,1} = 6.635}.
 // //'
 // //' \strong{2) Weight of Evidence (WoE)}
 // //'
 // //' For bin \eqn{i}, WoE measures the strength of relationship with the target:
 // //' \deqn{\text{WoE}_i = \ln\left(\frac{P(X \text{ in bin}_i, Y=1)}{P(X \text{ in bin}_i, Y=0)}\right) = \ln\left(\frac{\text{Dist}_{\text{Good},i}}{\text{Dist}_{\text{Bad},i}}\right)}
 // //' where:
 // //' \deqn{\text{Dist}_{\text{Good},i} = \frac{n_{1i}}{n_1}, \quad \text{Dist}_{\text{Bad},i} = \frac{n_{0i}}{n_0}}
 // //'
 // //' \strong{Laplace Smoothing}: To handle zero frequencies, applies smoothing factor \eqn{\lambda}:
 // //' \deqn{\text{WoE}_i = \ln\left(\frac{(n_{1i} + \lambda) / (n_1 + k\lambda)}{(n_{0i} + \lambda) / (n_0 + k\lambda)}\right)}
 // //' where \eqn{k} is the number of bins.
 // //'
 // //' \strong{3) Information Value (IV)}
 // //'
 // //' Total IV quantifies the predictive power of the variable:
 // //' \deqn{\text{IV} = \sum_{i=1}^{k} (\text{Dist}_{\text{Good},i} - \text{Dist}_{\text{Bad},i}) \times \text{WoE}_i}
 // //'
 // //' Interpretation guidelines:
 // //' \itemize{
 // //'   \item IV \eqn{< 0.02}: Useless predictor
 // //'   \item \eqn{0.02 \leq} IV \eqn{< 0.1}: Weak predictor
 // //'   \item \eqn{0.1 \leq} IV \eqn{< 0.3}: Medium predictor
 // //'   \item \eqn{0.3 \leq} IV \eqn{< 0.5}: Strong predictor
 // //'   \item IV \eqn{\geq 0.5}: Suspicious (potential overfitting)
 // //' }
 // //'
 // //' \strong{4) Monotonicity Enforcement}
 // //'
 // //' When \code{monotonic = TRUE}, applies isotonic regression using the Pool Adjacent Violators
 // //' Algorithm (PAVA) to ensure monotonic WoE values across bins, preserving the ordinal relationship
 // //' while minimizing weighted squared deviations.
 // //'
 // //' \strong{5) Special Value Handling}
 // //'
 // //' Missing values and special codes are handled according to \code{miss_policy}:
 // //' \itemize{
 // //'   \item \code{separate}: Creates separate bin(s) for missing/special values
 // //'   \item \code{remove}: Excludes missing values from binning
 // //'   \item \code{impute}: Imputes with mode (categorical) or median (numeric)
 // //'   \item \code{merge}: Merges with nearest bin based on event rate
 // //' }
 // //'
 // //' @param data DataFrame containing features and target variable. Must contain at least the target column
 // //'   and one feature column.
 // //'
 // //' @param target_col String specifying the name of the binary target column. The target must be binary
 // //'   (0/1, TRUE/FALSE, or two-level factor).
 // //'
 // //' @param feature_cols CharacterVector of feature column names to process. These columns can be numeric,
 // //'   integer, logical, character, or factor types. Date/time columns are automatically skipped.
 // //'
 // //' @param min_bins Integer specifying minimum number of bins to create. Must be at least 2.
 // //'   Default: 2
 // //'
 // //' @param max_bins Integer specifying maximum number of bins allowed. Must be greater than or equal to
 // //'   min_bins. Values above 100 may lead to overfitting.
 // //'   Default: 10
 // //'
 // //' @param sig_level Numeric significance level for chi-square test used in ChiMerge algorithm.
 // //'   Must be between 0 and 1. Common values: 0.10 (90\% confidence), 0.05 (95\% confidence),
 // //'   0.01 (99\% confidence). Lower values create fewer bins.
 // //'   Default: 0.05
 // //'
 // //' @param min_size Numeric minimum bin size as proportion of total observations. Must be between 0 and 1.
 // //'   Prevents creation of very small bins. For example, 0.05 means each bin must contain at least 5\\%
 // //'   of observations.
 // //'   Default: 0.05
 // //'
 // //' @param smooth Numeric Laplace smoothing factor for WoE calculation. Must be non-negative.
 // //'   Prevents undefined WoE values when bins have zero counts. Higher values provide more smoothing.
 // //'   Default: 0.5
 // //'
 // //' @param monotonic Logical indicating whether to enforce monotonic WoE values across bins using
 // //'   isotonic regression. Useful for regulatory compliance and model interpretability.
 // //'   Default: TRUE
 // //'
 // //' @param min_iv Numeric minimum Information Value threshold for feature selection. Features with
 // //'   IV below this threshold generate a warning. Must be non-negative.
 // //'   Default: 0.01
 // //'
 // //' @param digits Integer number of decimal places for numeric bin boundaries in labels.
 // //'   Must be between 0 and 10. Affects display only, not calculation precision.
 // //'   Default: 4
 // //'
 // //' @param miss_policy String specifying how to handle missing values. Options:
 // //'   \itemize{
 // //'     \item \code{separate}: Create separate bin(s) for missing values (default)
 // //'     \item \code{remove}: Exclude missing values from analysis
 // //'     \item \code{impute}: Impute with mode (categorical) or median (numeric)
 // //'     \item \code{merge}: Merge with nearest bin based on event rate
 // //'   }
 // //'   Default: separate
 // //'
 // //' @param special_vals NumericVector of special values to handle separately (e.g., -999, -888 for
 // //'   special codes). These values are isolated in separate bins when encountered.
 // //'   Default: empty vector (no special values)
 // //'
 // //' @param max_cat Integer maximum number of categories before automatic grouping for categorical variables.
 // //'   Must be at least 2. Categories beyond this limit are grouped based on event rate similarity.
 // //'   Default: 20
 // //'
 // //' @param rare_pct Numeric threshold for rare category grouping as proportion of total.
 // //'   Categories with frequency below this threshold are combined. Must be between 0 and 0.5.
 // //'   Default: 0.01 (1\%)
 // //'
 // //' @param cat_sep String separator used when combining multiple categories into a single bin label.
 // //'   Should be a pattern unlikely to appear in actual category names.
 // //'   Default: "\%;\%"
 // //'
 // //' @param method String specifying the binning algorithm to use. Options:
 // //'   \itemize{
 // //'     \item \code{chimerge}: Original ChiMerge algorithm
 // //'     \item \code{chimerge_mod}: Modified ChiMerge with Yates correction (default)
 // //'     \item \code{chimerge_opt}: Optimized ChiMerge with caching
 // //'     \item \code{d2}: D2 entropy-based algorithm
 // //'     \item \code{hybrid}: Hybrid approach combining multiple methods
 // //'   }
 // //'   Default: chimerge_mod
 // //'
 // //' @param parallel Logical indicating whether to use parallel processing via OpenMP for multiple features.
 // //'   Requires OpenMP support in compilation.
 // //'   Default: FALSE
 // //'
 // //' @param cache Logical indicating whether to enable chi-square value caching for performance optimization.
 // //'   Recommended for large datasets.
 // //'   Default: TRUE
 // //'
 // //' @param weights Optional NumericVector of observation weights for weighted binning. Length must equal
 // //'   number of rows in data. All weights must be non-negative. NULL indicates equal weights.
 // //'   Default: NULL
 // //'
 // //' @param verbose Logical indicating whether to print detailed processing information.
 // //'   Default: FALSE
 // //'
 // //' @return
 // //' Named list with one element per processed feature, each containing:
 // //' \itemize{
 // //'   \item \code{bins}: data.frame with columns:
 // //'     \itemize{
 // //'       \item \code{bin_id}: Sequential bin identifier
 // //'       \item \code{bin_label}: Formatted bin range/categories
 // //'       \item \code{total_count}: Total observations in bin
 // //'       \item \code{neg_count}: Count of negative class (Y=0)
 // //'       \item \code{pos_count}: Count of positive class (Y=1)
 // //'       \item \code{woe}: Weight of Evidence value
 // //'       \item \code{iv}: Information Value contribution
 // //'       \item \code{ks}: Kolmogorov-Smirnov statistic
 // //'       \item \code{gini}: Gini impurity coefficient
 // //'     }
 // //'   \item \code{total_iv}: Total Information Value for the feature
 // //'   \item \code{variable_type}: Detected type (numeric_continuous, numeric_discrete, categorical, boolean)
 // //'   \item \code{method_used}: Algorithm actually used for binning
 // //'   \item \code{n_bins}: Final number of bins created
 // //'   \item \code{total_samples}: Total valid observations processed
 // //'   \item \code{event_rate}: Overall positive class rate
 // //'   \item \code{messages}: Processing messages or warnings
 // //'   \item \code{transform}: data.frame with original values and transformations:
 // //'     \itemize{
 // //'       \item \code{[feature]}: Original feature values
 // //'       \item \code{[feature]_bin}: Assigned bin labels
 // //'       \item \code{[feature]_woe}: Assigned WoE values
 // //'     }
 // //' }
 // //'
 // //' @section Algorithm Details:
 // //' The ChiMerge algorithm proceeds as follows:
 // //' \enumerate{
 // //'   \item \strong{Initialization}: Create initial bins (one per unique value for numeric, one per category for categorical)
 // //'   \item \strong{Iterative Merging}:
 // //'     \itemize{
 // //'       \item Calculate chi-square statistic for all adjacent bin pairs
 // //'       \item Find pair with minimum chi-square value
 // //'       \item If min chi-square < threshold and bins > min_bins, merge the pair
 // //'       \item Repeat until stopping criteria met
 // //'     }
 // //'   \item \strong{Post-processing}:
 // //'     \itemize{
 // //'       \item Ensure max_bins constraint
 // //'       \item Calculate WoE and IV for final bins
 // //'       \item Apply monotonicity constraint if requested
 // //'     }
 // //' }
 // //'
 // //' @note
 // //' \itemize{
 // //'   \item Date/time variables are automatically excluded
 // //'   \item Factor variables are processed as categorical
 // //'   \item Integer variables with 20 or fewer unique values are treated as discrete
 // //'   \item Special values (-999, -888, etc.) can be isolated in separate bins
 // //'   \item Uses (-inf,x) and [x,+inf) notation for unbounded intervals
 // //' }
 // //'
 // //' @references
 // //' \itemize{
 // //'   \item Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128).
 // //'   \item Liu, H., Hussain, F., Tan, C. L., & Dash, M. (2002). Discretization: An enabling technique. Data mining and knowledge discovery, 6(4), 393-423.
 // //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
 // //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
 // //' }
 // //'
 // //' @examples
 // //' \dontrun{
 // //' # Load credit data
 // //' data <- scorecard::germancredit
 // //' data$target <- ifelse(data$creditability == "good", 1, 0)
 // //'
 // //' # Basic binning with all defaults
 // //' result <- chimerge_woe(
 // //'   data = data,
 // //'   target_col = "target",
 // //'   feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
 // //' )
 // //'
 // //' # Custom binning with specific parameters
 // //' result_custom <- chimerge_woe(
 // //'   data = data,
 // //'   target_col = "target",
 // //'   feature_cols = names(data)[1:10],
 // //'   min_bins = 3,              # Minimum 3 bins
 // //'   max_bins = 8,               # Maximum 8 bins
 // //'   sig_level = 0.10,           # 90% confidence level
 // //'   min_size = 0.03,            # Each bin at least 3% of data
 // //'   smooth = 1.0,               # Higher smoothing
 // //'   monotonic = TRUE,           # Enforce monotonic WoE
 // //'   min_iv = 0.05,              # Warn if IV < 0.05
 // //'   digits = 2,                 # 2 decimal places in labels
 // //'   miss_policy = "separate",   # Missing values in separate bin
 // //'   special_vals = c(-999, -888), # Special codes to isolate
 // //'   max_cat = 15,               # Group if >15 categories
 // //'   rare_pct = 0.02,            # Group categories <2% frequency
 // //'   cat_sep = " | ",            # Use pipe separator
 // //'   method = "chimerge_mod",    # Modified ChiMerge
 // //'   parallel = FALSE,           # No parallel processing
 // //'   cache = TRUE,               # Enable caching
 // //'   weights = NULL,             # No observation weights
 // //'   verbose = TRUE              # Show detailed logs
 // //' )
 // //'
 // //' # Extract results for a specific feature
 // //' duration_bins <- result$duration.in.month$bins
 // //' duration_iv <- result$duration.in.month$total_iv
 // //' duration_transform <- result$duration.in.month$transform
 // //' }
 // //'
 // //' @export
 // // [[Rcpp::export]]
 // List chimerge_woe(
 //     DataFrame data,
 //     std::string target_col,
 //     CharacterVector feature_cols,
 //     int min_bins = 2,
 //     int max_bins = 5,
 //     double sig_level = 0.05,
 //     double min_size = 0.05,
 //     double smooth = 0,
 //     bool monotonic = false,
 //     double min_iv = 0.01,
 //     int digits = 4,
 //     std::string miss_policy = "separate",
 //     NumericVector special_vals = NumericVector::create(),
 //     int max_cat = 20,
 //     double rare_pct = 0.01,
 //     std::string cat_sep = "%;%",
 //     std::string method = "chimerge_mod",
 //     bool parallel = false,
 //     bool cache = true,
 //     Nullable<NumericVector> weights = R_NilValue,
 //     bool verbose = false) {
 //   
 //   try {
 //     // Initialize logger with verbose setting
 //     chimerge_logging::global_logger.set_verbose(verbose);
 //     
 //     CHIMERGE_LOG_INFO("Starting ChiMerge WoE processing");
 //     CHIMERGE_TIMER("Total processing");
 //     
 //     // Validate parameters
 //     auto param_validation = InputValidator::validate_binning_parameters(
 //       min_bins, max_bins, sig_level, min_size, smooth, monotonic, min_iv, digits,
 //       miss_policy, max_cat, rare_pct, method, parallel, cache
 //     );
 //     
 //     if (!param_validation.is_valid) {
 //       std::string error_msg = "Parameter validation failed:\n";
 //       for (const auto& err : param_validation.errors) {
 //         error_msg += "  - " + err + "\n";
 //       }
 //       CHIMERGE_LOG_ERROR(error_msg);
 //       Rcpp::stop(error_msg);
 //     }
 //     
 //     // Extract target
 //     LogicalVector target;
 //     SEXP target_sexp = data[target_col];
 //     
 //     if (Rf_isLogical(target_sexp)) {
 //       target = as<LogicalVector>(target_sexp);
 //     } else if (Rf_isInteger(target_sexp)) {
 //       IntegerVector iv = as<IntegerVector>(target_sexp);
 //       target = LogicalVector(iv.size());
 //       for (int i = 0; i < iv.size(); ++i) {
 //         target[i] = (iv[i] == 1);
 //       }
 //     } else if (Rf_isNumeric(target_sexp)) {
 //       NumericVector nv = as<NumericVector>(target_sexp);
 //       target = LogicalVector(nv.size());
 //       for (int i = 0; i < nv.size(); ++i) {
 //         target[i] = (nv[i] == 1.0);
 //       }
 //     }
 //     
 //     // Extract weights
 //     NumericVector weight_vec;
 //     if (weights.isNotNull()) {
 //       weight_vec = weights.get();
 //     } else {
 //       weight_vec = NumericVector(data.nrows(), 1.0);
 //     }
 //     
 //     // Convert special values
 //     std::vector<double> special_vals_vec;
 //     for (int i = 0; i < special_vals.size(); ++i) {
 //       special_vals_vec.push_back(special_vals[i]);
 //     }
 //     
 //     // Process features
 //     List results;
 //     OptimalBinningOrchestrator orchestrator;
 //     std::vector<std::string> features = Rcpp::as<std::vector<std::string>>(feature_cols);
 //     
 //     CHIMERGE_LOG_INFO("Processing ", features.size(), " features");
 //     
 //     for (const std::string& feature : features) {
 //       try {
 //         CHIMERGE_LOG_INFO("Processing feature: ", feature);
 //         
 //         if (!data.containsElementNamed(feature.c_str())) {
 //           Rcpp::warning("Feature '" + feature + "' not found in data");
 //           CHIMERGE_LOG_WARNING("Feature '", feature, "' not found in data");
 //           continue;
 //         }
 //         
 //         SEXP feature_sexp = data[feature];
 //         
 //         // Check variable type
 //         VariableType var_type = detect_r_variable_type(feature_sexp);
 //         
 //         if (var_type == VariableType::DATE_TIME) {
 //           Rcpp::warning("Feature '" + feature + "' is a date/time type, skipping");
 //           CHIMERGE_LOG_WARNING("Feature '", feature, "' is a date/time type, skipping");
 //           continue;
 //         }
 //         
 //         if (var_type == VariableType::UNSUPPORTED) {
 //           Rcpp::warning("Feature '" + feature + "' has unsupported type, skipping");
 //           CHIMERGE_LOG_WARNING("Feature '", feature, "' has unsupported type, skipping");
 //           continue;
 //         }
 //         
 //         List feature_result;
 //         
 //         // Process based on type
 //         if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN ||
 //             Rf_isFactor(feature_sexp) || Rf_isString(feature_sexp)) {
 //           
 //           CHIMERGE_LOG_DEBUG("Processing as categorical feature");
 //           feature_result = orchestrator.process_categorical_feature(
 //             feature_sexp, feature, target, weight_vec,
 //             min_bins, max_bins, min_size,
 //             smooth, monotonic, min_iv,
 //             miss_policy,
 //             max_cat, rare_pct, cat_sep
 //           );
 //         } else {
 //           CHIMERGE_LOG_DEBUG("Processing as numeric feature");
 //           feature_result = orchestrator.process_numeric_feature(
 //             feature_sexp, feature, target, weight_vec,
 //             min_bins, max_bins, sig_level, min_size,
 //             smooth, monotonic, min_iv, digits,
 //             miss_policy, special_vals_vec,
 //             method, cache
 //           );
 //         }
 //         
 //         // Check IV threshold
 //         double iv = as<double>(feature_result["total_iv"]);
 //         if (iv < min_iv) {
 //           List messages = feature_result["messages"];
 //           feature_result["messages"] = "IV below threshold (" + std::to_string(iv) + ")";
 //           CHIMERGE_LOG_WARNING("Feature '", feature, "' has IV below threshold: ", iv);
 //         }
 //         
 //         results[feature] = feature_result;
 //         CHIMERGE_LOG_INFO("Completed processing feature: ", feature);
 //         
 //       } catch (const std::exception& e) {
 //         Rcpp::warning("Error processing feature '" + feature + "': " + e.what());
 //         CHIMERGE_LOG_ERROR("Error processing feature '", feature, "': ", e.what());
 //       }
 //     }
 //     
 //     CHIMERGE_LOG_INFO("ChiMerge WoE processing completed");
 //     return results;
 //     
 //   } catch (const std::exception& e) {
 //     CHIMERGE_LOG_ERROR("Error in chimerge_woe: ", e.what());
 //     Rcpp::stop(std::string("Error in chimerge_woe: ") + e.what());
 //   }
 // }
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 // =====================================================================
 // SECTION 17: GAINS TABLE METRICS CALCULATION
 // =====================================================================
 
 struct GainsMetrics {
   // Basic counts
   std::vector<int> bin_id;
   std::vector<std::string> bin_label;
   std::vector<int> total_count;
   std::vector<int> neg_count;
   std::vector<int> pos_count;
   std::vector<double> woe;
   std::vector<double> iv;
   
   // Percentages
   std::vector<double> pct_total;      // % of total population in bin
   std::vector<double> pct_pos;        // % of positives in bin
   std::vector<double> pct_neg;        // % of negatives in bin
   std::vector<double> event_rate;     // P(Y=1|bin)
   
   // Cumulative statistics
   std::vector<double> cum_total;      // Cumulative % of total
   std::vector<double> cum_pos;        // Cumulative % of positives (Sensitivity)
   std::vector<double> cum_neg;        // Cumulative % of negatives
   std::vector<double> cum_event_rate; // Cumulative event rate
   
   // Lift metrics
   std::vector<double> lift;           // Lift = event_rate / base_rate
   std::vector<double> cum_lift;       // Cumulative lift
   std::vector<double> capture_rate;   // Capture rate = cum_pos
   
   // KS and Gini
   std::vector<double> ks;             // |cum_pos - cum_neg|
   std::vector<double> gini;           // Gini coefficient per bin
   double max_ks;                      // Maximum KS value
   double auc;                          // Area Under ROC Curve
   double gini_coefficient;             // Overall Gini = 2*AUC - 1
   
   // Entropy and Information
   std::vector<double> entropy;        // -p*log(p) - (1-p)*log(1-p)
   std::vector<double> divergence;     // KL divergence from base rate
   std::vector<double> info_gain;      // Information gain
   
   // Risk metrics
   std::vector<double> odds;           // p/(1-p)
   std::vector<double> log_odds;       // log(odds)
   std::vector<double> relative_risk;  // event_rate / base_rate
   
   // Performance metrics
   std::vector<double> precision;      // PPV = TP/(TP+FP)
   std::vector<double> npv;            // NPV = TN/(TN+FN)
   std::vector<double> f1_score;       // 2*precision*recall/(precision+recall)
   std::vector<double> accuracy;       // (TP+TN)/total
   std::vector<double> balanced_acc;   // (sensitivity + specificity)/2
   
   // Statistical tests
   std::vector<double> chi_square;     // Chi-square test statistic
   std::vector<double> z_score;        // Z-score for proportion test
   std::vector<double> p_value;        // P-value for statistical significance
   
   // Stability metrics
   std::vector<double> iv_component;   // IV component for each bin (renamed from psi_component)
   double total_psi;                   // Total PSI (actually total IV)
   
   // Concentration metrics
   std::vector<double> concentration_ratio;  // Concentration in each bin
   double herfindahl_index;                  // Sum of squared shares
   double normalized_herfindahl;             // Normalized HHI
   
   // Additional metrics
   std::vector<double> specificity;          // TN/(TN+FP) = 1 - cum_neg
   std::vector<double> miss_rate;            // FN/(FN+TP) = 1 - sensitivity
   std::vector<double> fall_out;             // FP/(FP+TN) = cum_neg
   std::vector<double> discovery_rate;       // FP/(FP+TP) = 1 - precision
   std::vector<double> mcc;                  // Matthews Correlation Coefficient
   
   // Summary statistics
   double base_rate;                         // Overall event rate
   double total_observations;                // Total number of observations
 };
 
 class GainsTableCalculator {
 private:
   double calculate_entropy(double p) {
     if (p <= 0 || p >= 1) return 0;
     return -p * std::log(p) - (1 - p) * std::log(1 - p);
   }
   
   double calculate_kl_divergence(double p, double q) {
     if (p <= 0 || p >= 1 || q <= 0 || q >= 1) return 0;
     if (std::abs(p - q) < 1e-10) return 0;
     return p * std::log(p / q) + (1 - p) * std::log((1 - p) / (1 - q));
   }
   
   double calculate_gini_impurity(double p) {
     return 1 - (p * p + (1 - p) * (1 - p));
   }
   
   double calculate_chi_square_stat(int obs_pos, int obs_neg, double exp_pos, double exp_neg) {
     double chi_sq = 0;
     if (exp_pos > 0.5) {
       chi_sq += std::pow(obs_pos - exp_pos, 2) / exp_pos;
     }
     if (exp_neg > 0.5) {
       chi_sq += std::pow(obs_neg - exp_neg, 2) / exp_neg;
     }
     return chi_sq;
   }
   
   double calculate_z_score(double p1, double p2, double n1, double n2) {
     if (n1 <= 0 || n2 <= 0) return 0;
     
     double p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2);
     if (p_pooled <= 0 || p_pooled >= 1) return 0;
     
     double se = std::sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2));
     if (se < 1e-10) return 0;
     
     return (p1 - p2) / se;
   }
   
   double calculate_p_value(double z_score) {
     double p = 2.0 * (1.0 - normal_cdf(std::abs(z_score)));
     return std::min(1.0, std::max(0.0, p));
   }
   
   double normal_cdf(double x) {
     return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
   }
   
   double calculate_mcc(double tp, double tn, double fp, double fn) {
     double numerator = tp * tn - fp * fn;
     double denominator = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
     if (denominator < 1e-10) return 0;
     return numerator / denominator;
   }
   
   double calculate_auc_trapezoidal(const std::vector<double>& tpr, const std::vector<double>& fpr) {
     double auc = 0;
     for (size_t i = 1; i < tpr.size(); ++i) {
       double dx = fpr[i] - fpr[i-1];
       double avg_y = (tpr[i] + tpr[i-1]) / 2.0;
       auc += avg_y * dx;
     }
     if (fpr.size() > 0 && fpr[0] > 0) {
       auc += 0.5 * tpr[0] * fpr[0];
     }
     size_t last = tpr.size() - 1;
     if (last < tpr.size() && fpr[last] < 1.0) {
       auc += (1.0 + tpr[last]) / 2.0 * (1.0 - fpr[last]);
     }
     return std::min(1.0, std::max(0.0, auc));
   }
   
 public:
   GainsMetrics calculate_gains_metrics(
       const std::vector<int>& bin_ids,
       const std::vector<std::string>& bin_labels,
       const std::vector<int>& total_counts,
       const std::vector<int>& neg_counts,
       const std::vector<int>& pos_counts,
       const std::vector<double>& woes,
       const std::vector<double>& ivs,
       bool sort_by_woe = false) {  // Default para false para preservar ordem dos bins
     
     GainsMetrics metrics;
     size_t n_bins = bin_ids.size();
     
     if (n_bins == 0) return metrics;
     
     // SEMPRE ordenar por bin_id para garantir integridade das métricas
     std::vector<size_t> indices(n_bins);
     std::iota(indices.begin(), indices.end(), 0);
     
     // Ordenar por bin_id (ignorando sort_by_woe para métricas cumulativas)
     std::sort(indices.begin(), indices.end(),
               [&bin_ids](size_t i, size_t j) { 
                 return bin_ids[i] < bin_ids[j]; 
               });
     
     // Calculate totals
     double total_obs = 0;
     double total_pos = 0;
     double total_neg = 0;
     
     for (size_t i = 0; i < n_bins; ++i) {
       total_obs += total_counts[i];
       total_pos += pos_counts[i];
       total_neg += neg_counts[i];
     }
     
     if (total_obs == 0) return metrics;
     
     double base_rate = total_pos / total_obs;
     
     // Initialize metrics
     metrics.base_rate = base_rate;
     metrics.total_observations = total_obs;
     
     // Process each bin in bin_id order
     double cum_obs = 0;
     double cum_pos = 0;
     double cum_neg = 0;
     
     // Vetores para AUC
     std::vector<double> fpr_points = {0.0};  // Começar em (0,0)
     std::vector<double> tpr_points = {0.0};
     
     for (size_t i = 0; i < n_bins; ++i) {
       size_t idx = indices[i];
       
       // Basic metrics
       metrics.bin_id.push_back(bin_ids[idx]);
       metrics.bin_label.push_back(bin_labels[idx]);
       metrics.total_count.push_back(total_counts[idx]);
       metrics.neg_count.push_back(neg_counts[idx]);
       metrics.pos_count.push_back(pos_counts[idx]);
       metrics.woe.push_back(woes[idx]);
       metrics.iv.push_back(ivs[idx]);
       
       // Update cumulative counts
       cum_obs += total_counts[idx];
       cum_pos += pos_counts[idx];
       cum_neg += neg_counts[idx];
       
       // Percentages
       double pct_total = total_counts[idx] / total_obs;
       double pct_pos = total_pos > 0 ? pos_counts[idx] / total_pos : 0;
       double pct_neg = total_neg > 0 ? neg_counts[idx] / total_neg : 0;
       double event_rate = total_counts[idx] > 0 ? 
       (double)pos_counts[idx] / total_counts[idx] : 0;
       
       metrics.pct_total.push_back(pct_total);  // Em percentual
       metrics.pct_pos.push_back(pct_pos);
       metrics.pct_neg.push_back(pct_neg);
       metrics.event_rate.push_back(event_rate);
       
       // Cumulative percentages
       double cum_pct_total = cum_obs / total_obs;
       double cum_pct_pos = total_pos > 0 ? cum_pos / total_pos : 0;  // TPR
       double cum_pct_neg = total_neg > 0 ? cum_neg / total_neg : 0;  // FPR
       
       metrics.cum_total.push_back(cum_pct_total);
       metrics.cum_pos.push_back(cum_pct_pos);
       metrics.cum_neg.push_back(cum_pct_neg);
       
       // Cumulative event rate
       double cum_event_rate = cum_obs > 0 ? cum_pos / cum_obs : 0;
       metrics.cum_event_rate.push_back(cum_event_rate);
       
       // Para AUC
       tpr_points.push_back(cum_pct_pos);
       fpr_points.push_back(cum_pct_neg);
       
       // Lift metrics
       double lift = base_rate > 0 ? event_rate / base_rate : 1.0;
       double cum_lift = base_rate > 0 ? cum_event_rate / base_rate : 1.0;
       
       metrics.lift.push_back(lift);
       metrics.cum_lift.push_back(cum_lift);
       metrics.capture_rate.push_back(cum_pct_pos);  // % de positivos capturados
       
       // KS
       double ks = std::abs(cum_pct_pos - cum_pct_neg);
       metrics.ks.push_back(ks);
       
       // Gini (impurity) para o bin
       double gini_impurity = calculate_gini_impurity(event_rate);
       metrics.gini.push_back(gini_impurity);
       
       // Entropy and Information
       double entropy = calculate_entropy(event_rate);
       double base_entropy = calculate_entropy(base_rate);
       double divergence = calculate_kl_divergence(event_rate, base_rate);
       double info_gain = base_entropy - entropy;
       
       metrics.entropy.push_back(entropy);
       metrics.divergence.push_back(divergence);
       metrics.info_gain.push_back(info_gain);
       
       // Risk metrics
       double odds = (event_rate > 1e-10 && event_rate < 1 - 1e-10) ? 
       event_rate / (1 - event_rate) : 
         (event_rate > 0.5 ? 1000.0 : 0.001);
       double log_odds = std::log(odds);
       double relative_risk = base_rate > 0 ? event_rate / base_rate : 1.0;
       
       metrics.odds.push_back(odds);
       metrics.log_odds.push_back(log_odds);
       metrics.relative_risk.push_back(relative_risk);
       
       // Performance metrics (CORRIGIDO)
       double tp = cum_pos;
       double fn = total_pos - cum_pos;
       double fp = cum_neg;
       double tn = total_neg - cum_neg;
       
       double sensitivity = total_pos > 0 ? tp / total_pos : 0;  // TPR
       double specificity = total_neg > 0 ? tn / total_neg : 0;  // TNR
       double ppv = (tp + fp) > 0 ? tp / (tp + fp) : 0;  // Precision
       double npv = (tn + fn) > 0 ? tn / (tn + fn) : 0;
       double f1 = (ppv + sensitivity) > 0 ? 
       2 * ppv * sensitivity / (ppv + sensitivity) : 0;
       double accuracy = (tp + tn) / total_obs;
       double balanced_acc = (sensitivity + specificity) / 2.0;
       
       metrics.precision.push_back(ppv);
       metrics.npv.push_back(npv);
       metrics.f1_score.push_back(f1);
       metrics.accuracy.push_back(accuracy);
       metrics.balanced_acc.push_back(balanced_acc);
       metrics.specificity.push_back(specificity);
       
       // Statistical tests
       double exp_pos = total_counts[idx] * base_rate;
       double exp_neg = total_counts[idx] * (1 - base_rate);
       double chi_sq = calculate_chi_square_stat(pos_counts[idx], neg_counts[idx], 
                                                 exp_pos, exp_neg);
       double z = calculate_z_score(event_rate, base_rate, 
                                    (double)total_counts[idx], 
                                                        total_obs - total_counts[idx]);
       double p_val = calculate_p_value(z);
       
       metrics.chi_square.push_back(chi_sq);
       metrics.z_score.push_back(z);
       metrics.p_value.push_back(p_val);
       
       // IV component (not PSI)
       double iv_comp = 0;
       if (pct_pos > 1e-10 && pct_neg > 1e-10) {
         iv_comp = (pct_pos - pct_neg) * std::log(pct_pos / pct_neg);
       }
       metrics.iv_component.push_back(iv_comp);
       
       // Concentration metrics
       metrics.concentration_ratio.push_back(pct_total);
       
       // Additional metrics
       double miss_rate = 1 - sensitivity;
       double fall_out = total_neg > 0 ? fp / total_neg : 0;  // FPR
       double fdr = (tp + fp) > 0 ? fp / (tp + fp) : 0;  // False Discovery Rate
       double mcc = calculate_mcc(tp, tn, fp, fn);
       
       metrics.miss_rate.push_back(miss_rate);
       metrics.fall_out.push_back(fall_out);
       metrics.discovery_rate.push_back(fdr);
       metrics.mcc.push_back(mcc);
     }
     
     // Calculate summary metrics
     metrics.max_ks = *std::max_element(metrics.ks.begin(), metrics.ks.end());
     metrics.auc = calculate_auc_trapezoidal(tpr_points, fpr_points);
     metrics.gini_coefficient = 2 * metrics.auc - 1;
     
     // Total IV (soma dos componentes)
     metrics.total_psi = std::accumulate(metrics.iv_component.begin(), 
                                         metrics.iv_component.end(), 0.0);
     
     // Herfindahl Index
     double hhi = 0;
     for (size_t i = 0; i < n_bins; ++i) {
       double share = total_counts[indices[i]] / total_obs;
       hhi += share * share;
     }
     metrics.herfindahl_index = hhi;
     metrics.normalized_herfindahl = n_bins > 1 ? 
     (hhi - 1.0/n_bins) / (1.0 - 1.0/n_bins) : 0;
     
     return metrics;
   }
 };
 
 // =====================================================================
 // SECTION 18: GAINS TABLE EXPORT FUNCTION
 // =====================================================================
 
 //' Gains Table Analysis for WoE Binning Results
 //'
 //' @description
 //' Function to compute comprehensive gains table metrics from WoE binning results. Calculates over 20 statistical
 //' measures for evaluating binning quality, predictive power, and model performance.
 //'
 //' The gains table provides detailed insights into how well each bin separates positive and negative cases,
 //' enabling assessment of binning effectiveness and feature predictive power.
 //'
 //' @details
 //' \strong{1) Core Metrics}
 //'
 //' \emph{Basic Counts and Rates}:
 //' \itemize{
 //'   \item \code{total_count}, \code{pos_count}, \code{neg_count}: Raw counts per bin
 //'   \item \code{event_rate}: Proportion of positive cases in each bin
 //'   \item \code{pct_total}: Percentage of total population in each bin
 //' }
 //'
 //' \emph{Cumulative Statistics}:
 //' \itemize{
 //'   \item \code{cum_pos}: Cumulative percentage of positive cases (True Positive Rate/Sensitivity)
 //'   \item \code{cum_neg}: Cumulative percentage of negative cases (False Positive Rate)
 //'   \item \code{cum_event_rate}: Cumulative event rate
 //' }
 //'
 //' \strong{2) Performance Metrics}
 //'
 //' \emph{Lift and Capture}:
 //' \deqn{\text{Lift} = \frac{\text{Event Rate}}{\text{Base Rate}}}
 //' \deqn{\text{Capture Rate} = \frac{\text{Cumulative Positives}}{\text{Total Positives}}}
 //' where Base Rate is the overall proportion of positive cases.
 //'
 //' \emph{KS Statistic}:
 //' \deqn{\text{KS} = \max|\text{Cumulative Positives} - \text{Cumulative Negatives}|}
 //' measuring the maximum separation between positive and negative distributions.
 //'
 //' \emph{ROC Analysis}:
 //' Area Under Curve (AUC) is calculated using the trapezoidal rule on the ROC curve:
 //' \deqn{\text{AUC} = \sum_{i=1}^{n} \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2} \times (\text{FPR}_{i-1} - \text{FPR}_i)}
 //'
 //' \emph{Classification Metrics}:
 //' \itemize{
 //'   \item \code{precision} (Positive Predictive Value): \eqn{\frac{TP}{TP + FP}}
 //'   \item \code{sensitivity} (True Positive Rate): \eqn{\frac{TP}{TP + FN}}
 //'   \item \code{specificity} (True Negative Rate): \eqn{\frac{TN}{TN + FP}}
 //'   \item \code{F1 Score}: \eqn{2 \times \frac{\text{precision} \times \text{sensitivity}}{\text{precision} + \text{sensitivity}}}
 //'   \item \code{Matthews Correlation Coefficient}: \eqn{\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}}
 //' }
 //'
 //' \strong{3) Information Theory Metrics}
 //'
 //' \emph{Entropy}:
 //' \deqn{H(p) = -p \log_2(p) - (1-p) \log_2(1-p)}
 //' where \eqn{p} is the event rate in the bin.
 //'
 //' \emph{Kullback-Leibler Divergence}:
 //' \deqn{D_{KL}(p||q) = p \log\left(\frac{p}{q}\right) + (1-p) \log\left(\frac{1-p}{1-q}\right)}
 //' measuring divergence from the base rate \eqn{q}.
 //'
 //' \emph{Information Value Component}:
 //' \deqn{\text{IV}_i = \left(\frac{\text{Count}_{1i}}{\text{Total}_1} - \frac{\text{Count}_{0i}}{\text{Total}_0}\right) \times \text{WoE}_i}
 //'
 //' \strong{4) Statistical Tests}
 //'
 //' \emph{Chi-Square Test} for each bin against the null hypothesis of independence.
 //'
 //' \emph{Z-Score} for testing if bin event rate differs significantly from base rate:
 //' \deqn{Z = \frac{p_1 - p_2}{\sqrt{p_{\text{pooled}}(1-p_{\text{pooled}})(\frac{1}{n_1} + \frac{1}{n_2})}}}
 //' where \eqn{p_{\text{pooled}} = \frac{p_1 n_1 + p_2 n_2}{n_1 + n_2}}.
 //'
 //' @param binning_result list output from \code{chimerge_woe}
 //' @param sort_by_woe Logical, whether to sort bins by WoE (default: FALSE)
 //' @param include_all_metrics Logical, whether to include all metrics or just essential ones (default: TRUE)
 //' @param selected_features Optional character vector of features to process (default: NULL)
 //' @param verbose Logical indicating whether to print detailed processing information (default: FALSE)
 //'
 //' @return
 //' list with one element per feature, each containing:
 //' \itemize{
 //'   \item \code{gains_table}: data.frame with detailed metrics for each bin
 //'   \item \code{summary_stats}: list with overall statistics
 //'   \item \code{variable_type}: String, variable type
 //'   \item \code{sorted_by}: String, sorting method used
 //' }
 //'
 //' The \code{gains_table} includes the following metrics:
 //' \itemize{
 //'   \item \strong{Basic Information}: bin_id, bin_label, counts, woe, iv
 //'   \item \strong{Percentages}: pct_total, pct_pos, pct_neg, event_rate
 //'   \item \strong{Cumulative Statistics}: cum_total, cum_pos, cum_neg, cum_event_rate
 //'   \item \strong{Lift Metrics}: lift, cum_lift, capture_rate
 //'   \item \strong{KS and Gini}: ks, gini
 //'   \item \strong{Entropy and Information}: entropy, divergence, info_gain
 //'   \item \strong{Risk Metrics}: odds, log_odds, relative_risk
 //'   \item \strong{Performance Metrics}: precision, npv, f1_score, accuracy, balanced_acc, specificity
 //'   \item \strong{Statistical Tests}: chi_square, z_score, p_value
 //'   \item \strong{Stability and Concentration}: psi_component, concentration_ratio
 //'   \item \strong{Additional Metrics}: miss_rate, fall_out, discovery_rate, mcc
 //' }
 //'
 //' The \code{summary_stats} includes:
 //' \itemize{
 //'   \item \code{base_rate}: Overall event rate
 //'   \item \code{total_observations}: Total number of observations
 //'   \item \code{max_ks}: Maximum KS statistic
 //'   \item \code{auc}: Area Under ROC Curve
 //'   \item \code{gini_coefficient}: Gini coefficient (2 * AUC - 1)
 //'   \item \code{total_psi}: Total Population Stability Index
 //'   \item \code{herfindahl_index}: Herfindahl-Hirschman Index
 //'   \item \code{normalized_herfindahl}: Normalized HHI
 //'   \item \code{total_iv}: Total Information Value
 //'   \item \code{n_bins}: Number of bins
 //' }
 //'
 //' @section Best practices for interpretation:
 //' \itemize{
 //'   \item \strong{KS Statistic}: Values > 0.4 indicate good separation power
 //'   \item \strong{AUC}: Values > 0.7 indicate acceptable discrimination
 //'   \item \strong{Lift}: Values > 2 in top deciles indicate good targeting
 //'   \item \strong{Capture Rate}: Should increase monotonically with cumulative percentage
 //'   \item \strong{IV Components}: Large positive/negative values indicate strong predictive bins
 //' }
 //'
 //' @seealso
 //' \itemize{
 //'   \item \code{chimerge_woe} for binning
 //'   \item \code{woe_gains_compare} for feature comparison
 //' }
 //'
 //' @references
 //' \itemize{
 //'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. Wiley.
 //'   \item Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation. Oxford University Press.
 //'   \item Hand, D. J., & Henley, W. E. (1997). Statistical Classification Methods in Consumer Credit. Journal of the Royal Statistical Society, Series A, 160(3), 523-541.
 //'   \item Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. Giornale dell'Istituto Italiano degli Attuari, 4, 83-91.
 //'   \item Smirnov, N. V. (1948). Table for Estimating the Goodness of Fit of Empirical Distributions. Annals of Mathematical Statistics, 19(2), 279-281.
 //' }
 //'
 //' @examples
 //' \dontrun{
 //' # Assuming binning_result from chimerge_woe
 //' gains_result <- woe_gains(binning_result, sort_by_woe = FALSE)
 //'
 //' # View gains table for first feature
 //' head(gains_result[[1]]$gains_table)
 //'
 //' # View summary statistics
 //' gains_result[[1]]$summary_stats
 //'
 //' # Compare features using essential metrics only
 //' simple_gains <- woe_gains(binning_result, include_all_metrics = FALSE)
 //' }
 //'
 //' @keywords gains-table woe iv ks-statistic auc roc
 //' @export
 // [[Rcpp::export]]
 List woe_gains(List binning_result, 
                bool sort_by_woe = false,  // Default false para manter ordem dos bins
                bool include_all_metrics = true,
                Nullable<CharacterVector> selected_features = R_NilValue,
                bool verbose = false) {  // Adicionado parâmetro verbose
   
   try {
     // Initialize logger with verbose setting
     chimerge_logging::global_logger.set_verbose(verbose);
     
     CHIMERGE_LOG_INFO("Starting WoE gains table calculation");
     CHIMERGE_TIMER("Gains table calculation");
     
     List gains_results;
     CharacterVector feature_names;
     
     // Determine which features to process
     if (selected_features.isNotNull()) {
       feature_names = selected_features.get();
     } else {
       feature_names = binning_result.names();
     }
     
     GainsTableCalculator calculator;
     
     // Process each feature
     for (int i = 0; i < feature_names.size(); ++i) {
       std::string feature = Rcpp::as<std::string>(feature_names[i]);
       
       CHIMERGE_LOG_INFO("Processing gains for feature: ", feature);
       
       if (!binning_result.containsElementNamed(feature.c_str())) {
         Rcpp::warning("Feature '" + feature + "' not found in binning results");
         CHIMERGE_LOG_WARNING("Feature '", feature, "' not found in binning results");
         continue;
       }
       
       List feature_result = binning_result[feature];
       
       if (feature_result.isNULL()) {
         CHIMERGE_LOG_WARNING("Feature '", feature, "' result is NULL");
         continue;
       }
       
       // Extract bins data
       DataFrame bins_df = as<DataFrame>(feature_result["bins"]);
       
       std::vector<int> bin_ids = as<std::vector<int>>(bins_df["bin_id"]);
       std::vector<std::string> bin_labels;
       CharacterVector labels = bins_df["bin_label"];
       for (int j = 0; j < labels.size(); ++j) {
         bin_labels.push_back(as<std::string>(labels[j]));
       }
       
       std::vector<int> total_counts = as<std::vector<int>>(bins_df["total_count"]);
       std::vector<int> neg_counts = as<std::vector<int>>(bins_df["neg_count"]);
       std::vector<int> pos_counts = as<std::vector<int>>(bins_df["pos_count"]);
       std::vector<double> woes = as<std::vector<double>>(bins_df["woe"]);
       std::vector<double> ivs = as<std::vector<double>>(bins_df["iv"]);
       
       // Calculate gains metrics (sempre ordenado por bin_id internamente)
       GainsMetrics metrics = calculator.calculate_gains_metrics(
         bin_ids, bin_labels, total_counts, neg_counts, pos_counts,
         woes, ivs, sort_by_woe
       );
       
       // Create output DataFrames
       DataFrame gains_df;
       
       if (include_all_metrics) {
         // Full gains table with all metrics
         gains_df = DataFrame::create(
           // Basic information
           Named("bin_id") = metrics.bin_id,
           Named("bin_label") = metrics.bin_label,
           Named("total_count") = metrics.total_count,
           Named("neg_count") = metrics.neg_count,
           Named("pos_count") = metrics.pos_count,
           Named("woe") = metrics.woe,
           Named("iv") = metrics.iv,
           
           // Percentages
           Named("pct_total") = metrics.pct_total,
           Named("pct_pos") = metrics.pct_pos,
           Named("pct_neg") = metrics.pct_neg,
           Named("event_rate") = metrics.event_rate,
           
           // Cumulative statistics
           Named("cum_total") = metrics.cum_total,
           Named("cum_pos") = metrics.cum_pos,
           Named("cum_neg") = metrics.cum_neg,
           Named("cum_event_rate") = metrics.cum_event_rate,
           
           // Lift metrics
           Named("lift") = metrics.lift,
           Named("cum_lift") = metrics.cum_lift,
           Named("capture_rate") = metrics.capture_rate,
           
           // KS and Gini
           Named("ks") = metrics.ks,
           Named("gini") = metrics.gini,
           
           // Entropy and Information
           Named("entropy") = metrics.entropy,
           Named("divergence") = metrics.divergence,
           Named("info_gain") = metrics.info_gain,
           
           // Risk metrics
           Named("odds") = metrics.odds,
           Named("log_odds") = metrics.log_odds,
           Named("relative_risk") = metrics.relative_risk,
           
           // Performance metrics
           Named("precision") = metrics.precision,
           Named("npv") = metrics.npv,
           Named("f1_score") = metrics.f1_score,
           Named("accuracy") = metrics.accuracy,
           Named("balanced_acc") = metrics.balanced_acc,
           Named("specificity") = metrics.specificity,
           
           // Statistical tests
           Named("chi_square") = metrics.chi_square,
           Named("z_score") = metrics.z_score,
           Named("p_value") = metrics.p_value,
           
           // Stability and concentration
           Named("iv_component") = metrics.iv_component,  // Renomeado de psi_component
           Named("concentration_ratio") = metrics.concentration_ratio,
           
           // Additional metrics
           Named("miss_rate") = metrics.miss_rate,
           Named("fall_out") = metrics.fall_out,
           Named("discovery_rate") = metrics.discovery_rate,
           Named("mcc") = metrics.mcc
         );
       } else {
         // Simplified gains table with essential metrics
         gains_df = DataFrame::create(
           Named("bin_id") = metrics.bin_id,
           Named("bin_label") = metrics.bin_label,
           Named("total_count") = metrics.total_count,
           Named("pos_count") = metrics.pos_count,
           Named("event_rate") = metrics.event_rate,
           Named("woe") = metrics.woe,
           Named("iv") = metrics.iv,
           Named("pct_total") = metrics.pct_total,
           Named("cum_pos") = metrics.cum_pos,
           Named("cum_neg") = metrics.cum_neg,
           Named("lift") = metrics.lift,
           Named("cum_lift") = metrics.cum_lift,
           Named("ks") = metrics.ks
         );
       }
       
       // Create summary statistics
       List summary_stats = List::create(
         Named("base_rate") = metrics.base_rate,
         Named("total_observations") = metrics.total_observations,
         Named("max_ks") = metrics.max_ks,
         Named("auc") = metrics.auc,
         Named("gini_coefficient") = metrics.gini_coefficient,
         Named("total_iv") = std::accumulate(metrics.iv.begin(), metrics.iv.end(), 0.0),
         Named("herfindahl_index") = metrics.herfindahl_index,
         Named("normalized_herfindahl") = metrics.normalized_herfindahl,
         Named("n_bins") = (int)metrics.bin_id.size()
       );
       
       // Store results for this feature
       List feature_gains = List::create(
         Named("gains_table") = gains_df,
         Named("summary_stats") = summary_stats,
         Named("variable_type") = feature_result["variable_type"],
                                                Named("sorted_by") = "bin_id"  // Sempre ordenado por bin_id
       );
       
       gains_results[feature] = feature_gains;
       CHIMERGE_LOG_INFO("Completed gains calculation for feature: ", feature);
     }
     
     CHIMERGE_LOG_INFO("WoE gains table calculation completed");
     return gains_results;
     
   } catch (const std::exception& e) {
     CHIMERGE_LOG_ERROR("Error in woe_gains: ", e.what());
     Rcpp::stop(std::string("Error in woe_gains: ") + e.what());
   }
 }
 
 
 
 
 
 
 //' Compare Features by Gains Table Metrics
 //'
 //' @description
 //' Function to rank and compare features based on various gains table metrics. Provides a consolidated
 //' view of feature performance for model selection and variable screening.
 //'
 //' @details
 //' This function aggregates summary statistics from gains tables and ranks features according to
 //' a selected performance metric. It's particularly useful for:
 //' \itemize{
 //'   \item Feature selection based on predictive power
 //'   \item Model validation and comparison
 //'   \item Monitoring feature stability over time
 //'   \item Business reporting and dashboard creation
 //' }
 //'
 //' The comparison includes several key metrics:
 //' \itemize{
 //'   \item \code{max_ks}: Maximum Kolmogorov-Smirnov statistic
 //'   \item \code{auc}: Area Under ROC Curve
 //'   \item \code{gini}: Gini coefficient (\eqn{2 \times} AUC \eqn{- 1})
 //'   \item \code{total_iv}: Total Information Value
 //'   \item \code{psi}: Population Stability Index
 //' }
 //'
 //' When the selected metric is "iv", the column containing the metric values is named "selected_metric"
 //' to avoid naming conflicts with the "total_iv" column.
 //'
 //' @param gains_results \code{list} output from \code{\link{woe_gains}}
 //' @param metric String, metric to use for ranking (default: "max_ks")
 //' @param descending Logical, whether to sort in descending order (default: TRUE)
 //'
 //' @return
 //' \code{data.frame} with features ranked by the selected metric:
 //' \itemize{
 //'   \item \code{rank}: Integer ranking position
 //'   \item \code{feature}: String feature name
 //'   \item \code{[metric]}: Double value of selected metric (named according to metric parameter)
 //'   \item \code{total_iv}: Double total Information Value
 //'   \item \code{max_ks}: Double maximum KS statistic
 //'   \item \code{auc}: Double Area Under Curve
 //'   \item \code{gini}: Double Gini coefficient
 //'   \item \code{psi}: Double Population Stability Index
 //'   \item \code{n_bins}: Integer number of bins
 //' }
 //'
 //' @section Metric Interpretation:
 //' \itemize{
 //'   \item \strong{max_ks}: Higher values indicate better separation (range: 0-1)
 //'   \item \strong{auc}: Higher values indicate better discrimination (range: 0.5-1)
 //'   \item \strong{gini}: Higher values indicate better ranking power (range: 0-1)
 //'   \item \strong{total_iv}: Higher values indicate stronger predictive power
 //'   \item \strong{psi}: Lower values indicate better stability (rule of thumb: \eqn{< 0.1} stable, 0.1-0.2 moderate, \eqn{> 0.2} unstable)
 //' }
 //'
 //' @seealso
 //' \itemize{
 //'   \item \code{\link{chimerge_woe}} for binning
 //'   \item \code{\link{woe_gains}} for detailed gains analysis
 //' }
 //'
 //' @references
 //' \itemize{
 //'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. Wiley.
 //'   \item Powers, D. M. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation. Journal of Machine Learning Technologies, 2(1), 37-63.
 //' }
 //'
 //' @examples
 //' \dontrun{
 //' # Assuming gains_result from woe_gains
 //' 
 //' # Rank by KS statistic (default)
 //' ks_ranking <- woe_gains_compare(gains_result)
 //' 
 //' # Rank by Information Value
 //' iv_ranking <- woe_gains_compare(gains_result, metric = "iv")
 //' 
 //' # Rank by AUC in ascending order
 //' auc_ranking <- woe_gains_compare(gains_result, metric = "auc", descending = FALSE)
 //' 
 //' # View top 10 features by IV
 //' head(iv_ranking, 10)
 //' }
 //'
 //' @keywords feature-selection model-validation ranking comparison
 //' @export
 // [[Rcpp::export]]
 DataFrame woe_gains_compare(List gains_results,
                             std::string metric = "max_ks",
                             bool descending = true) {
   try {
     CharacterVector feature_names = gains_results.names();
     
     std::vector<std::string> features;
     std::vector<double> metric_values;
     std::vector<double> total_ivs;
     std::vector<double> max_kss;
     std::vector<double> aucs;
     std::vector<double> ginis;
     std::vector<int> n_bins;
     
     for (int i = 0; i < feature_names.size(); ++i) {
       std::string feature = Rcpp::as<std::string>(feature_names[i]);
       List feature_gains = gains_results[feature];
       
       if (feature_gains.isNULL()) continue;
       
       List summary = feature_gains["summary_stats"];
       
       features.push_back(feature);
       total_ivs.push_back(as<double>(summary["total_iv"]));
       max_kss.push_back(as<double>(summary["max_ks"]));
       aucs.push_back(as<double>(summary["auc"]));
       ginis.push_back(as<double>(summary["gini_coefficient"]));
       n_bins.push_back(as<int>(summary["n_bins"]));
       
       // Select the requested metric for sorting
       if (metric == "max_ks") {
         metric_values.push_back(as<double>(summary["max_ks"]));
       } else if (metric == "auc") {
         metric_values.push_back(as<double>(summary["auc"]));
       } else if (metric == "gini") {
         metric_values.push_back(as<double>(summary["gini_coefficient"]));
       } else if (metric == "iv") {
         metric_values.push_back(as<double>(summary["total_iv"]));
       } else {
         // Default to max_ks if metric not recognized
         metric_values.push_back(as<double>(summary["max_ks"]));
       }
     }
     
     // Sort by metric
     std::vector<size_t> indices(features.size());
     std::iota(indices.begin(), indices.end(), 0);
     
     std::sort(indices.begin(), indices.end(),
               [&metric_values, descending](size_t i, size_t j) {
                 return descending ? metric_values[i] > metric_values[j] :
                 metric_values[i] < metric_values[j];
               });
     
     // Create sorted vectors
     std::vector<std::string> sorted_features;
     std::vector<double> sorted_metric_values;
     std::vector<double> sorted_ivs;
     std::vector<double> sorted_kss;
     std::vector<double> sorted_aucs;
     std::vector<double> sorted_ginis;
     std::vector<int> sorted_bins;
     
     for (size_t idx : indices) {
       sorted_features.push_back(features[idx]);
       sorted_metric_values.push_back(metric_values[idx]);
       sorted_ivs.push_back(total_ivs[idx]);
       sorted_kss.push_back(max_kss[idx]);
       sorted_aucs.push_back(aucs[idx]);
       sorted_ginis.push_back(ginis[idx]);
       sorted_bins.push_back(n_bins[idx]);
     }
     
     // Create ranking
     std::vector<int> rank(sorted_features.size());
     std::iota(rank.begin(), rank.end(), 1);
     
     // Build DataFrame with conditional columns to avoid duplication
     DataFrame comparison_df;
     
     // Always include rank and feature
     List df_columns;
     CharacterVector df_names;
     
     df_columns.push_back(rank);
     df_names.push_back("rank");
     
     df_columns.push_back(sorted_features);
     df_names.push_back("feature");
     
     // Add the sorted metric column (only if it's not already going to be added)
     if (metric != "iv" && metric != "max_ks" && metric != "auc" && metric != "gini") {
       // Unknown metric, add it with its name
       df_columns.push_back(sorted_metric_values);
       df_names.push_back(metric);
     }
     
     // Add standard metrics (avoiding duplication)
     if (metric != "iv") {
       df_columns.push_back(sorted_ivs);
       df_names.push_back("iv");
     } else {
       // If metric is "iv", add it as the primary sorted column
       df_columns.push_back(sorted_metric_values);
       df_names.push_back("iv");
     }
     
     if (metric != "max_ks") {
       df_columns.push_back(sorted_kss);
       df_names.push_back("max_ks");
     } else {
       // If metric is "max_ks", add it as the primary sorted column
       df_columns.push_back(sorted_metric_values);
       df_names.push_back("max_ks");
     }
     
     if (metric != "auc") {
       df_columns.push_back(sorted_aucs);
       df_names.push_back("auc");
     } else {
       // If metric is "auc", add it as the primary sorted column
       df_columns.push_back(sorted_metric_values);
       df_names.push_back("auc");
     }
     
     if (metric != "gini") {
       df_columns.push_back(sorted_ginis);
       df_names.push_back("gini");
     } else {
       // If metric is "gini", add it as the primary sorted column
       df_columns.push_back(sorted_metric_values);
       df_names.push_back("gini");
     }
     
     // Always add n_bins at the end
     df_columns.push_back(sorted_bins);
     df_names.push_back("n_bins");
     
     // Set names and create DataFrame
     df_columns.names() = df_names;
     comparison_df = DataFrame(df_columns);
     
     return comparison_df;
     
   } catch (const std::exception& e) {
     Rcpp::stop(std::string("Error in woe_gains_compare: ") + e.what());
   }
 }
 
 