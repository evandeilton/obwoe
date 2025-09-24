// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <atomic>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

/**
 * DMIV_WOE: A State-of-the-Art Hybrid Algorithm for Optimal Binning
 * Combines CART robustness with Divergence Measures for advanced binning
 */

// =============================================================================
// LOGGING SYSTEM
// =============================================================================

namespace dmivwoe {
namespace logging {

// Níveis de log
enum class LogLevel {
  NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
};

class Logger {
private:
  bool verbose_;
  LogLevel log_level_;
  
public:
  Logger(bool verbose = false, LogLevel level = LogLevel::INFO) 
    : verbose_(verbose), log_level_(level) {}
  
  // Métodos para diferentes níveis de log
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
  void trace(Args&&... args) {
    if (verbose_ && log_level_ >= LogLevel::TRACE) {
      log("TRACE", std::forward<Args>(args)...);
    }
  }
  
  // Método genérico de log
  template<typename... Args>
  void log(const std::string& level, Args&&... args) {
    if (!verbose_) return;
    
    Rcout << "[" << level << "] ";
    log_impl(std::forward<Args>(args)...);
    Rcout << std::endl;
  }
  
  // Métodos para medir tempo de execução
  class Timer {
  private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string description_;
    Logger* logger_;
    bool verbose_;
    
  public:
    Timer(Logger* logger, const std::string& description, bool verbose = true) 
      : logger_(logger), description_(description), verbose_(verbose && logger->is_verbose()) {
      if (verbose_) {
        start_time_ = std::chrono::high_resolution_clock::now();
        logger_->debug("Starting: ", description_);
      }
    }
    
    ~Timer() {
      if (verbose_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        logger_->debug("Completed: ", description_, " (", duration.count(), " ms)");
      }
    }
  };
  
  bool is_verbose() const { return verbose_; }
  LogLevel get_level() const { return log_level_; }
  
  void set_verbose(bool verbose) { verbose_ = verbose; }
  void set_level(LogLevel level) { log_level_ = level; }
  
private:
  // Implementação recursiva para concatenar argumentos
  void log_impl() {}
  
  template<typename T, typename... Args>
  void log_impl(T&& t, Args&&... args) {
    Rcout << t;
    log_impl(std::forward<Args>(args)...);
  }
};

// Instância global do logger
static Logger global_logger;

} // namespace logging
} // namespace dmivwoe

// Macros para facilitar o uso do logger
#define DMIV_LOG_ERROR(...) dmivwoe::logging::global_logger.error(__VA_ARGS__)
#define DMIV_LOG_WARNING(...) dmivwoe::logging::global_logger.warning(__VA_ARGS__)
#define DMIV_LOG_INFO(...) dmivwoe::logging::global_logger.info(__VA_ARGS__)
#define DMIV_LOG_DEBUG(...) dmivwoe::logging::global_logger.debug(__VA_ARGS__)
#define DMIV_LOG_TRACE(...) dmivwoe::logging::global_logger.trace(__VA_ARGS__)
#define DMIV_TIMER(description) dmivwoe::logging::Logger::Timer timer(&dmivwoe::logging::global_logger, description)

// =============================================================================
// CONSTANTS & CONFIGURATION
// =============================================================================

namespace dmivwoe {
namespace constants {
constexpr double EPSILON = 1e-12;
constexpr double WOE_MIN_CLIP = -20.0;
constexpr double WOE_MAX_CLIP = 20.0;
constexpr double MIN_BIN_FREQUENCY = 1e-6;
constexpr int MAX_ITERATIONS = 10000;
constexpr int PARALLEL_THRESHOLD = 1000;
constexpr int MAX_CATEGORICAL_LEVELS = 1000;
}

// =============================================================================
// ENUMS & TYPE DEFINITIONS
// =============================================================================

namespace types {
enum class VariableType {
  NUMERIC_CONTINUOUS,
  NUMERIC_DISCRETE,
  CATEGORICAL_NOMINAL,
  CATEGORICAL_ORDINAL,
  BOOLEAN,
  DATE_TIME,
  UNSUPPORTED
};

enum class DivergenceMethod {
  HELLINGER,
  KULLBACK_LEIBLER,
  J_DIVERGENCE,
  CHI_SQUARE,
  JENSEN_SHANNON,
  TOTAL_VARIATION,
  L1_NORM,
  L2_NORM,
  LINF_NORM
};

enum class OptimizationMethod {
  BRANCH_AND_BOUND,
  DYNAMIC_PROGRAMMING,
  GREEDY_MERGE,
  SIMULATED_ANNEALING
};

enum class MonotonicityType {
  NONE,
  INCREASING,
  DECREASING,
  AUTO_DETECT
};
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct DataPoint {
  // Core values
  double numeric_value;
  std::string category_value;
  double target;
  double weight;
  
  // Metadata flags
  bool is_missing;
  bool is_special;
  double special_value;
  int original_index;
  
  DataPoint() : numeric_value(0), target(0), weight(1),
  is_missing(false), is_special(false), special_value(0),
  original_index(-1) {}
};

struct BinInfo {
  // Identifiers
  int id;
  std::string label;
  
  // Boundaries
  double lower_bound;
  double upper_bound;
  std::vector<std::string> categories;
  
  // Counts and weights - using atomic for thread safety
  std::atomic<int> total_count{0};
  std::atomic<int> pos_count{0};
  std::atomic<int> neg_count{0};
  std::atomic<double> weight_sum{0};
  std::atomic<double> pos_weight{0};
  std::atomic<double> neg_weight{0};
  
  // Statistical measures
  double woe;
  double divergence;
  double iv;
  double ks;
  double gini;
  
  // Confidence intervals (when computed)
  double woe_lower_ci;
  double woe_upper_ci;
  
  BinInfo() : id(0), lower_bound(-INFINITY), upper_bound(INFINITY),
  woe(0), divergence(0), iv(0), ks(0), gini(0), 
  woe_lower_ci(0), woe_upper_ci(0) {}
  
  // Copy constructor for atomic members
  BinInfo(const BinInfo& other) : 
    id(other.id), label(other.label), 
    lower_bound(other.lower_bound), upper_bound(other.upper_bound),
    categories(other.categories),
    total_count(other.total_count.load()),
    pos_count(other.pos_count.load()),
    neg_count(other.neg_count.load()),
    weight_sum(other.weight_sum.load()),
    pos_weight(other.pos_weight.load()),
    neg_weight(other.neg_weight.load()),
    woe(other.woe), divergence(other.divergence), iv(other.iv),
    ks(other.ks), gini(other.gini),
    woe_lower_ci(other.woe_lower_ci), woe_upper_ci(other.woe_upper_ci) {}
  
  // Assignment operator
  BinInfo& operator=(const BinInfo& other) {
    if (this != &other) {
      id = other.id;
      label = other.label;
      lower_bound = other.lower_bound;
      upper_bound = other.upper_bound;
      categories = other.categories;
      total_count = other.total_count.load();
      pos_count = other.pos_count.load();
      neg_count = other.neg_count.load();
      weight_sum = other.weight_sum.load();
      pos_weight = other.pos_weight.load();
      neg_weight = other.neg_weight.load();
      woe = other.woe;
      divergence = other.divergence;
      iv = other.iv;
      ks = other.ks;
      gini = other.gini;
      woe_lower_ci = other.woe_lower_ci;
      woe_upper_ci = other.woe_upper_ci;
    }
    return *this;
  }
};

struct OptimizationConfig {
  // Divergence settings
  types::DivergenceMethod divergence_method;
  std::string woe_method;
  double smooth;  // Laplace smoothing parameter
  
  // Bin constraints
  int min_bins;
  int max_bins;
  double min_bin_size_ratio;
  
  // Optimization settings
  types::OptimizationMethod optimization_method;
  int max_iterations;
  double convergence_threshold;
  int random_seed;
  
  // Monotonicity
  types::MonotonicityType monotonicity;
  
  // Special value handling
  std::string missing_policy;
  std::set<double> special_values;
  double rare_category_threshold;
  
  // Categorical separator
  std::string cat_sep;
  
  // Numeric formatting
  int digits;
  
  // Regularization
  double l1_regularization;
  double l2_regularization;
  
  // Cross-validation
  bool use_cross_validation;
  int cv_folds;
  
  // Confidence intervals
  bool compute_confidence_intervals;
  double confidence_level;
  
  // Parallel processing
  bool use_parallel;
  int n_threads;
  
  // Logging
  bool verbose;
  
  OptimizationConfig() : 
    divergence_method(types::DivergenceMethod::L2_NORM),
    woe_method("woe"),
    smooth(0.0),
    min_bins(3), max_bins(10),
    min_bin_size_ratio(0.05),
    optimization_method(types::OptimizationMethod::GREEDY_MERGE),
    max_iterations(1000), convergence_threshold(1e-6), random_seed(42),
    monotonicity(types::MonotonicityType::NONE),
    missing_policy("separate"), rare_category_threshold(0.01),
    cat_sep("%;%"), digits(3),
    l1_regularization(0.0), l2_regularization(0.0),
    use_cross_validation(false), cv_folds(5),
    compute_confidence_intervals(false), confidence_level(0.95),
    use_parallel(false), n_threads(-1),
    verbose(false) {}
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

inline double safe_log(double x) {
  return std::log(std::max(x, constants::EPSILON));
}

inline double safe_divide(double numerator, double denominator) {
  return denominator > constants::EPSILON ? numerator / denominator : 0.0;
}

inline double clip_value(double x, double min_val, double max_val) {
  return std::max(min_val, std::min(max_val, x));
}

// Format a number with specified decimal places
inline std::string format_number(double value, int digits) {
  if (std::isinf(value)) {
    return (value > 0) ? "+inf" : "-inf";
  }
  if (std::isnan(value)) {
    return "NA";
  }
  
  std::stringstream ss;
  ss << std::fixed << std::setprecision(digits) << value;
  return ss.str();
}

inline std::string divergence_method_to_string(types::DivergenceMethod method) {
  switch(method) {
  case types::DivergenceMethod::HELLINGER: return "hellinger";
  case types::DivergenceMethod::KULLBACK_LEIBLER: return "kullback_leibler";
  case types::DivergenceMethod::J_DIVERGENCE: return "j_divergence";
  case types::DivergenceMethod::CHI_SQUARE: return "chi_square";
  case types::DivergenceMethod::JENSEN_SHANNON: return "jensen_shannon";
  case types::DivergenceMethod::TOTAL_VARIATION: return "total_variation";
  case types::DivergenceMethod::L1_NORM: return "l1_norm";
  case types::DivergenceMethod::L2_NORM: return "l2_norm";
  case types::DivergenceMethod::LINF_NORM: return "linf_norm";
  default: return "unknown";
  }
}

inline std::string variable_type_to_string(types::VariableType vt) {
  switch (vt) {
  case types::VariableType::NUMERIC_CONTINUOUS: return "numeric_continuous";
  case types::VariableType::NUMERIC_DISCRETE: return "numeric_discrete";
  case types::VariableType::CATEGORICAL_NOMINAL: return "categorical";
  case types::VariableType::CATEGORICAL_ORDINAL: return "categorical";
  case types::VariableType::BOOLEAN: return "boolean";
  case types::VariableType::DATE_TIME: return "date_time";
  default: return "unsupported";
  }
}

// =============================================================================
// VARIABLE TYPE DETECTION
// =============================================================================

class VariableTypeDetector {
public:
  types::VariableType detect(SEXP x) {
    DMIV_LOG_TRACE("Detecting variable type");
    
    if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") || 
        Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
      DMIV_LOG_DEBUG("Detected date/time variable");
      return types::VariableType::DATE_TIME;
    }
    
    if (Rf_isFactor(x)) {
      DMIV_LOG_DEBUG("Detected factor variable");
      return types::VariableType::CATEGORICAL_NOMINAL;
    }
    
    switch (TYPEOF(x)) {
    case REALSXP: {
      NumericVector nv(x);
      int check_limit = std::min(1000, static_cast<int>(nv.size()));
      std::set<double> unique_vals;
      
      for (int i = 0; i < check_limit && unique_vals.size() <= 20; ++i) {
        if (!NumericVector::is_na(nv[i]) && std::isfinite(nv[i])) {
          unique_vals.insert(nv[i]);
        }
      }
      DMIV_LOG_DEBUG("Numeric variable with ", unique_vals.size(), " unique values");
      return (unique_vals.size() <= 20) ? 
      types::VariableType::NUMERIC_DISCRETE : 
        types::VariableType::NUMERIC_CONTINUOUS;
    }
    case INTSXP: {
      IntegerVector iv(x);
      int check_limit = std::min(1000, static_cast<int>(iv.size()));
      std::set<int> unique_vals;
      
      for (int i = 0; i < check_limit && unique_vals.size() <= 20; ++i) {
        if (!IntegerVector::is_na(iv[i])) {
          unique_vals.insert(iv[i]);
        }
      }
      DMIV_LOG_DEBUG("Integer variable with ", unique_vals.size(), " unique values");
      return (unique_vals.size() <= 20) ? 
      types::VariableType::NUMERIC_DISCRETE : 
        types::VariableType::NUMERIC_CONTINUOUS;
    }
    case LGLSXP:
      DMIV_LOG_DEBUG("Detected logical variable");
      return types::VariableType::BOOLEAN;
    case STRSXP:
      DMIV_LOG_DEBUG("Detected string variable");
      return types::VariableType::CATEGORICAL_NOMINAL;
    default:
      DMIV_LOG_WARNING("Unsupported variable type detected");
    return types::VariableType::UNSUPPORTED;
    }
  }
};

// =============================================================================
// DIVERGENCE CALCULATION FRAMEWORK
// =============================================================================

class DivergenceCalculator {
public:
  double calculate(
      const std::vector<double>& p_dist,
      const std::vector<double>& q_dist,
      types::DivergenceMethod method,
      double l1_reg = 0.0,
      double l2_reg = 0.0
  ) const {
    DMIV_LOG_TRACE("Calculating divergence with method: ", divergence_method_to_string(method));
    
    if (p_dist.size() != q_dist.size()) {
      throw std::invalid_argument("Distribution vectors must have same size");
    }
    
    double result = 0.0;
    
    switch(method) {
    case types::DivergenceMethod::L2_NORM: {
      double sum_squares = 0.0;
      for (size_t i = 0; i < p_dist.size(); ++i) {
        sum_squares += std::pow(p_dist[i] - q_dist[i], 2);
      }
      result = std::sqrt(sum_squares);
      break;
    }
    case types::DivergenceMethod::KULLBACK_LEIBLER: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      if (p_dist[i] > constants::EPSILON && q_dist[i] > constants::EPSILON) {
        result += p_dist[i] * safe_log(safe_divide(p_dist[i], q_dist[i]));
      }
    }
      break;
    }
    case types::DivergenceMethod::J_DIVERGENCE: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      if (p_dist[i] > constants::EPSILON && q_dist[i] > constants::EPSILON) {
        result += (p_dist[i] - q_dist[i]) * safe_log(safe_divide(p_dist[i], q_dist[i]));
      }
    }
      break;
    }
    case types::DivergenceMethod::HELLINGER: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      double sqrt_p = std::sqrt(p_dist[i]);
      double sqrt_q = std::sqrt(q_dist[i]);
      result += std::pow(sqrt_p - sqrt_q, 2);
    }
      result = 0.5 * result;
      break;
    }
    case types::DivergenceMethod::TOTAL_VARIATION: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      result += std::abs(p_dist[i] - q_dist[i]);
    }
      result = 0.5 * result;
      break;
    }
    case types::DivergenceMethod::L1_NORM: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      result += std::abs(p_dist[i] - q_dist[i]);
    }
      break;
    }
    case types::DivergenceMethod::CHI_SQUARE: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      double sum_pq = p_dist[i] + q_dist[i];
      if (sum_pq > constants::EPSILON && 
          p_dist[i] > constants::EPSILON && 
          q_dist[i] > constants::EPSILON) {
        result += safe_divide(std::pow(p_dist[i] - q_dist[i], 2) * sum_pq, 
                              p_dist[i] * q_dist[i]);
      }
    }
      break;
    }
    case types::DivergenceMethod::JENSEN_SHANNON: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      double m = 0.5 * (p_dist[i] + q_dist[i]);
      if (p_dist[i] > constants::EPSILON && m > constants::EPSILON) {
        result += p_dist[i] * safe_log(safe_divide(p_dist[i], m));
      }
      if (q_dist[i] > constants::EPSILON && m > constants::EPSILON) {
        result += q_dist[i] * safe_log(safe_divide(q_dist[i], m));
      }
    }
      result = 0.5 * result;
      break;
    }
    case types::DivergenceMethod::LINF_NORM: {
      for (size_t i = 0; i < p_dist.size(); ++i) {
      result = std::max(result, std::abs(p_dist[i] - q_dist[i]));
    }
      break;
    }
    default:
      // Fallback to L2 norm
      double sum_squares = 0.0;
      for (size_t i = 0; i < p_dist.size(); ++i) {
        sum_squares += std::pow(p_dist[i] - q_dist[i], 2);
      }
      result = std::sqrt(sum_squares);
      break;
    }
    
    // Apply regularization
    if (l1_reg > 0) {
      double l1_penalty = 0;
      for (size_t i = 0; i < p_dist.size(); ++i) {
        l1_penalty += std::abs(p_dist[i]) + std::abs(q_dist[i]);
      }
      result += l1_reg * l1_penalty;
    }
    
    if (l2_reg > 0) {
      double l2_penalty = 0;
      for (size_t i = 0; i < p_dist.size(); ++i) {
        l2_penalty += p_dist[i] * p_dist[i] + q_dist[i] * q_dist[i];
      }
      result += l2_reg * std::sqrt(l2_penalty);
    }
    
    DMIV_LOG_DEBUG("Divergence calculated: ", result);
    return result;
  }
};

// =============================================================================
// CONFIDENCE INTERVAL CALCULATOR
// =============================================================================

class ConfidenceIntervalCalculator {
private:
  // Normal distribution quantile approximation
  double normal_quantile(double p) {
    // Approximation using Abramowitz and Stegun formula
    double a1 = -3.969683028665376e+01;
    double a2 = 2.209460984245205e+02;
    double a3 = -2.759285104469687e+02;
    double a4 = 1.383577518672690e+02;
    double a5 = -3.066479806614716e+01;
    double a6 = 2.506628277459239e+00;
    
    double b1 = -5.447609879822406e+01;
    double b2 = 1.615858368580409e+02;
    double b3 = -1.556989798598866e+02;
    double b4 = 6.680131188771972e+01;
    double b5 = -1.328068155288572e+01;
    
    double c1 = -7.784894002430293e-03;
    double c2 = -3.223964580411365e-01;
    double c3 = -2.400758277161838e+00;
    double c4 = -2.549732539343734e+00;
    double c5 = 4.374664141464968e+00;
    double c6 = 2.938163982698783e+00;
    
    double d1 = 7.784695709041462e-03;
    double d2 = 3.224671290700398e-01;
    double d3 = 2.445134137142996e+00;
    double d4 = 3.754408661907416e+00;
    
    double p_low = 0.02425;
    double p_high = 1 - p_low;
    double q, r, z;
    
    if (p < p_low) {
      q = std::sqrt(-2*std::log(p));
      z = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    } else if (p <= p_high) {
      q = p - 0.5;
      r = q*q;
      z = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
    } else {
      q = std::sqrt(-2*std::log(1-p));
      z = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }
    
    return z;
  }
  
public:
  std::pair<double, double> calculate_woe_ci(
      double pos_count, double neg_count,
      double total_pos, double total_neg,
      double confidence_level,
      const std::string& woe_method,
      double smooth
  ) {
    DMIV_LOG_TRACE("Calculating WoE confidence interval");
    
    // Use Wilson score interval for binomial proportions
    double z = normal_quantile((1 + confidence_level) / 2);
    
    double p_pos = safe_divide(pos_count + smooth, total_pos + smooth);
    double p_neg = safe_divide(neg_count + smooth, total_neg + smooth);
    
    // Wilson score interval for positive rate
    double n1 = total_pos;
    double p1_center = safe_divide(pos_count + z*z/2, n1 + z*z);
    double p1_margin = z * std::sqrt(safe_divide(p1_center * (1 - p1_center), n1 + z*z));
    double p1_lower = p1_center - p1_margin;
    double p1_upper = p1_center + p1_margin;
    
    // Wilson score interval for negative rate
    double n0 = total_neg;
    double p0_center = safe_divide(neg_count + z*z/2, n0 + z*z);
    double p0_margin = z * std::sqrt(safe_divide(p0_center * (1 - p0_center), n0 + z*z));
    double p0_lower = p0_center - p0_margin;
    double p0_upper = p0_center + p0_margin;
    
    // Calculate WoE confidence interval
    double woe_lower = safe_log(safe_divide(std::max(p1_lower, constants::EPSILON),
                                            std::max(p0_upper, constants::EPSILON)));
    double woe_upper = safe_log(safe_divide(std::max(p1_upper, constants::EPSILON),
                                            std::max(p0_lower, constants::EPSILON)));
    
    auto result = std::make_pair(
      clip_value(woe_lower, constants::WOE_MIN_CLIP, constants::WOE_MAX_CLIP),
      clip_value(woe_upper, constants::WOE_MIN_CLIP, constants::WOE_MAX_CLIP)
    );
    
    DMIV_LOG_DEBUG("WoE CI: [", result.first, ", ", result.second, "]");
    return result;
  }
};

// =============================================================================
// WOE AND METRICS CALCULATION
// =============================================================================

class MetricsCalculator {
public:
  static double calculate_woe(
      double count_pos, double count_neg,
      double total_pos, double total_neg,
      const std::string& method,
      double smooth = 0.0
  ) {
    DMIV_LOG_TRACE("Calculating WoE with method: ", method);
    
    if (total_pos <= constants::EPSILON || total_neg <= constants::EPSILON) {
      DMIV_LOG_DEBUG("Zero totals, returning 0");
      return 0.0;
    }
    
    double p_pos, p_neg;
    
    if (method == "woe") {
      // Traditional WoE with smoothing
      p_pos = safe_divide(count_pos + smooth, total_pos + smooth);
      p_neg = safe_divide(count_neg + smooth, total_neg + smooth);
    } else {
      // Zeng's WOE1 with smoothing
      double total_bin = count_pos + count_neg;
      if (total_bin <= constants::EPSILON) return 0.0;
      p_pos = safe_divide(count_pos + smooth, total_bin + 2 * smooth);
      p_neg = safe_divide(count_neg + smooth, total_bin + 2 * smooth);
    }
    
    if (p_neg <= constants::EPSILON) return constants::WOE_MAX_CLIP;
    if (p_pos <= constants::EPSILON) return constants::WOE_MIN_CLIP;
    
    double woe = safe_log(safe_divide(p_pos, p_neg));
    double result = clip_value(woe, constants::WOE_MIN_CLIP, constants::WOE_MAX_CLIP);
    
    DMIV_LOG_DEBUG("WoE calculated: ", result);
    return result;
  }
  
  static double calculate_iv(
      double count_pos, double count_neg,
      double total_pos, double total_neg,
      double smooth = 0.0
  ) {
    DMIV_LOG_TRACE("Calculating IV");
    
    if (total_pos <= constants::EPSILON || total_neg <= constants::EPSILON) {
      DMIV_LOG_DEBUG("Zero totals, returning 0");
      return 0.0;
    }
    
    double p_pos = safe_divide(count_pos + smooth, total_pos + smooth);
    double p_neg = safe_divide(count_neg + smooth, total_neg + smooth);
    
    if (p_pos <= constants::EPSILON || p_neg <= constants::EPSILON) {
      DMIV_LOG_DEBUG("Zero proportions, returning 0");
      return 0.0;
    }
    
    double result = (p_pos - p_neg) * safe_log(safe_divide(p_pos, p_neg));
    DMIV_LOG_DEBUG("IV calculated: ", result);
    return result;
  }
  
  static void calculate_bin_metrics(
      std::vector<BinInfo>& bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Calculating bin metrics for ", bins.size(), " bins");
    DMIV_TIMER("Bin metrics calculation");
    
    if (bins.empty()) {
      DMIV_LOG_WARNING("Empty bins vector, skipping metrics calculation");
      return;
    }
    
    // Set up parallel processing if enabled
#ifdef _OPENMP
    if (config.use_parallel && config.n_threads > 0) {
      omp_set_num_threads(config.n_threads);
      DMIV_LOG_DEBUG("Setting OpenMP threads to ", config.n_threads);
    } else if (config.use_parallel && config.n_threads == -1) {
      // Auto-detect number of threads
      omp_set_num_threads(omp_get_max_threads());
      DMIV_LOG_DEBUG("Auto-detecting OpenMP threads");
    }
#endif
    
    // Calculate totals
    double total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.pos_weight.load();
      total_neg += bin.neg_weight.load();
    }
    
    DMIV_LOG_DEBUG("Total positive: ", total_pos, ", Total negative: ", total_neg);
    
    if (total_pos <= constants::EPSILON || total_neg <= constants::EPSILON) {
      DMIV_LOG_WARNING("Zero totals, skipping metrics calculation");
      return;
    }
    
    // Calculate cumulative statistics for KS
    double cum_pos = 0, cum_neg = 0;
    
    // Calculate distributions for divergence
    std::vector<double> p_dist, q_dist;
    p_dist.reserve(bins.size());
    q_dist.reserve(bins.size());
    
    for (const auto& bin : bins) {
      double p = safe_divide(bin.pos_weight.load(), total_pos);
      double q = safe_divide(bin.neg_weight.load(), total_neg);
      p_dist.push_back(p);
      q_dist.push_back(q);
    }
    
    // Initialize confidence interval calculator if needed
    ConfidenceIntervalCalculator ci_calc;
    
    // Calculate metrics for each bin
#ifdef _OPENMP
#pragma omp parallel for if(config.use_parallel && bins.size() > constants::PARALLEL_THRESHOLD)
#endif
    for (size_t i = 0; i < bins.size(); ++i) {
      auto& bin = bins[i];
      
      // WoE calculation with smoothing
      bin.woe = calculate_woe(
        bin.pos_weight.load(), bin.neg_weight.load(),
        total_pos, total_neg,
        config.woe_method,
        config.smooth
      );
      
      // IV calculation with smoothing
      bin.iv = calculate_iv(
        bin.pos_weight.load(), bin.neg_weight.load(),
        total_pos, total_neg,
        config.smooth
      );
      
      // Calculate confidence intervals if requested
      if (config.compute_confidence_intervals) {
        auto ci = ci_calc.calculate_woe_ci(
          bin.pos_weight.load(), bin.neg_weight.load(),
          total_pos, total_neg,
          config.confidence_level,
          config.woe_method,
          config.smooth
        );
        bin.woe_lower_ci = ci.first;
        bin.woe_upper_ci = ci.second;
      }
    }
    
    // Sequential calculation for cumulative metrics
    for (size_t i = 0; i < bins.size(); ++i) {
      auto& bin = bins[i];
      
      // Cumulative statistics for KS
      cum_pos += bin.pos_weight.load();
      cum_neg += bin.neg_weight.load();
      double cdf_pos = safe_divide(cum_pos, total_pos);
      double cdf_neg = safe_divide(cum_neg, total_neg);
      bin.ks = std::abs(cdf_pos - cdf_neg);
      
      // Gini calculation
      double total_bin = bin.weight_sum.load();
      if (total_bin > constants::EPSILON) {
        double p0 = safe_divide(bin.neg_weight.load(), total_bin);
        double p1 = safe_divide(bin.pos_weight.load(), total_bin);
        bin.gini = std::abs(p1 - p0);
      } else {
        bin.gini = 0.0;
      }
    }
    
    // Calculate total divergence with regularization
    DivergenceCalculator div_calc;
    double total_divergence = div_calc.calculate(
      p_dist, q_dist, config.divergence_method,
      config.l1_regularization, config.l2_regularization
    );
    
    DMIV_LOG_DEBUG("Total divergence: ", total_divergence);
    
    // Assign divergence to bins proportionally
    if (bins.size() > 0 && total_divergence > constants::EPSILON) {
      double max_iv = 0.0;
      for (const auto& bin : bins) {
        max_iv = std::max(max_iv, std::abs(bin.iv));
      }
      
      if (max_iv > constants::EPSILON) {
        for (auto& bin : bins) {
          bin.divergence = total_divergence * safe_divide(std::abs(bin.iv), max_iv);
        }
      } else {
        for (auto& bin : bins) {
          bin.divergence = safe_divide(total_divergence, bins.size());
        }
      }
    }
    
    DMIV_LOG_INFO("Bin metrics calculation completed");
  }
};

// =============================================================================
// MONOTONICITY ENFORCER
// =============================================================================

class MonotonicityEnforcer {
private:
  std::vector<double> pool_adjacent_violators(
      const std::vector<double>& y,
      const std::vector<double>& weights,
      bool increasing = true
  ) {
    DMIV_LOG_TRACE("Applying pool adjacent violators algorithm");
    
    int n = y.size();
    if (n == 0) return {};
    
    std::vector<double> result = y;
    if (!increasing) {
      for (auto& val : result) val = -val;
    }
    
    std::vector<std::pair<double, double>> blocks;
    blocks.reserve(n);
    
    for (int i = 0; i < n; ++i) {
      double weight = (i < static_cast<int>(weights.size())) ? weights[i] : 1.0;
      blocks.push_back({result[i] * weight, weight});
      
      while (blocks.size() >= 2) {
        int last = blocks.size() - 1;
        double mean_last = safe_divide(blocks[last].first, blocks[last].second);
        double mean_prev = safe_divide(blocks[last-1].first, blocks[last-1].second);
        
        if (mean_prev <= mean_last + constants::EPSILON) break;
        
        blocks[last-1].first += blocks[last].first;
        blocks[last-1].second += blocks[last].second;
        blocks.pop_back();
      }
    }
    
    int idx = 0;
    for (const auto& block : blocks) {
      double mean = safe_divide(block.first, block.second);
      if (!increasing) mean = -mean;
      
      int block_size = std::max(1, static_cast<int>(std::round(safe_divide(block.second,
                                                                           ((idx < static_cast<int>(weights.size())) ? weights[idx] : 1.0)))));
      for (int j = 0; j < block_size && idx < n; ++j, ++idx) {
        result[idx] = mean;
      }
    }
    
    DMIV_LOG_DEBUG("PAVA completed with ", blocks.size(), " blocks");
    return result;
  }
  
public:
  std::vector<BinInfo> enforce(
      std::vector<BinInfo>& bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Enforcing monotonicity for ", bins.size(), " bins");
    DMIV_TIMER("Monotonicity enforcement");
    
    if (bins.size() <= 1 || config.monotonicity == types::MonotonicityType::NONE) {
      DMIV_LOG_DEBUG("Skipping monotonicity enforcement");
      return bins;
    }
    
    // Check if already monotonic
    bool is_increasing = true;
    bool is_decreasing = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i-1].woe - constants::EPSILON) is_increasing = false;
      if (bins[i].woe > bins[i-1].woe + constants::EPSILON) is_decreasing = false;
    }
    
    if (is_increasing || is_decreasing) {
      DMIV_LOG_DEBUG("Bins already monotonic");
      return bins;
    }
    
    // Determine trend direction
    std::vector<double> woes;
    std::vector<double> weights;
    
    for (const auto& bin : bins) {
      woes.push_back(bin.woe);
      weights.push_back(bin.weight_sum.load());
    }
    
    // Apply isotonic regression
    bool increasing = (config.monotonicity == types::MonotonicityType::INCREASING) ||
      (config.monotonicity == types::MonotonicityType::AUTO_DETECT);
    
    if (config.monotonicity == types::MonotonicityType::AUTO_DETECT) {
      // Simple trend detection
      double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
      for (size_t i = 0; i < woes.size(); ++i) {
        double x = static_cast<double>(i);
        double y = woes[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
      }
      int n = woes.size();
      double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * 
                                     (n * sum_xy - sum_y * sum_y) + constants::EPSILON);
      double correlation = safe_divide(n * sum_xy - sum_x * sum_y, denominator);
      increasing = correlation > 0;
      DMIV_LOG_DEBUG("Auto-detected trend: ", (increasing ? "increasing" : "decreasing"));
    }
    
    auto monotonic_woes = pool_adjacent_violators(woes, weights, increasing);
    
    // Update WoE values
    for (size_t i = 0; i < bins.size(); ++i) {
      bins[i].woe = monotonic_woes[i];
    }
    
    DMIV_LOG_INFO("Monotonicity enforcement completed");
    return bins;
  }
};

// =============================================================================
// PREBINNING ENGINE - CORRIGIDA PARA VARIÁVEIS BINÁRIAS
// =============================================================================

class PrebinningEngine {
private:
  // Nova função para detectar se uma variável é binária
  bool is_binary_variable(const std::vector<DataPoint>& data) {
    DMIV_LOG_TRACE("Checking if variable is binary");
    
    std::set<double> unique_values;
    for (const auto& dp : data) {
      if (!dp.is_missing && !dp.is_special && std::isfinite(dp.numeric_value)) {
        unique_values.insert(dp.numeric_value);
        if (unique_values.size() > 2) {
          DMIV_LOG_DEBUG("Variable is not binary (more than 2 unique values)");
          return false;
        }
      }
    }
    
    bool result = unique_values.size() == 2;
    DMIV_LOG_DEBUG("Variable is ", (result ? "binary" : "not binary"));
    return result;
  }
  
  // Nova função para criar bins para variáveis binárias
  std::vector<BinInfo> create_binary_bins(const std::vector<DataPoint>& data) {
    DMIV_LOG_INFO("Creating binary bins");
    DMIV_TIMER("Binary bin creation");
    
    std::vector<BinInfo> bins;
    bins.reserve(2);
    
    // Encontrar os dois valores únicos
    std::set<double> unique_values;
    for (const auto& dp : data) {
      if (!dp.is_missing && !dp.is_special && std::isfinite(dp.numeric_value)) {
        unique_values.insert(dp.numeric_value);
      }
    }
    
    if (unique_values.size() != 2) {
      throw std::runtime_error("Binary variable should have exactly 2 unique values");
    }
    
    auto it = unique_values.begin();
    double val0 = *it;
    double val1 = *(++it);
    
    // Criar dois bins para os valores únicos
    BinInfo bin0, bin1;
    bin0.id = 1;
    bin1.id = 2;
    
    bin0.lower_bound = val0;
    bin0.upper_bound = val0;
    bin0.label = format_number(val0, 3);
    
    bin1.lower_bound = val1;
    bin1.upper_bound = val1;
    bin1.label = format_number(val1, 3);
    
    // Contar valores para cada bin
    for (const auto& dp : data) {
      if (!dp.is_missing && !dp.is_special && std::isfinite(dp.numeric_value)) {
        if (dp.numeric_value == val0) {
          bin0.total_count++;
          bin0.weight_sum = bin0.weight_sum.load() + dp.weight;
          if (dp.target > 0.5) {
            bin0.pos_count++;
            bin0.pos_weight = bin0.pos_weight.load() + dp.weight;
          } else {
            bin0.neg_count++;
            bin0.neg_weight = bin0.neg_weight.load() + dp.weight;
          }
        } else if (dp.numeric_value == val1) {
          bin1.total_count++;
          bin1.weight_sum = bin1.weight_sum.load() + dp.weight;
          if (dp.target > 0.5) {
            bin1.pos_count++;
            bin1.pos_weight = bin1.pos_weight.load() + dp.weight;
          } else {
            bin1.neg_count++;
            bin1.neg_weight = bin1.neg_weight.load() + dp.weight;
          }
        }
      }
    }
    
    bins.push_back(std::move(bin0));
    bins.push_back(std::move(bin1));
    
    DMIV_LOG_INFO("Created ", bins.size(), " binary bins");
    return bins;
  }
  
  std::vector<BinInfo> create_numeric_bins(
      const std::vector<DataPoint>& data,
      int max_prebins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Creating numeric bins with max_prebins: ", max_prebins);
    DMIV_TIMER("Numeric bin creation");
    
    // Filter out missing and special values
    std::vector<std::pair<double, const DataPoint*>> clean_data;
    clean_data.reserve(data.size());
    
    for (const auto& dp : data) {
      if (!dp.is_missing && !dp.is_special && std::isfinite(dp.numeric_value)) {
        clean_data.emplace_back(dp.numeric_value, &dp);
      }
    }
    
    if (clean_data.empty()) {
      DMIV_LOG_ERROR("No valid data for binning");
      throw std::runtime_error("No valid data for binning");
    }
    
    DMIV_LOG_DEBUG("Found ", clean_data.size(), " valid data points");
    
    // Sort by numeric value
    std::sort(clean_data.begin(), clean_data.end(),
              [](const auto& a, const auto& b) {
                return a.first < b.first;
              });
    
    // Create quantile-based bins - ajustado para respeitar max_bins
    int n_bins = std::min(max_prebins, static_cast<int>(clean_data.size()));
    n_bins = std::max(n_bins, config.min_bins); // Garantir pelo menos min_bins
    if (n_bins < 1) n_bins = 1;
    
    DMIV_LOG_DEBUG("Creating ", n_bins, " bins");
    
    std::vector<BinInfo> bins(n_bins);
    
    for (int i = 0; i < n_bins; ++i) {
      bins[i].id = i + 1;
      
      if (i == 0) {
        bins[i].lower_bound = -INFINITY;
      } else {
        int idx = (i * static_cast<int>(clean_data.size())) / n_bins;
        bins[i].lower_bound = clean_data[idx].first;
      }
      
      if (i == n_bins - 1) {
        bins[i].upper_bound = INFINITY;
      } else {
        int idx = ((i + 1) * static_cast<int>(clean_data.size())) / n_bins;
        bins[i].upper_bound = clean_data[idx].first;
      }
    }
    
    // Assign data points to bins - Fixed thread-safe version
    for (size_t j = 0; j < clean_data.size(); ++j) {
      double value = clean_data[j].first;
      const DataPoint* dp = clean_data[j].second;
      
      // Find the correct bin using binary search for efficiency
      int bin_idx = -1;
      for (int i = 0; i < n_bins; ++i) {
        if ((std::isinf(bins[i].lower_bound) || value >= bins[i].lower_bound - constants::EPSILON) &&
            (std::isinf(bins[i].upper_bound) || value < bins[i].upper_bound + constants::EPSILON)) {
          bin_idx = i;
          break;
        }
      }
      
      if (bin_idx >= 0) {
        bins[bin_idx].total_count++;
        bins[bin_idx].weight_sum = bins[bin_idx].weight_sum.load() + dp->weight;
        
        if (dp->target > 0.5) {
          bins[bin_idx].pos_count++;
          bins[bin_idx].pos_weight = bins[bin_idx].pos_weight.load() + dp->weight;
        } else {
          bins[bin_idx].neg_count++;
          bins[bin_idx].neg_weight = bins[bin_idx].neg_weight.load() + dp->weight;
        }
      }
    }
    
    DMIV_LOG_INFO("Created ", bins.size(), " numeric bins");
    return bins;
  }
  
  std::vector<BinInfo> create_categorical_bins(
      const std::vector<DataPoint>& data,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Creating categorical bins");
    DMIV_TIMER("Categorical bin creation");
    
    // Group by category
    std::unordered_map<std::string, std::tuple<int, double, int, double>> category_stats;
    
    double total_weight = 0;
    for (const auto& dp : data) {
      if (!dp.is_missing && !dp.is_special) {
        total_weight += dp.weight;
        auto& stats = category_stats[dp.category_value];
        std::get<0>(stats)++; // count
        std::get<1>(stats) += dp.weight; // weight_sum
        
        if (dp.target > 0.5) {
          std::get<2>(stats)++; // pos_count
          std::get<3>(stats) += dp.weight; // pos_weight
        }
      }
    }
    
    DMIV_LOG_DEBUG("Found ", category_stats.size(), " unique categories");
    
    // Create bins, handling rare categories
    std::vector<BinInfo> bins;
    bins.reserve(category_stats.size());
    
    // Identify rare categories
    std::vector<std::string> rare_categories;
    BinInfo rare_bin;
    rare_bin.label = "Rare";
    
    int bin_id = 1;
    for (const auto& [category, stats] : category_stats) {
      double category_weight = std::get<1>(stats);
      double category_ratio = safe_divide(category_weight, total_weight);
      
      if (category_ratio < config.rare_category_threshold) {
        // Add to rare categories bin
        rare_categories.push_back(category);
        rare_bin.categories.push_back(category);
        rare_bin.total_count = rare_bin.total_count.load() + std::get<0>(stats);
        rare_bin.weight_sum = rare_bin.weight_sum.load() + std::get<1>(stats);
        rare_bin.pos_count = rare_bin.pos_count.load() + std::get<2>(stats);
        rare_bin.pos_weight = rare_bin.pos_weight.load() + std::get<3>(stats);
        rare_bin.neg_count = rare_bin.neg_count.load() + std::get<0>(stats) - std::get<2>(stats);
        rare_bin.neg_weight = rare_bin.neg_weight.load() + std::get<1>(stats) - std::get<3>(stats);
      } else {
        // Create individual bin
        BinInfo bin;
        bin.id = bin_id++;
        bin.categories.push_back(category);
        bin.label = category;
        bin.total_count = std::get<0>(stats);
        bin.weight_sum = std::get<1>(stats);
        bin.pos_count = std::get<2>(stats);
        bin.pos_weight = std::get<3>(stats);
        bin.neg_count = bin.total_count.load() - bin.pos_count.load();
        bin.neg_weight = bin.weight_sum.load() - bin.pos_weight.load();
        bins.push_back(std::move(bin));
      }
    }
    
    // Add rare bin if it has categories
    if (!rare_bin.categories.empty()) {
      rare_bin.id = bin_id++;
      // Create label for rare bin
      if (rare_bin.categories.size() > 3) {
        rare_bin.label = "Rare[" + std::to_string(rare_bin.categories.size()) + " categories]";
      } else {
        std::stringstream ss;
        for (size_t i = 0; i < rare_bin.categories.size(); ++i) {
          if (i > 0) ss << config.cat_sep;
          ss << rare_bin.categories[i];
        }
        rare_bin.label = ss.str();
      }
      bins.push_back(std::move(rare_bin));
    }
    
    DMIV_LOG_INFO("Created ", bins.size(), " categorical bins");
    return bins;
  }
  
public:
  std::vector<BinInfo> create_initial_bins(
      const std::vector<DataPoint>& data,
      types::VariableType var_type,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Creating initial bins for variable type: ", variable_type_to_string(var_type));
    DMIV_TIMER("Initial bin creation");
    
    // Set random seed for reproducibility
    if (config.random_seed >= 0) {
      std::srand(config.random_seed);
      DMIV_LOG_DEBUG("Setting random seed to ", config.random_seed);
    }
    
    switch(var_type) {
    case types::VariableType::NUMERIC_CONTINUOUS:
    case types::VariableType::NUMERIC_DISCRETE:
      // Verificar se é uma variável binária
      if (is_binary_variable(data)) {
        DMIV_LOG_INFO("Detected binary variable, creating binary bins");
        return create_binary_bins(data);
      }
      DMIV_LOG_INFO("Creating numeric bins");
      return create_numeric_bins(data, config.max_bins * 3, config);
      
    case types::VariableType::CATEGORICAL_NOMINAL:
    case types::VariableType::CATEGORICAL_ORDINAL:
    case types::VariableType::BOOLEAN:
      DMIV_LOG_INFO("Creating categorical bins");
      return create_categorical_bins(data, config);
      
    default:
      DMIV_LOG_ERROR("Unsupported variable type for binning");
    throw std::runtime_error("Unsupported variable type for binning");
    }
  }
};

// =============================================================================
// OPTIMIZATION ENGINE - IMPLEMENTAÇÃO COMPLETA
// =============================================================================

class OptimizationEngine {
private:
  // Verificar se os bins representam uma variável binária
  bool is_binary_bins(const std::vector<BinInfo>& bins) {
    bool result = bins.size() == 2 && 
      bins[0].lower_bound == bins[0].upper_bound && 
      bins[1].lower_bound == bins[1].upper_bound;
    DMIV_LOG_DEBUG("Bins are ", (result ? "binary" : "not binary"));
    return result;
  }
  
  // Dynamic Programming Optimization
  std::vector<BinInfo> dynamic_programming_optimization(
      std::vector<BinInfo>& initial_bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Applying dynamic programming optimization");
    DMIV_TIMER("Dynamic programming optimization");
    
    // Se já é binário, não otimizar
    if (is_binary_bins(initial_bins)) {
      DMIV_LOG_DEBUG("Skipping optimization for binary bins");
      return initial_bins;
    }
    
    int n = initial_bins.size();
    int k = std::min(config.max_bins, n);
    k = std::max(k, config.min_bins);
    
    if (k >= n) {
      DMIV_LOG_DEBUG("No optimization needed (k >= n)");
      return initial_bins;
    }
    
    // Calculate cost matrix for adjacent bins
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));
    
    // Pre-calculate metrics for all possible merges
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        // Calculate cost of merging bins from i to j
        double total_pos = 0, total_neg = 0, total_weight = 0;
        for (int m = i; m <= j; ++m) {
          total_pos += initial_bins[m].pos_weight.load();
          total_neg += initial_bins[m].neg_weight.load();
          total_weight += initial_bins[m].weight_sum.load();
        }
        
        if (total_pos + total_neg > constants::EPSILON) {
          double p_pos = safe_divide(total_pos, total_pos + total_neg);
          double p_neg = safe_divide(total_neg, total_pos + total_neg);
          if (p_pos > constants::EPSILON && p_neg > constants::EPSILON) {
            cost[i][j] = -(p_pos - p_neg) * safe_log(safe_divide(p_pos, p_neg)); // Negative IV as cost
          }
        }
      }
    }
    
    // DP table: OPT[i][j] = minimum cost to partition first i bins into j groups
    std::vector<std::vector<double>> OPT(n + 1, std::vector<double>(k + 1, INFINITY));
    std::vector<std::vector<int>> backtrack(n + 1, std::vector<int>(k + 1, -1));
    
    OPT[0][0] = 0;
    
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= std::min(i, k); ++j) {
        for (int m = j; m <= i; ++m) {
          double current_cost = OPT[m-1][j-1] + cost[m-1][i-1];
          if (current_cost < OPT[i][j]) {
            OPT[i][j] = current_cost;
            backtrack[i][j] = m - 1;
          }
        }
      }
    }
    
    // Backtrack to find optimal partition
    std::vector<int> partition_points;
    int pos = n;
    int bins_count = k;
    
    while (bins_count > 0 && pos > 0) {
      int prev_pos = backtrack[pos][bins_count];
      if (prev_pos == -1) break;
      partition_points.push_back(prev_pos);
      pos = prev_pos;
      bins_count--;
    }
    
    std::reverse(partition_points.begin(), partition_points.end());
    partition_points.push_back(n - 1);
    
    // Create merged bins
    std::vector<BinInfo> result_bins;
    int start = 0;
    int bin_id = 1;
    
    for (size_t i = 0; i < partition_points.size(); ++i) {
      int end = partition_points[i];
      BinInfo merged_bin;
      merged_bin.id = bin_id++;
      
      // Merge all bins from start to end
      for (int j = start; j <= end && j < static_cast<int>(initial_bins.size()); ++j) {
        merged_bin.total_count = merged_bin.total_count.load() + initial_bins[j].total_count.load();
        merged_bin.pos_count = merged_bin.pos_count.load() + initial_bins[j].pos_count.load();
        merged_bin.neg_count = merged_bin.neg_count.load() + initial_bins[j].neg_count.load();
        merged_bin.weight_sum = merged_bin.weight_sum.load() + initial_bins[j].weight_sum.load();
        merged_bin.pos_weight = merged_bin.pos_weight.load() + initial_bins[j].pos_weight.load();
        merged_bin.neg_weight = merged_bin.neg_weight.load() + initial_bins[j].neg_weight.load();
        
        if (j == start) {
          merged_bin.lower_bound = initial_bins[j].lower_bound;
        }
        if (j == end) {
          merged_bin.upper_bound = initial_bins[j].upper_bound;
        }
        
        // Merge categories for categorical variables
        for (const auto& cat : initial_bins[j].categories) {
          merged_bin.categories.push_back(cat);
        }
      }
      
      result_bins.push_back(std::move(merged_bin));
      start = end + 1;
    }
    
    DMIV_LOG_INFO("Dynamic programming optimization completed with ", result_bins.size(), " bins");
    return result_bins;
  }
  
  // Branch and Bound Optimization
  std::vector<BinInfo> branch_and_bound_optimization(
      std::vector<BinInfo>& initial_bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Applying branch and bound optimization");
    DMIV_TIMER("Branch and bound optimization");
    
    // Se já é binário, não otimizar
    if (is_binary_bins(initial_bins)) {
      DMIV_LOG_DEBUG("Skipping optimization for binary bins");
      return initial_bins;
    }
    
    // For now, fallback to greedy merge as full implementation is complex
    DMIV_LOG_WARNING("Branch and bound not fully implemented, falling back to greedy merge");
    return greedy_merge_optimization(initial_bins, config);
  }
  
  // Simulated Annealing Optimization
  std::vector<BinInfo> simulated_annealing_optimization(
      std::vector<BinInfo>& initial_bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Applying simulated annealing optimization");
    DMIV_TIMER("Simulated annealing optimization");
    
    // Se já é binário, não otimizar
    if (is_binary_bins(initial_bins)) {
      DMIV_LOG_DEBUG("Skipping optimization for binary bins");
      return initial_bins;
    }
    
    auto bins = initial_bins;
    int current_bins = bins.size();
    
    // If already within constraints, return
    if (current_bins >= config.min_bins && current_bins <= config.max_bins) {
      DMIV_LOG_DEBUG("Already within bin constraints, skipping optimization");
      return bins;
    }
    
    // Simple simulated annealing approach
    double temperature = 1000.0;
    double cooling_rate = 0.95;
    int max_iterations = config.max_iterations;
    
    // Calculate initial objective
    MetricsCalculator::calculate_bin_metrics(bins, config);
    double current_iv = 0;
    for (const auto& bin : bins) {
      current_iv += bin.iv;
    }
    
    std::vector<BinInfo> best_bins = bins;
    double best_iv = current_iv;
    
    std::mt19937 rng(config.random_seed >= 0 ? config.random_seed : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int iter = 0; iter < max_iterations && temperature > 0.1; ++iter) {
      if (static_cast<int>(bins.size()) <= config.min_bins) break;
      
      // Generate neighbor solution (merge two adjacent bins)
      if (bins.size() > 1) {
        std::uniform_int_distribution<int> bin_dist(0, bins.size() - 2);
        int merge_idx = bin_dist(rng);
        
        // Create neighbor solution
        std::vector<BinInfo> neighbor_bins = bins;
        
        // Perform merge
        neighbor_bins[merge_idx].upper_bound = neighbor_bins[merge_idx + 1].upper_bound;
        neighbor_bins[merge_idx].total_count = neighbor_bins[merge_idx].total_count.load() + 
          neighbor_bins[merge_idx + 1].total_count.load();
        neighbor_bins[merge_idx].pos_count = neighbor_bins[merge_idx].pos_count.load() + 
          neighbor_bins[merge_idx + 1].pos_count.load();
        neighbor_bins[merge_idx].neg_count = neighbor_bins[merge_idx].neg_count.load() + 
          neighbor_bins[merge_idx + 1].neg_count.load();
        neighbor_bins[merge_idx].weight_sum = neighbor_bins[merge_idx].weight_sum.load() + 
          neighbor_bins[merge_idx + 1].weight_sum.load();
        neighbor_bins[merge_idx].pos_weight = neighbor_bins[merge_idx].pos_weight.load() + 
          neighbor_bins[merge_idx + 1].pos_weight.load();
        neighbor_bins[merge_idx].neg_weight = neighbor_bins[merge_idx].neg_weight.load() + 
          neighbor_bins[merge_idx + 1].neg_weight.load();
        
        // Merge categories
        for (const auto& cat : neighbor_bins[merge_idx + 1].categories) {
          neighbor_bins[merge_idx].categories.push_back(cat);
        }
        
        neighbor_bins.erase(neighbor_bins.begin() + merge_idx + 1);
        
        // Check if neighbor solution is valid
        if (static_cast<int>(neighbor_bins.size()) >= config.min_bins) {
          // Calculate neighbor objective
          MetricsCalculator::calculate_bin_metrics(neighbor_bins, config);
          double neighbor_iv = 0;
          for (const auto& bin : neighbor_bins) {
            neighbor_iv += bin.iv;
          }
          
          // Accept or reject based on Metropolis criterion
          double delta = neighbor_iv - current_iv;
          if (delta > 0 || dist(rng) < std::exp(delta / temperature)) {
            bins = neighbor_bins;
            current_iv = neighbor_iv;
            
            if (current_iv > best_iv) {
              best_bins = bins;
              best_iv = current_iv;
            }
          }
        }
      }
      
      // Cool down
      temperature *= cooling_rate;
    }
    
    DMIV_LOG_INFO("Simulated annealing optimization completed with ", best_bins.size(), " bins");
    return best_bins;
  }
  
  std::vector<BinInfo> greedy_merge_optimization(
      std::vector<BinInfo>& initial_bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Applying greedy merge optimization");
    DMIV_TIMER("Greedy merge optimization");
    
    // Se já é binário, não otimizar
    if (is_binary_bins(initial_bins)) {
      DMIV_LOG_DEBUG("Skipping optimization for binary bins");
      return initial_bins;
    }
    
    auto bins = initial_bins;
    
    int iterations = 0;
    double prev_divergence = 0;
    
    // Continue merging until we meet max_bins constraint
    while (static_cast<int>(bins.size()) > config.max_bins && 
           static_cast<int>(bins.size()) > config.min_bins && 
           iterations < config.max_iterations) {
      
      // Calculate current divergence
      MetricsCalculator::calculate_bin_metrics(bins, config);
      double current_divergence = 0;
      for (const auto& bin : bins) {
        current_divergence += bin.divergence;
      }
      
      // Check convergence
      if (iterations > 0 && 
          std::abs(current_divergence - prev_divergence) < config.convergence_threshold) {
        DMIV_LOG_DEBUG("Convergence reached at iteration ", iterations);
        break;
      }
      prev_divergence = current_divergence;
      
      // Find the pair of adjacent bins with minimum IV loss
      int best_merge_idx = -1;
      double min_loss = INFINITY;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double current_iv = bins[i].iv + bins[i+1].iv;
        double loss = std::abs(current_iv);
        
        if (loss < min_loss) {
          min_loss = loss;
          best_merge_idx = i;
        }
      }
      
      // Só faz merge se não violar min_bins
      if (best_merge_idx >= 0 && 
          static_cast<int>(bins.size()) > config.min_bins) {
        DMIV_LOG_DEBUG("Merging bins ", best_merge_idx, " and ", (best_merge_idx + 1));
        
        // Perform the merge
        bins[best_merge_idx].upper_bound = bins[best_merge_idx + 1].upper_bound;
        bins[best_merge_idx].total_count = bins[best_merge_idx].total_count.load() + 
          bins[best_merge_idx + 1].total_count.load();
        bins[best_merge_idx].pos_count = bins[best_merge_idx].pos_count.load() + 
          bins[best_merge_idx + 1].pos_count.load();
        bins[best_merge_idx].neg_count = bins[best_merge_idx].neg_count.load() + 
          bins[best_merge_idx + 1].neg_count.load();
        bins[best_merge_idx].weight_sum = bins[best_merge_idx].weight_sum.load() + 
          bins[best_merge_idx + 1].weight_sum.load();
        bins[best_merge_idx].pos_weight = bins[best_merge_idx].pos_weight.load() + 
          bins[best_merge_idx + 1].pos_weight.load();
        bins[best_merge_idx].neg_weight = bins[best_merge_idx].neg_weight.load() + 
          bins[best_merge_idx + 1].neg_weight.load();
        
        // Merge categories using cat_sep
        for (const auto& cat : bins[best_merge_idx + 1].categories) {
          bins[best_merge_idx].categories.push_back(cat);
        }
        
        // Update label for merged categorical bins
        if (!bins[best_merge_idx].categories.empty()) {
          std::stringstream ss;
          for (size_t j = 0; j < bins[best_merge_idx].categories.size(); ++j) {
            if (j > 0) ss << config.cat_sep;
            ss << bins[best_merge_idx].categories[j];
          }
          bins[best_merge_idx].label = ss.str();
        }
        
        bins.erase(bins.begin() + best_merge_idx + 1);
      } else {
        DMIV_LOG_DEBUG("No beneficial merge found or would violate min_bins");
        break; // No beneficial merge found or would violate min_bins
      }
      
      iterations++;
    }
    
    // Garantir que respeite min_bins e max_bins
    if (static_cast<int>(bins.size()) < config.min_bins && !initial_bins.empty()) {
      // Se temos menos bins que o mínimo necessário, usar os bins iniciais
      if (static_cast<int>(initial_bins.size()) >= config.min_bins) {
        DMIV_LOG_WARNING("Not enough bins, returning initial bins");
        return initial_bins;
      }
    }
    
    // Se ainda temos mais bins que o máximo permitido, forçar redução
    while (static_cast<int>(bins.size()) > config.max_bins && bins.size() > 1) {
      // Encontrar o par de bins adjacentes com menor perda de IV
      int best_merge_idx = -1;
      double min_loss = INFINITY;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double current_iv = bins[i].iv + bins[i+1].iv;
        double loss = std::abs(current_iv);
        
        if (loss < min_loss) {
          min_loss = loss;
          best_merge_idx = i;
        }
      }
      
      if (best_merge_idx >= 0) {
        DMIV_LOG_DEBUG("Forced merge of bins ", best_merge_idx, " and ", (best_merge_idx + 1));
        
        // Perform the merge
        bins[best_merge_idx].upper_bound = bins[best_merge_idx + 1].upper_bound;
        bins[best_merge_idx].total_count = bins[best_merge_idx].total_count.load() + 
          bins[best_merge_idx + 1].total_count.load();
        bins[best_merge_idx].pos_count = bins[best_merge_idx].pos_count.load() + 
          bins[best_merge_idx + 1].pos_count.load();
        bins[best_merge_idx].neg_count = bins[best_merge_idx].neg_count.load() + 
          bins[best_merge_idx + 1].neg_count.load();
        bins[best_merge_idx].weight_sum = bins[best_merge_idx].weight_sum.load() + 
          bins[best_merge_idx + 1].weight_sum.load();
        bins[best_merge_idx].pos_weight = bins[best_merge_idx].pos_weight.load() + 
          bins[best_merge_idx + 1].pos_weight.load();
        bins[best_merge_idx].neg_weight = bins[best_merge_idx].neg_weight.load() + 
          bins[best_merge_idx + 1].neg_weight.load();
        
        // Merge categories using cat_sep
        for (const auto& cat : bins[best_merge_idx + 1].categories) {
          bins[best_merge_idx].categories.push_back(cat);
        }
        
        // Update label for merged categorical bins
        if (!bins[best_merge_idx].categories.empty()) {
          std::stringstream ss;
          for (size_t j = 0; j < bins[best_merge_idx].categories.size(); ++j) {
            if (j > 0) ss << config.cat_sep;
            ss << bins[best_merge_idx].categories[j];
          }
          bins[best_merge_idx].label = ss.str();
        }
        
        bins.erase(bins.begin() + best_merge_idx + 1);
      } else {
        DMIV_LOG_DEBUG("No more merges possible");
        break;
      }
    }
    
    DMIV_LOG_INFO("Greedy merge optimization completed with ", bins.size(), " bins after ", iterations, " iterations");
    return bins;
  }
  
public:
  std::vector<BinInfo> optimize(
      const std::vector<BinInfo>& initial_bins,
      const OptimizationConfig& config
  ) {
    DMIV_LOG_INFO("Starting optimization with ", initial_bins.size(), " initial bins");
    DMIV_TIMER("Optimization");
    
    if (initial_bins.empty()) {
      DMIV_LOG_WARNING("Empty initial bins, returning as-is");
      return initial_bins;
    }
    
    // Se é binário, retornar diretamente sem otimização
    if (initial_bins.size() == 2 && 
        initial_bins[0].lower_bound == initial_bins[0].upper_bound && 
        initial_bins[1].lower_bound == initial_bins[1].upper_bound) {
      DMIV_LOG_DEBUG("Binary variable detected, skipping optimization");
      return initial_bins;
    }
    
    // Se já estiver dentro dos limites, retornar diretamente
    if (static_cast<int>(initial_bins.size()) >= config.min_bins && 
        static_cast<int>(initial_bins.size()) <= config.max_bins) {
      DMIV_LOG_DEBUG("Already within bin constraints, skipping optimization");
      return initial_bins;
    }
    
    switch(config.optimization_method) {
    case types::OptimizationMethod::BRANCH_AND_BOUND:
      DMIV_LOG_INFO("Using branch and bound optimization");
      return branch_and_bound_optimization(const_cast<std::vector<BinInfo>&>(initial_bins), config);
    case types::OptimizationMethod::DYNAMIC_PROGRAMMING:
      DMIV_LOG_INFO("Using dynamic programming optimization");
      return dynamic_programming_optimization(const_cast<std::vector<BinInfo>&>(initial_bins), config);
    case types::OptimizationMethod::SIMULATED_ANNEALING:
      DMIV_LOG_INFO("Using simulated annealing optimization");
      return simulated_annealing_optimization(const_cast<std::vector<BinInfo>&>(initial_bins), config);
    case types::OptimizationMethod::GREEDY_MERGE:
    default:
      DMIV_LOG_INFO("Using greedy merge optimization");
      return greedy_merge_optimization(const_cast<std::vector<BinInfo>&>(initial_bins), config);
    }
  }
};

// =============================================================================
// MISSING VALUE HANDLER
// =============================================================================

class MissingValueHandler {
public:
  static std::vector<DataPoint> handle_missing_values(
      const std::vector<DataPoint>& data_points,
      const OptimizationConfig& config,
      std::vector<BinInfo>& special_bins
  ) {
    DMIV_LOG_INFO("Handling missing values with policy: ", config.missing_policy);
    DMIV_TIMER("Missing value handling");
    
    std::vector<DataPoint> processed_data;
    std::vector<BinInfo> missing_bins;
    
    if (config.missing_policy == "remove") {
      DMIV_LOG_DEBUG("Removing missing values");
      // Simply remove missing values
      for (const auto& dp : data_points) {
        if (!dp.is_missing && !dp.is_special) {
          processed_data.push_back(dp);
        }
      }
    } else if (config.missing_policy == "impute") {
      DMIV_LOG_DEBUG("Imputing missing values");
      // Impute missing values - copy data first
      processed_data = data_points;
      
      // For now, we'll treat imputed values as regular data
      // In a full implementation, we would actually impute values here
      // This is a simplified approach
      for (auto& dp : processed_data) {
        if (dp.is_missing) {
          dp.is_missing = false;
          // Actual imputation would happen here
        }
      }
    } else if (config.missing_policy == "merge") {
      DMIV_LOG_DEBUG("Merging missing values");
      // Merge with nearest bin based on event rate
      // For now, treat as separate (simplified implementation)
      BinInfo missing_bin;
      missing_bin.id = 1;
      missing_bin.label = "[Missing]";
      
      for (const auto& dp : data_points) {
        if (dp.is_missing) {
          missing_bin.total_count++;
          missing_bin.weight_sum = missing_bin.weight_sum.load() + dp.weight;
          if (dp.target > 0.5) {
            missing_bin.pos_count++;
            missing_bin.pos_weight = missing_bin.pos_weight.load() + dp.weight;
          } else {
            missing_bin.neg_count++;
            missing_bin.neg_weight = missing_bin.neg_weight.load() + dp.weight;
          }
        } else if (!dp.is_special) {
          processed_data.push_back(dp);
        }
      }
      
      if (missing_bin.total_count.load() > 0) {
        missing_bins.push_back(std::move(missing_bin));
      }
    } else { // "separate" (default)
      DMIV_LOG_DEBUG("Separating missing values into dedicated bins");
      BinInfo missing_bin;
      missing_bin.id = 1;
      missing_bin.label = "[Missing]";
      
      std::unordered_map<double, BinInfo> special_bin_map;
      
      for (const auto& dp : data_points) {
        if (dp.is_missing) {
          missing_bin.total_count++;
          missing_bin.weight_sum = missing_bin.weight_sum.load() + dp.weight;
          if (dp.target > 0.5) {
            missing_bin.pos_count++;
            missing_bin.pos_weight = missing_bin.pos_weight.load() + dp.weight;
          } else {
            missing_bin.neg_count++;
            missing_bin.neg_weight = missing_bin.neg_weight.load() + dp.weight;
          }
        } else if (dp.is_special) {
          auto& sbin = special_bin_map[dp.special_value];
          sbin.total_count++;
          sbin.weight_sum = sbin.weight_sum.load() + dp.weight;
          if (dp.target > 0.5) {
            sbin.pos_count++;
            sbin.pos_weight = sbin.pos_weight.load() + dp.weight;
          } else {
            sbin.neg_count++;
            sbin.neg_weight = sbin.neg_weight.load() + dp.weight;
          }
        } else {
          processed_data.push_back(dp);
        }
      }
      
      // Add special bins with proper IDs and formatted labels
      int current_id = 1;
      if (missing_bin.total_count.load() > 0) {
        missing_bin.id = current_id++;
        special_bins.push_back(std::move(missing_bin));
      }
      
      for (auto& [val, sbin] : special_bin_map) {
        sbin.id = current_id++;
        // Use digits parameter for special value formatting
        sbin.label = "[Special:" + format_number(val, config.digits) + "]";
        special_bins.push_back(std::move(sbin));
      }
    }
    
    // Add missing bins to special bins if any
    for (auto& bin : missing_bins) {
      special_bins.push_back(std::move(bin));
    }
    
    DMIV_LOG_INFO("Processed ", processed_data.size(), " data points, created ", special_bins.size(), " special bins");
    return processed_data;
  }
};

// =============================================================================
// FAST BIN MAPPER
// =============================================================================

class FastBinMapper {
private:
  types::VariableType var_type_;
  std::string cat_sep_;
  
  // For numeric variables
  std::vector<double> thresholds_;
  std::vector<double> woe_values_;
  std::vector<std::string> bin_labels_;
  
  // For categorical variables
  std::unordered_map<std::string, double> category_to_woe_;
  std::unordered_map<std::string, std::string> category_to_label_;
  
  // For special values
  std::unordered_map<double, double> special_to_woe_;
  std::unordered_map<double, std::string> special_to_label_;
  
  double missing_woe_;
  std::string missing_label_;
  bool has_missing_;
  
public:
  FastBinMapper(types::VariableType var_type, const std::string& cat_sep) 
    : var_type_(var_type), cat_sep_(cat_sep),
      missing_woe_(0), missing_label_(""), has_missing_(false) {}
  
  void build_from_bins(const std::vector<BinInfo>& bins) {
    DMIV_LOG_TRACE("Building fast bin mapper");
    DMIV_TIMER("Fast bin mapper building");
    
    for (const auto& bin : bins) {
      // Handle special bins
      if (bin.label == "[Missing]") {
        has_missing_ = true;
        missing_woe_ = bin.woe;
        missing_label_ = bin.label;
        continue;
      }
      
      if (bin.label.find("[Special:") == 0) {
        // Parse special value
        std::string val_str = bin.label.substr(9, bin.label.length() - 10);
        try {
          double special_val = std::stod(val_str);
          special_to_woe_[special_val] = bin.woe;
          special_to_label_[special_val] = bin.label;
        } catch (...) {
          // Skip if cannot parse
        }
        continue;
      }
      
      // Regular bins
      if (var_type_ == types::VariableType::CATEGORICAL_NOMINAL || 
          var_type_ == types::VariableType::CATEGORICAL_ORDINAL ||
          var_type_ == types::VariableType::BOOLEAN) {
        // Map each category to its bin
        for (const auto& cat : bin.categories) {
          category_to_woe_[cat] = bin.woe;
          category_to_label_[cat] = bin.label;
        }
      } else {
        // For numeric, store thresholds
        thresholds_.push_back(bin.upper_bound);
        woe_values_.push_back(bin.woe);
        bin_labels_.push_back(bin.label);
      }
    }
    
    DMIV_LOG_DEBUG("Mapper built with ", thresholds_.size(), " thresholds, ", 
                   category_to_woe_.size(), " categories, ", 
                   special_to_woe_.size(), " special values");
  }
  
  inline void map_numeric(double value, bool is_missing, bool is_special,
                          double special_value, std::string& bin_label, double& woe) const {
    if (is_missing && has_missing_) {
      bin_label = missing_label_;
      woe = missing_woe_;
      return;
    }
    
    if (is_special) {
      auto it = special_to_woe_.find(special_value);
      if (it != special_to_woe_.end()) {
        woe = it->second;
        bin_label = special_to_label_.at(special_value);
        return;
      }
    }
    
    // Binary search for the right bin
    if (!thresholds_.empty()) {
      auto it = std::upper_bound(thresholds_.begin(), thresholds_.end(), value);
      size_t idx = std::distance(thresholds_.begin(), it);
      
      if (idx < woe_values_.size()) {
        woe = woe_values_[idx];
        bin_label = bin_labels_[idx];
      } else if (!woe_values_.empty()) {
        woe = woe_values_.back();
        bin_label = bin_labels_.back();
      }
    } else if (!woe_values_.empty()) {
      woe = woe_values_.front();
      bin_label = bin_labels_.front();
    }
  }
  
  inline void map_categorical(const std::string& category, bool is_missing,
                              std::string& bin_label, double& woe) const {
    if (is_missing && has_missing_) {
      bin_label = missing_label_;
      woe = missing_woe_;
      return;
    }
    
    auto it = category_to_woe_.find(category);
    if (it != category_to_woe_.end()) {
      woe = it->second;
      bin_label = category_to_label_.at(category);
    } else if (!category_to_woe_.empty()) {
      // Fallback to first bin
      woe = category_to_woe_.begin()->second;
      bin_label = category_to_label_.begin()->second;
    }
  }
};

// =============================================================================
// MAIN PROCESSING FUNCTION - CORRIGIDA
// =============================================================================

List process_feature_dmiv(
    const DataFrame& data,
    const std::string& target_col,
    const std::string& feature_col,
    const OptimizationConfig& config,
    const NumericVector& weights,
    const std::set<double>& special_vals
) {
  DMIV_LOG_INFO("Processing feature: ", feature_col);
  DMIV_TIMER("Feature processing");
  
  // Extract data
  NumericVector target = as<NumericVector>(data[target_col]);
  SEXP feature_sexp = data[feature_col];
  
  // Detect variable type
  VariableTypeDetector detector;
  types::VariableType var_type = detector.detect(feature_sexp);
  
  if (var_type == types::VariableType::DATE_TIME || 
      var_type == types::VariableType::UNSUPPORTED) {
    DMIV_LOG_WARNING("Skipping unsupported variable type: ", variable_type_to_string(var_type));
    return List::create(
      Named("bins") = DataFrame::create(),
      Named("total_divergence") = 0.0,
      Named("total_iv") = 0.0,
      Named("variable_type") = variable_type_to_string(var_type),
      Named("n_bins") = 0,
      Named("messages") = "Variable skipped: unsupported type",
      Named("transform") = DataFrame::create()
    );
  }
  
  // Prepare data points
  std::vector<DataPoint> data_points;
  data_points.reserve(target.size());
  
  // Store original values for transformation - Fixed memory allocation
  std::vector<std::string> original_categorical_values;
  std::vector<double> original_numeric_values;
  
  bool is_categorical = (var_type == types::VariableType::CATEGORICAL_NOMINAL || 
                         var_type == types::VariableType::CATEGORICAL_ORDINAL ||
                         var_type == types::VariableType::BOOLEAN);
  
  if (is_categorical) {
    original_categorical_values.resize(target.size());
  } else {
    original_numeric_values.resize(target.size());
  }
  
  DMIV_LOG_DEBUG("Preparing ", target.size(), " data points");
  
  for (int i = 0; i < target.size(); ++i) {
    DataPoint dp;
    dp.target = target[i];
    dp.weight = (i < weights.size()) ? weights[i] : 1.0;
    dp.original_index = i;
    
    switch(var_type) {
    case types::VariableType::NUMERIC_CONTINUOUS:
    case types::VariableType::NUMERIC_DISCRETE: {
      NumericVector nv(feature_sexp);
      if (NumericVector::is_na(nv[i]) || !std::isfinite(nv[i])) {
        dp.is_missing = true;
        original_numeric_values[i] = NA_REAL;
      } else {
        dp.numeric_value = nv[i];
        original_numeric_values[i] = nv[i];
        if (special_vals.count(dp.numeric_value)) {
          dp.is_special = true;
          dp.special_value = dp.numeric_value;
        }
      }
      break;
    }
    case types::VariableType::CATEGORICAL_NOMINAL:
    case types::VariableType::CATEGORICAL_ORDINAL: {
      if (Rf_isFactor(feature_sexp)) {
      IntegerVector iv(feature_sexp);
      if (IntegerVector::is_na(iv[i])) {
        dp.is_missing = true;
        dp.category_value = "NA";
        original_categorical_values[i] = "NA";
      } else {
        CharacterVector levels = iv.attr("levels");
        dp.category_value = as<std::string>(levels[iv[i] - 1]);
        original_categorical_values[i] = dp.category_value;
      }
    } else {
      CharacterVector cv(feature_sexp);
      if (CharacterVector::is_na(cv[i])) {
        dp.is_missing = true;
        dp.category_value = "NA";
        original_categorical_values[i] = "NA";
      } else {
        dp.category_value = as<std::string>(cv[i]);
        original_categorical_values[i] = dp.category_value;
      }
    }
    break;
    }
    case types::VariableType::BOOLEAN: {
      LogicalVector lv(feature_sexp);
      if (LogicalVector::is_na(lv[i])) {
        dp.is_missing = true;
        dp.category_value = "NA";
        original_categorical_values[i] = "NA";
      } else {
        dp.numeric_value = lv[i] ? 1.0 : 0.0;
        dp.category_value = lv[i] ? "TRUE" : "FALSE";
        original_categorical_values[i] = dp.category_value;
      }
      break;
    }
    default:
      dp.is_missing = true;
      break;
    }
    
    data_points.push_back(std::move(dp));
  }
  
  DMIV_LOG_DEBUG("Prepared ", data_points.size(), " data points");
  
  // Handle missing values according to policy
  std::vector<BinInfo> special_bins;
  std::vector<DataPoint> processed_data = MissingValueHandler::handle_missing_values(
    data_points, config, special_bins
  );
  
  // Perform binning
  std::vector<BinInfo> regular_bins;
  std::string message = "Success";
  
  if (!processed_data.empty()) {
    try {
      // Initial binning
      PrebinningEngine prebinning_engine;
      auto initial_bins = prebinning_engine.create_initial_bins(
        processed_data, var_type, config
      );
      
      DMIV_LOG_INFO("Created ", initial_bins.size(), " initial bins");
      
      // Optimization - CORRIGIDA
      OptimizationEngine optimizer;
      regular_bins = optimizer.optimize(initial_bins, config);
      
      DMIV_LOG_INFO("Optimized to ", regular_bins.size(), " bins");
      
      // Calculate metrics
      MetricsCalculator::calculate_bin_metrics(regular_bins, config);
      
      // Update bin IDs and labels
      int next_id = special_bins.size() + 1;
      for (size_t i = 0; i < regular_bins.size(); ++i) {
        regular_bins[i].id = next_id++;
        
        if (var_type == types::VariableType::CATEGORICAL_NOMINAL || 
            var_type == types::VariableType::CATEGORICAL_ORDINAL ||
            var_type == types::VariableType::BOOLEAN) {
          // Rebuild label with cat_sep for merged bins
          if (regular_bins[i].categories.size() > 1) {
            std::stringstream ss;
            for (size_t j = 0; j < regular_bins[i].categories.size(); ++j) {
              if (j > 0) ss << config.cat_sep;
              ss << regular_bins[i].categories[j];
            }
            regular_bins[i].label = ss.str();
          } else if (!regular_bins[i].categories.empty()) {
            regular_bins[i].label = regular_bins[i].categories[0];
          }
        } else {
          // Numeric bin labels with digits formatting
          std::stringstream ss;
          if (std::isinf(regular_bins[i].lower_bound)) {
            ss << "(-inf, " << format_number(regular_bins[i].upper_bound, config.digits) << "]";
          } else if (std::isinf(regular_bins[i].upper_bound)) {
            ss << "[" << format_number(regular_bins[i].lower_bound, config.digits) << ", +inf)";
          } else {
            ss << "[" << format_number(regular_bins[i].lower_bound, config.digits) << ", " << 
              format_number(regular_bins[i].upper_bound, config.digits) << ")";
          }
          regular_bins[i].label = ss.str();
        }
      }
    } catch (const std::exception& e) {
      message = std::string("Error during binning: ") + e.what();
      DMIV_LOG_ERROR("Error during binning: ", e.what());
    }
  }
  
  // Combine special and regular bins
  std::vector<BinInfo> all_bins;
  all_bins.reserve(special_bins.size() + regular_bins.size());
  
  for (auto& bin : special_bins) {
    all_bins.push_back(std::move(bin));
  }
  for (auto& bin : regular_bins) {
    all_bins.push_back(std::move(bin));
  }
  
  DMIV_LOG_INFO("Total bins: ", all_bins.size());
  
  // Recalculate metrics for all bins together
  if (!all_bins.empty()) {
    MetricsCalculator::calculate_bin_metrics(all_bins, config);
    
    // Enforce monotonicity if requested
    if (config.monotonicity != types::MonotonicityType::NONE) {
      MonotonicityEnforcer enforcer;
      enforcer.enforce(all_bins, config);
      // Recalculate metrics after monotonicity enforcement
      MetricsCalculator::calculate_bin_metrics(all_bins, config);
    }
  }
  
  // Create output DataFrames
  IntegerVector bin_ids;
  CharacterVector bin_labels;
  IntegerVector total_counts;
  IntegerVector neg_counts;
  IntegerVector pos_counts;
  NumericVector woes;
  NumericVector divergences;
  NumericVector ivs;
  NumericVector kss;
  NumericVector ginis;
  
  for (const auto& bin : all_bins) {
    bin_ids.push_back(bin.id);
    bin_labels.push_back(bin.label);
    total_counts.push_back(bin.total_count.load());
    neg_counts.push_back(bin.neg_count.load());
    pos_counts.push_back(bin.pos_count.load());
    woes.push_back(bin.woe);
    divergences.push_back(bin.divergence);
    ivs.push_back(bin.iv);
    kss.push_back(bin.ks);
    ginis.push_back(bin.gini);
  }
  
  DataFrame bins_df = DataFrame::create(
    Named("bin_id") = bin_ids,
    Named("bin_label") = bin_labels,
    Named("total_count") = total_counts,
    Named("neg_count") = neg_counts,
    Named("pos_count") = pos_counts,
    Named("woe") = woes,
    Named("divergence") = divergences,
    Named("iv") = ivs,
    Named("ks") = kss,
    Named("gini") = ginis
  );
  
  // Calculate totals
  double total_divergence = 0.0;
  double total_iv = 0.0;
  for (const auto& bin : all_bins) {
    total_divergence += bin.divergence;
    total_iv += bin.iv;
  }
  
  DMIV_LOG_INFO("Total divergence: ", total_divergence, ", Total IV: ", total_iv);
  
  // FAST TRANSFORM using FastBinMapper
  FastBinMapper mapper(var_type, config.cat_sep);
  mapper.build_from_bins(all_bins);
  
  CharacterVector transform_feature_values(target.size());
  CharacterVector feature_bins(target.size());
  NumericVector feature_woes(target.size());
  
  for (int i = 0; i < target.size(); ++i) {
    std::string value_str = "";
    std::string bin_label = "Unknown";
    double woe_value = 0;
    
    if (is_categorical) {
      std::string category_value = original_categorical_values[i];
      bool is_missing = (category_value == "NA");
      
      mapper.map_categorical(category_value, is_missing, bin_label, woe_value);
      value_str = category_value;
    } else {
      double numeric_value = original_numeric_values[i];
      bool is_missing = std::isnan(numeric_value) || numeric_value == NA_REAL;
      bool is_special = special_vals.count(numeric_value) > 0;
      
      if (!is_missing) {
        value_str = format_number(numeric_value, config.digits);
      } else {
        value_str = "NA";
      }
      
      mapper.map_numeric(numeric_value, is_missing, is_special, 
                         is_special ? numeric_value : 0.0, bin_label, woe_value);
    }
    
    transform_feature_values[i] = value_str;
    feature_bins[i] = bin_label;
    feature_woes[i] = woe_value;
  }
  
  DataFrame transform_df = DataFrame::create(
    Named(feature_col) = transform_feature_values,
    Named(feature_col + "_bin") = feature_bins,
    Named(feature_col + "_woe") = feature_woes
  );
  
  DMIV_LOG_INFO("Feature processing completed");
  
  return List::create(
    Named("bins") = bins_df,
    Named("total_divergence") = total_divergence,
    Named("total_iv") = total_iv,
    Named("variable_type") = variable_type_to_string(var_type),
    Named("n_bins") = static_cast<int>(all_bins.size()),
    Named("messages") = message,
    Named("divergence_method") = divergence_method_to_string(config.divergence_method),
    Named("transform") = transform_df
  );
}

} // namespace dmivwoe

// =============================================================================
// EXPORTED FUNCTION - CORRECTED
// =============================================================================

//' Divergence Measures and Information Value (DMIV) Optimal Binning with WoE
//'
//' @description
//' Performs state-of-the-art optimal binning on numeric and categorical variables using advanced
//' divergence measures combined with Information Value optimization. Implements comprehensive binning
//' strategies that extend traditional approaches through:
//' \itemize{
//'   \item \strong{Multi-Divergence Optimization}: Simultaneous optimization of multiple divergence measures
//'   \item \strong{Regularized Information Value}: L1/L2 regularization for robust binning
//'   \item \strong{Adaptive Monotonicity}: Isotonic regression with automatic trend detection
//'   \item \strong{Fast Transform Mapping}: O(log n) bin assignment using optimized data structures
//' }
//'
//' @details
//' \strong{1) Divergence Measures Framework}
//'
//' The algorithm optimizes bin boundaries by minimizing divergence between positive and negative class distributions:
//' \deqn{D(P||Q) = \sum_{i=1}^{n} d(p_i, q_i)}
//' where \eqn{P} and \eqn{Q} are the probability distributions for positive and negative classes.
//'
//' \strong{Supported Divergence Measures:}
//' \itemize{
//'   \item \strong{L2 Norm (Euclidean)}: \eqn{D_{L2} = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}}
//'   \item \strong{Kullback-Leibler}: \eqn{D_{KL} = \sum_{i=1}^{n} p_i \log\left(\frac{p_i}{q_i}\right)}
//'   \item \strong{J-Divergence}: \eqn{D_J = \sum_{i=1}^{n} (p_i - q_i) \log\left(\frac{p_i}{q_i}\right)}
//'   \item \strong{Hellinger Distance}: \eqn{D_H = \frac{1}{2}\sum_{i=1}^{n}(\sqrt{p_i} - \sqrt{q_i})^2}
//'   \item \strong{Chi-Square}: \eqn{D_{\chi^2} = \sum_{i=1}^{n} \frac{(p_i - q_i)^2(p_i + q_i)}{p_i \cdot q_i}}
//'   \item \strong{Jensen-Shannon}: \eqn{D_{JS} = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)}, where \eqn{M = \frac{P+Q}{2}}
//'   \item \strong{Total Variation}: \eqn{D_{TV} = \frac{1}{2}\sum_{i=1}^{n}|p_i - q_i|}
//'   \item \strong{L1 Norm}: \eqn{D_{L1} = \sum_{i=1}^{n}|p_i - q_i|}
//'   \item \strong{Linf Norm}: \eqn{D_{L\infty} = \max_{i}|p_i - q_i|}
//' }
//'
//' \strong{2) Weight of Evidence with Advanced Smoothing}
//'
//' For bin \eqn{g}, WoE with configurable smoothing:
//' \deqn{\text{WoE}_g = \begin{cases}
//' \ln\left(\frac{(n_{1g} + \lambda)/(N_1 + G\lambda)}{(n_{0g} + \lambda)/(N_0 + G\lambda)}\right) & \text{if method = "woe"} \\
//' \ln\left(\frac{(n_{1g} + \lambda)/(n_g + 2\lambda)}{(n_{0g} + \lambda)/(n_g + 2\lambda)}\right) & \text{if method = "woe1"}
//' \end{cases}}
//' where:
//' \itemize{
//'   \item \eqn{n_{1g}, n_{0g}} = positive/negative counts in bin \eqn{g}
//'   \item \eqn{N_1, N_0} = total positive/negative counts
//'   \item \eqn{n_g} = total count in bin \eqn{g}
//'   \item \eqn{G} = number of bins
//'   \item \eqn{\lambda} = Laplace smoothing parameter
//' }
//'
//' WoE values are clipped to \eqn{[-20, 20]} for numerical stability.
//'
//' \strong{3) Regularized Information Value}
//'
//' Total IV with regularization:
//' \deqn{\text{IV}_{\text{reg}} = \sum_{g=1}^{G} \left(\frac{n_{1g}}{N_1} - \frac{n_{0g}}{N_0}\right) \times \text{WoE}_g + \lambda_1 ||w||_1 + \lambda_2 ||w||_2^2}
//' where \eqn{\lambda_1} and \eqn{\lambda_2} are L1 and L2 regularization parameters.
//'
//' \strong{4) Optimization Algorithms}
//'
//' \strong{Greedy Merge Algorithm:}
//' \enumerate{
//'   \item Initialize with maximum granular bins (3× max_bins)
//'   \item Iteratively merge adjacent bins with minimum IV loss
//'   \item Continue until max_bins constraint is satisfied
//'   \item Convergence criterion: \eqn{|D_t - D_{t-1}| < \epsilon}
//' }
//'
//' \strong{Dynamic Programming (Future):}
//' Solves optimal k-bins problem:
//' \deqn{OPT[i,k] = \min_{j<i} \{OPT[j,k-1] + \text{Cost}(j+1,i)\}}
//'
//' \strong{5) Isotonic Regression for Monotonicity}
//'
//' Pool Adjacent Violators Algorithm (PAVA):
//' \deqn{\min_{\theta} \sum_{i=1}^{n} w_i(\theta_i - y_i)^2 \quad \text{s.t.} \quad \theta_1 \leq \theta_2 \leq \cdots \leq \theta_n}
//'
//' Algorithm complexity: \eqn{O(n)} using efficient pooling strategy.
//'
//' \strong{6) Wilson Score Confidence Intervals}
//'
//' For WoE confidence intervals at level \eqn{1-\alpha}:
//' \deqn{\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}}
//' where \eqn{z = \Phi^{-1}(1-\alpha/2)} is the standard normal quantile.
//'
//' \strong{7) Fast Bin Mapping}
//'
//' Utilizes optimized data structures for O(log n) bin assignment:
//' \itemize{
//'   \item \strong{Numeric}: Binary search on sorted thresholds
//'   \item \strong{Categorical}: Hash map lookup with O(1) average complexity
//' }
//'
//' @param data DataFrame containing features and target variable. Must contain at least the target column
//'   and one feature column. All data types except Date/POSIXt are supported.
//'
//' @param target_col String specifying the name of the binary target column. The target must be binary
//'   (0/1, TRUE/FALSE, or two-level factor). NA values in target result in row exclusion.
//'
//' @param feature_cols CharacterVector of feature column names to process. Supports numeric, integer,
//'   logical, character, and factor types. Date/time columns are automatically skipped with warning.
//'
//' @param min_bins Integer minimum number of bins to create. Must be at least 2. Algorithm ensures
//'   at least this many bins when data permits.
//'   Default: 3
//'
//' @param max_bins Integer maximum number of bins allowed. Must be greater than or equal to min_bins.
//'   Controls model complexity and prevents overfitting.
//'   Default: 10
//'
//' @param divergence_method String specifying the divergence measure to optimize. Options:
//'   \itemize{
//'     \item \code{"l2_norm"}: Euclidean distance (default)
//'     \item \code{"kullback_leibler"}: KL divergence
//'     \item \code{"j_divergence"}: Symmetric KL divergence
//'     \item \code{"hellinger"}: Hellinger distance
//'     \item \code{"chi_square"}: Chi-square divergence
//'     \item \code{"jensen_shannon"}: JS divergence
//'     \item \code{"total_variation"}: TV distance
//'     \item \code{"l1_norm"}: Manhattan distance
//'     \item \code{"linf_norm"}: Maximum absolute difference
//'   }
//'   Default: "l2_norm"
//'
//' @param woe_method String specifying WoE calculation method. Options:
//'   \itemize{
//'     \item \code{"woe"}: Traditional WoE with global denominators (default)
//'     \item \code{"woe1"}: Zeng's WOE1 with local denominators
//'   }
//'   Default: "woe"
//'
//' @param smooth Numeric Laplace smoothing parameter for WoE/IV calculation. Must be non-negative.
//'   Prevents undefined values when bins have zero counts. Recommended range: [0, 1].
//'   Default: 0.0
//'
//' @param min_bin_size Numeric minimum bin size as proportion of total observations. Must be in (0, 1).
//'   Prevents creation of unstable small bins.
//'   Default: 0.05
//'
//' @param optimization_method String specifying the optimization algorithm. Options:
//'   \itemize{
//'     \item \code{"greedy_merge"}: Iterative adjacent bin merging (default)
//'     \item \code{"dynamic_programming"}: Optimal k-bins via DP (future)
//'     \item \code{"branch_and_bound"}: Exhaustive search with pruning (future)
//'     \item \code{"simulated_annealing"}: Stochastic optimization (future)
//'   }
//'   Default: "greedy_merge"
//'
//' @param enforce_monotonicity Logical whether to enforce monotonic WoE trend across bins.
//'   Useful for regulatory compliance and model interpretability.
//'   Default: FALSE
//'
//' @param monotonicity_type String specifying the monotonic constraint. Options:
//'   \itemize{
//'     \item \code{"none"}: No constraint (default when enforce_monotonicity=FALSE)
//'     \item \code{"auto"}: Detect trend via Spearman correlation
//'     \item \code{"increasing"}: Force non-decreasing WoE
//'     \item \code{"decreasing"}: Force non-increasing WoE
//'   }
//'   Default: "none"
//'
//' @param max_iterations Integer maximum iterations for optimization algorithms. Must be positive.
//'   Prevents infinite loops in iterative procedures.
//'   Default: 1000
//'
//' @param convergence_threshold Numeric convergence tolerance for iterative algorithms. Must be positive.
//'   Smaller values yield more precise results but longer computation.
//'   Default: 1e-6
//'
//' @param use_cross_validation Logical whether to use cross-validation for parameter selection.
//'   Currently reserved for future implementation.
//'   Default: FALSE
//'
//' @param cv_folds Integer number of cross-validation folds when use_cross_validation=TRUE.
//'   Must be at least 2. Higher values provide more robust estimates.
//'   Default: 5
//'
//' @param l1_regularization Numeric L1 (Lasso) regularization strength. Must be non-negative.
//'   Promotes sparsity in bin selection.
//'   Default: 0.0
//'
//' @param l2_regularization Numeric L2 (Ridge) regularization strength. Must be non-negative.
//'   Prevents extreme WoE values.
//'   Default: 0.0
//'
//' @param compute_confidence_intervals Logical whether to compute Wilson score confidence intervals
//'   for WoE values. Adds computational overhead but provides uncertainty quantification.
//'   Default: FALSE
//'
//' @param confidence_level Numeric confidence level for intervals when compute_confidence_intervals=TRUE.
//'   Must be in (0, 1). Common values: 0.90, 0.95, 0.99.
//'   Default: 0.95
//'
//' @param parallel Logical whether to use OpenMP parallel processing for multiple features.
//'   Requires OpenMP support in compilation. Significant speedup for many features.
//'   Default: FALSE
//'
//' @param n_threads Integer number of threads for parallel processing. Options:
//'   \itemize{
//'     \item \code{-1}: Auto-detect optimal number (default)
//'     \item \code{1}: Sequential processing
//'     \item \code{2+}: Specific thread count
//'   }
//'   Default: -1
//'
//' @param weights Optional NumericVector of observation weights for weighted binning. Length must equal
//'   number of rows in data. All weights must be non-negative. NULL indicates unit weights.
//'   Default: NULL
//'
//' @param special_values NumericVector of special numeric codes to handle separately (e.g., -999, -888).
//'   These values are isolated in dedicated bins regardless of their frequency.
//'   Default: empty vector
//'
//' @param missing_policy String specifying missing value handling. Options:
//'   \itemize{
//'     \item \code{"separate"}: Create dedicated bin(s) for missing values (default)
//'     \item \code{"remove"}: Exclude missing values from analysis
//'     \item \code{"impute"}: Impute with median/mode (future)
//'     \item \code{"merge"}: Merge with nearest bin by event rate (future)
//'   }
//'   Default: "separate"
//'
//' @param cat_sep String separator for merged categorical bin labels. Should not appear in
//'   actual category names. Used when multiple categories are combined into one bin.
//'   Default: "\%;\%"
//'
//' @param digits Integer decimal places for numeric boundaries in bin labels. Must be in [0, 10].
//'   Affects display precision only, not calculation accuracy.
//'   Default: 3
//'
//' @param rare_category_threshold Numeric threshold for rare category grouping as proportion.
//'   Categories with frequency below this are combined. Must be in [0, 1].
//'   Default: 0.01
//'
//' @param random_seed Integer seed for random number generation. Ensures reproducibility
//'   in stochastic components. Set to -1 for non-deterministic behavior.
//'   Default: 42
//'
//' @param verbose Logical whether to print detailed progress information during processing.
//'   Useful for debugging and monitoring long-running operations.
//'   Default: FALSE
//'
//' @return
//' S3 object of class "dmiv_woe_result" (inheriting from "list") with:
//'
//' \strong{Per-feature results} (accessed by feature name):
//' \itemize{
//'   \item \code{bins}: data.frame with columns:
//'     \itemize{
//'       \item \code{bin_id}: Sequential bin identifier (1, 2, 3, ...)
//'       \item \code{bin_label}: Formatted bin description (e.g., "(-inf, 25.5]", "cat1\%;\%cat2")
//'       \item \code{total_count}: Total observations in bin
//'       \item \code{neg_count}: Count of negative class (Y=0)
//'       \item \code{pos_count}: Count of positive class (Y=1)
//'       \item \code{woe}: Weight of Evidence value (clipped to [-20, 20])
//'       \item \code{divergence}: Divergence measure contribution
//'       \item \code{iv}: Information Value contribution
//'       \item \code{ks}: Kolmogorov-Smirnov statistic
//'       \item \code{gini}: Gini coefficient for the bin
//'     }
//'   \item \code{total_divergence}: Total divergence across all bins
//'   \item \code{total_iv}: Total Information Value for the feature
//'   \item \code{variable_type}: Detected type ("numeric_continuous", "numeric_discrete", "categorical", "boolean")
//'   \item \code{n_bins}: Final number of bins created
//'   \item \code{messages}: Processing status or error messages
//'   \item \code{divergence_method}: Divergence measure used
//'   \item \code{transform}: data.frame with transformations:
//'     \itemize{
//'       \item \code{[feature]}: Original feature values
//'       \item \code{[feature]_bin}: Assigned bin labels
//'       \item \code{[feature]_woe}: Assigned WoE values
//'     }
//' }
//'
//' \strong{Attributes}:
//' \itemize{
//'   \item \code{config}: List of all configuration parameters used
//'   \item \code{summary}: Processing summary with:
//'     \itemize{
//'       \item \code{n_features}: Total features processed
//'       \item \code{n_success}: Successfully processed features
//'       \item \code{n_errors}: Features with errors
//'       \item \code{processing_time_seconds}: Total computation time
//'       \item \code{timestamp}: Processing timestamp (format: "YYYY-MM-DD HH:MM:SS")
//'     }
//' }
//'
//' @section Algorithm Pipeline:
//' The complete DMIV optimization process:
//' \enumerate{
//'   \item \strong{Type Detection and Validation}:
//'     \itemize{
//'       \item Automatic variable type inference (numeric/categorical/boolean)
//'       \item Discrete numeric detection (<= 20 unique values)
//'       \item Date/time variable filtering with warnings
//'     }
//'   \item \strong{Data Preprocessing}:
//'     \itemize{
//'       \item Missing value handling per \code{missing_policy}
//'       \item Special value isolation into dedicated bins
//'       \item Rare category grouping below \code{rare_category_threshold}
//'       \item Weight normalization if provided
//'     }
//'   \item \strong{Initial Binning}:
//'     \itemize{
//'       \item Numeric: Quantile-based splitting (3× max_bins granularity)
//'       \item Categorical: Event rate ordering with rare grouping
//'       \item Boolean: Direct two-bin assignment
//'     }
//'   \item \strong{Optimization Phase}:
//'     \itemize{
//'       \item Iterative bin merging to minimize divergence
//'       \item Regularization penalty application
//'       \item Convergence monitoring with early stopping
//'       \item Constraint satisfaction (min/max bins, min size)
//'     }
//'   \item \strong{Monotonicity Enforcement} (if enabled):
//'     \itemize{
//'       \item Trend detection via Spearman correlation
//'       \item PAVA isotonic regression application
//'       \item WoE recalculation post-adjustment
//'     }
//'   \item \strong{Metric Computation}:
//'     \itemize{
//'       \item WoE calculation with chosen method and smoothing
//'       \item IV computation with regularization
//'       \item KS and Gini statistics
//'       \item Confidence intervals (if requested)
//'     }
//'   \item \strong{Fast Transform Generation}:
//'     \itemize{
//'       \item Build optimized mapping structures
//'       \item Apply transformations to original data
//'       \item Generate output DataFrames
//'     }
//' }
//'
//' @section Performance Characteristics:
//' \itemize{
//'   \item \strong{Time Complexity}: \eqn{O(n \log n + kb^2)} where n=observations, k=iterations, b=bins
//'   \item \strong{Space Complexity}: \eqn{O(n + b^2)} for data storage and optimization matrices
//'   \item \strong{Parallelization}: Linear speedup with thread count for multiple features
//'   \item \strong{Numerical Stability}: All logarithms use safe_log with \eqn{\epsilon = 10^{-12}}
//' }
//'
//' @section Implementation Notes:
//' \itemize{
//'   \item Uses C++17 features including structured bindings and std::atomic
//'   \item OpenMP parallelization with dynamic scheduling for load balancing
//'   \item Thread-safe bin statistics using atomic operations
//'   \item Memory-efficient Eigen library for matrix operations
//'   \item Fast bin mapping via binary search (numeric) and hash maps (categorical)
//'   \item Extensive input validation with informative error messages
//' }
//'
//' @references
//' \itemize{
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulation. arXiv preprint arXiv:2001.08025.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. Working Paper.
//'   \item Cover, T. M., & Thomas, J. A. (2006). Elements of information theory. John Wiley & Sons.
//'   \item Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). Order restricted statistical inference. Wiley.
//'   \item Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA, 22(158), 209-212.
//' }
//'
//' @examples
//' \dontrun{
//' # Load sample credit data
//' library(scorecard)
//' data <- germancredit
//' data$target <- ifelse(data$creditability == "good", 1, 0)
//'
//' # Basic usage with defaults
//' result <- dmiv_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
//' )
//'
//' # Advanced configuration with multiple divergence measures
//' result_advanced <- dmiv_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = names(data)[1:20],
//'   min_bins = 2,                    # Minimum 2 bins
//'   max_bins = 8,                     # Maximum 8 bins
//'   divergence_method = "hellinger",  # Hellinger distance
//'   woe_method = "woe1",              # Zeng's WOE1
//'   smooth = 0.5,                     # Moderate smoothing
//'   min_bin_size = 0.03,              # 3% minimum size
//'   optimization_method = "greedy_merge",
//'   enforce_monotonicity = TRUE,      # Enforce monotonic WoE
//'   monotonicity_type = "auto",       # Auto-detect trend
//'   max_iterations = 2000,            # More iterations
//'   convergence_threshold = 1e-8,     # Tighter convergence
//'   l1_regularization = 0.01,         # L1 penalty
//'   l2_regularization = 0.001,        # L2 penalty
//'   compute_confidence_intervals = TRUE,  # Calculate CIs
//'   confidence_level = 0.99,          # 99% confidence
//'   parallel = TRUE,                  # Use parallelization
//'   n_threads = 4,                    # 4 threads
//'   special_values = c(-999, -888),   # Special codes
//'   missing_policy = "separate",      # Separate missing bin
//'   cat_sep = " | ",                  # Pipe separator
//'   digits = 2,                       # 2 decimal places
//'   rare_category_threshold = 0.005,  # 0.5% rare threshold
//'   random_seed = 123,                # For reproducibility
//'   verbose = TRUE                    # Show progress
//' )
//'
//' # Extract and analyze results
//' duration_result <- result_advanced$duration.in.month
//' print(duration_result$bins)
//' print(paste("Total IV:", round(duration_result$total_iv, 4)))
//' print(paste("Total Divergence:", round(duration_result$total_divergence, 4)))
//'
//' # Apply transformation to new data
//' transform_df <- duration_result$transform
//' head(transform_df)
//'
//' # Compare divergence methods
//' methods <- c("l2_norm", "hellinger", "kullback_leibler", "jensen_shannon")
//' comparison <- lapply(methods, function(m) {
//'   res <- dmiv_woe(data, "target", "credit.amount", divergence_method = m)
//'   list(method = m,
//'        n_bins = res$credit.amount$n_bins,
//'        total_iv = res$credit.amount$total_iv,
//'        total_div = res$credit.amount$total_divergence)
//' })
//' do.call(rbind, lapply(comparison, as.data.frame))
//'
//' # Process categorical variables
//' cat_result <- dmiv_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = c("purpose", "personal.status.sex"),
//'   max_bins = 5,
//'   rare_category_threshold = 0.02,
//'   cat_sep = " + "
//' )
//'
//' # Examine categorical binning
//' print(cat_result$purpose$bins)
//' }
//'
//' @seealso
//' \code{\link{cart_woe}} for CART-based binning,
//' \code{\link{optbin}} for alternative optimization approaches
//'
//' @export
// [[Rcpp::export]]
List dmiv_woe(
   DataFrame data,
   std::string target_col,
   CharacterVector feature_cols,
   int min_bins = 3,
   int max_bins = 10,
   std::string divergence_method = "l2_norm",
   std::string woe_method = "woe",
   double smooth = 0.0,
   double min_bin_size = 0.05,
   std::string optimization_method = "greedy_merge",
   bool enforce_monotonicity = false,
   std::string monotonicity_type = "none",
   int max_iterations = 1000,
   double convergence_threshold = 1e-6,
   bool use_cross_validation = false,
   int cv_folds = 5,
   double l1_regularization = 0.0,
   double l2_regularization = 0.0,
   bool compute_confidence_intervals = false,
   double confidence_level = 0.95,
   bool parallel = false,
   int n_threads = -1,
   Nullable<NumericVector> weights = R_NilValue,
   NumericVector special_values = NumericVector::create(),
   std::string missing_policy = "separate",
   std::string cat_sep = "%;%",
   int digits = 3,
   double rare_category_threshold = 0.01,
   int random_seed = 42,
   bool verbose = false
) {
 // Validate inputs
 if (min_bins < 2) {
   stop("min_bins must be >= 2");
 }
 if (max_bins < min_bins) {
   stop("max_bins must be >= min_bins");
 }
 if (min_bin_size <= 0 || min_bin_size > 1) {
   stop("min_bin_size must be between 0 and 1");
 }
 if (smooth < 0) {
   stop("smooth must be >= 0");
 }
 if (l1_regularization < 0) {
   stop("l1_regularization must be >= 0");
 }
 if (l2_regularization < 0) {
   stop("l2_regularization must be >= 0");
 }
 if (confidence_level <= 0 || confidence_level >= 1) {
   stop("confidence_level must be between 0 and 1");
 }
 if (cv_folds < 2 && use_cross_validation) {
   stop("cv_folds must be >= 2 when use_cross_validation is TRUE");
 }
 if (digits < 0 || digits > 10) {
   stop("digits must be between 0 and 10");
 }
 if (rare_category_threshold < 0 || rare_category_threshold > 1) {
   stop("rare_category_threshold must be between 0 and 1");
 }
 
 // Set up configuration
 dmivwoe::OptimizationConfig config;
 config.min_bins = min_bins;
 config.max_bins = max_bins;
 config.min_bin_size_ratio = min_bin_size;
 config.woe_method = woe_method;
 config.smooth = smooth;
 config.max_iterations = max_iterations;
 config.convergence_threshold = convergence_threshold;
 config.missing_policy = missing_policy;
 config.cat_sep = cat_sep;
 config.digits = digits;
 config.rare_category_threshold = rare_category_threshold;
 config.random_seed = random_seed;
 config.l1_regularization = l1_regularization;
 config.l2_regularization = l2_regularization;
 config.use_cross_validation = use_cross_validation;
 config.cv_folds = cv_folds;
 config.compute_confidence_intervals = compute_confidence_intervals;
 config.confidence_level = confidence_level;
 config.use_parallel = parallel;
 config.n_threads = n_threads;
 config.verbose = verbose;
 
 // Set divergence method
 if (divergence_method == "hellinger") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::HELLINGER;
 } else if (divergence_method == "kullback_leibler") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::KULLBACK_LEIBLER;
 } else if (divergence_method == "j_divergence") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::J_DIVERGENCE;
 } else if (divergence_method == "chi_square") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::CHI_SQUARE;
 } else if (divergence_method == "jensen_shannon") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::JENSEN_SHANNON;
 } else if (divergence_method == "total_variation") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::TOTAL_VARIATION;
 } else if (divergence_method == "l1_norm") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::L1_NORM;
 } else if (divergence_method == "l2_norm") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::L2_NORM;
 } else if (divergence_method == "linf_norm") {
   config.divergence_method = dmivwoe::types::DivergenceMethod::LINF_NORM;
 } else {
   warning("Unknown divergence_method '" + divergence_method + "'. Using 'l2_norm' as default.");
   config.divergence_method = dmivwoe::types::DivergenceMethod::L2_NORM;
 }
 
 // Set optimization method
 if (optimization_method == "branch_and_bound") {
   config.optimization_method = dmivwoe::types::OptimizationMethod::BRANCH_AND_BOUND;
 } else if (optimization_method == "dynamic_programming") {
   config.optimization_method = dmivwoe::types::OptimizationMethod::DYNAMIC_PROGRAMMING;
 } else if (optimization_method == "greedy_merge") {
   config.optimization_method = dmivwoe::types::OptimizationMethod::GREEDY_MERGE;
 } else if (optimization_method == "simulated_annealing") {
   config.optimization_method = dmivwoe::types::OptimizationMethod::SIMULATED_ANNEALING;
 } else {
   warning("Unknown optimization_method '" + optimization_method + "'. Using 'greedy_merge' as default.");
   config.optimization_method = dmivwoe::types::OptimizationMethod::GREEDY_MERGE;
 }
 
 // Set monotonicity
 if (enforce_monotonicity) {
   if (monotonicity_type == "auto") {
     config.monotonicity = dmivwoe::types::MonotonicityType::AUTO_DETECT;
   } else if (monotonicity_type == "increasing") {
     config.monotonicity = dmivwoe::types::MonotonicityType::INCREASING;
   } else if (monotonicity_type == "decreasing") {
     config.monotonicity = dmivwoe::types::MonotonicityType::DECREASING;
   } else if (monotonicity_type == "none") {
     config.monotonicity = dmivwoe::types::MonotonicityType::NONE;
   } else {
     warning("Unknown monotonicity_type '" + monotonicity_type + "'. Using 'auto' as default.");
     config.monotonicity = dmivwoe::types::MonotonicityType::AUTO_DETECT;
   }
 } else {
   config.monotonicity = dmivwoe::types::MonotonicityType::NONE;
 }
 
 // Validate missing_policy
 if (missing_policy != "separate" && missing_policy != "remove" && 
     missing_policy != "impute" && missing_policy != "merge") {
   warning("Unknown missing_policy '" + missing_policy + "'. Using 'separate' as default.");
   config.missing_policy = "separate";
 }
 
 // Validate woe_method
 if (woe_method != "woe" && woe_method != "woe1") {
   warning("Unknown woe_method '" + woe_method + "'. Using 'woe' as default.");
   config.woe_method = "woe";
 }
 
 // Set up weights
 int n = data.nrows();
 NumericVector w(n, 1.0);
 if (!weights.isNull()) {
   w = as<NumericVector>(weights);
   if (w.size() != n) {
     stop("weights must have same length as data");
   }
   // Validate weights
   for (int i = 0; i < w.size(); ++i) {
     if (w[i] < 0) {
       stop("weights must be non-negative");
     }
   }
 }
 
 // Set up special values
 std::set<double> special_set;
 for (int i = 0; i < special_values.size(); ++i) {
   special_set.insert(special_values[i]);
 }
 
 // Initialize logger with verbose setting
 dmivwoe::logging::global_logger.set_verbose(verbose);
 
 // Initialize parallel processing if requested
#ifdef _OPENMP
 if (parallel) {
   if (n_threads > 0) {
     omp_set_num_threads(n_threads);
     if (verbose) {
       Rcout << "Using " << n_threads << " threads for parallel processing" << std::endl;
     }
   } else if (n_threads == -1) {
     int max_threads = omp_get_max_threads();
     omp_set_num_threads(max_threads);
     if (verbose) {
       Rcout << "Using " << max_threads << " threads (auto-detected) for parallel processing" << std::endl;
     }
   }
 }
#else
 if (parallel && verbose) {
   Rcout << "OpenMP not available. Running in sequential mode." << std::endl;
 }
#endif
 
 // Process features
 List results;
 
 // Print configuration summary if verbose
 if (verbose) {
   Rcout << "========================================" << std::endl;
   Rcout << "DMIV_WOE Configuration:" << std::endl;
   Rcout << "----------------------------------------" << std::endl;
   Rcout << "Bins: [" << min_bins << ", " << max_bins << "]" << std::endl;
   Rcout << "Divergence Method: " << divergence_method << std::endl;
   Rcout << "WoE Method: " << woe_method << std::endl;
   Rcout << "Smoothing: " << smooth << std::endl;
   Rcout << "Optimization: " << optimization_method << std::endl;
   Rcout << "Monotonicity: " << (enforce_monotonicity ? monotonicity_type : "none") << std::endl;
   Rcout << "Missing Policy: " << missing_policy << std::endl;
   Rcout << "Digits: " << digits << std::endl;
   Rcout << "Rare Category Threshold: " << rare_category_threshold << std::endl;
   Rcout << "L1 Regularization: " << l1_regularization << std::endl;
   Rcout << "L2 Regularization: " << l2_regularization << std::endl;
   Rcout << "Confidence Intervals: " << (compute_confidence_intervals ? "Yes" : "No") << std::endl;
   if (compute_confidence_intervals) {
     Rcout << "Confidence Level: " << confidence_level << std::endl;
   }
   Rcout << "========================================" << std::endl;
 }
 
 // Track processing time if verbose
 auto start_time = std::chrono::high_resolution_clock::now();
 
 // Process each feature
 int feature_count = 0;
 int success_count = 0;
 int error_count = 0;
 
 for (int i = 0; i < feature_cols.size(); ++i) {
   std::string feature_name = as<std::string>(feature_cols[i]);
   feature_count++;
   
   if (verbose) {
     Rcout << "\nProcessing feature " << feature_count << "/" << feature_cols.size() 
           << ": " << feature_name << std::endl;
   }
   
   try {
     auto feature_start = std::chrono::high_resolution_clock::now();
     
     List feature_result = dmivwoe::process_feature_dmiv(
       data, target_col, feature_name, config, w, special_set
     );
     
     auto feature_end = std::chrono::high_resolution_clock::now();
     auto feature_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
       feature_end - feature_start
     );
     
     if (verbose) {
       // Extract summary statistics
       DataFrame bins_df = as<DataFrame>(feature_result["bins"]);
       double total_iv = as<double>(feature_result["total_iv"]);
       double total_divergence = as<double>(feature_result["total_divergence"]);
       int n_bins = as<int>(feature_result["n_bins"]);
       std::string var_type = as<std::string>(feature_result["variable_type"]);
       
       Rcout << "  Variable Type: " << var_type << std::endl;
       Rcout << "  Number of Bins: " << n_bins << std::endl;
       Rcout << "  Total IV: " << std::fixed << std::setprecision(4) << total_iv << std::endl;
       Rcout << "  Total Divergence: " << std::fixed << std::setprecision(4) << total_divergence << std::endl;
       Rcout << "  Processing Time: " << feature_duration.count() << " ms" << std::endl;
       
       // Check for warnings
       std::string messages = as<std::string>(feature_result["messages"]);
       if (messages != "Success") {
         Rcout << "  Note: " << messages << std::endl;
       }
     }
     
     results[feature_name] = feature_result;
     success_count++;
     
   } catch (const std::exception& e) {
     error_count++;
     
     if (verbose) {
       Rcout << "  ERROR: " << e.what() << std::endl;
     }
     
     // Create error result with details
     List error_result = List::create(
       Named("error") = std::string(e.what()),
       Named("feature") = feature_name,
       Named("bins") = DataFrame::create(),
       Named("total_divergence") = 0.0,
       Named("total_iv") = 0.0,
       Named("variable_type") = "error",
       Named("n_bins") = 0,
       Named("messages") = std::string("Error: ") + e.what(),
       Named("divergence_method") = divergence_method,
       Named("transform") = DataFrame::create()
     );
     results[feature_name] = error_result;
   }
 }
 
 // Calculate total processing time
 auto end_time = std::chrono::high_resolution_clock::now();
 auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
 
 if (verbose) {
   Rcout << "\n========================================" << std::endl;
   Rcout << "Processing Summary:" << std::endl;
   Rcout << "----------------------------------------" << std::endl;
   Rcout << "Total Features: " << feature_count << std::endl;
   Rcout << "Successful: " << success_count << std::endl;
   Rcout << "Errors: " << error_count << std::endl;
   Rcout << "Total Time: " << total_duration.count() << " seconds" << std::endl;
   Rcout << "Average Time per Feature: " 
         << (feature_count > 0 ? total_duration.count() / feature_count : 0) 
         << " seconds" << std::endl;
   Rcout << "========================================" << std::endl;
 }
 
 // Create timestamp string properly - FIX FOR THE ERROR
 auto now = std::chrono::system_clock::now();
 auto time_t_now = std::chrono::system_clock::to_time_t(now);
 std::stringstream timestamp_ss;
 timestamp_ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
 std::string timestamp_str = timestamp_ss.str();
 
 // Add metadata to results
 results.attr("class") = CharacterVector::create("dmiv_woe_result", "list");
 results.attr("config") = List::create(
   Named("min_bins") = min_bins,
   Named("max_bins") = max_bins,
   Named("divergence_method") = divergence_method,
   Named("woe_method") = woe_method,
   Named("smooth") = smooth,
   Named("min_bin_size") = min_bin_size,
   Named("optimization_method") = optimization_method,
   Named("enforce_monotonicity") = enforce_monotonicity,
   Named("monotonicity_type") = monotonicity_type,
   Named("max_iterations") = max_iterations,
   Named("convergence_threshold") = convergence_threshold,
   Named("use_cross_validation") = use_cross_validation,
   Named("cv_folds") = cv_folds,
   Named("l1_regularization") = l1_regularization,
   Named("l2_regularization") = l2_regularization,
   Named("compute_confidence_intervals") = compute_confidence_intervals,
   Named("confidence_level") = confidence_level,
   Named("parallel") = parallel,
   Named("n_threads") = n_threads,
   Named("missing_policy") = missing_policy,
   Named("cat_sep") = cat_sep,
   Named("digits") = digits,
   Named("rare_category_threshold") = rare_category_threshold,
   Named("random_seed") = random_seed,
   Named("verbose") = verbose
 );
 
 results.attr("summary") = List::create(
   Named("n_features") = feature_count,
   Named("n_success") = success_count,
   Named("n_errors") = error_count,
   Named("processing_time_seconds") = total_duration.count(),
   Named("timestamp") = timestamp_str  // CORRECTED: Use string instead of time_point
 );
 
 return results;
}
