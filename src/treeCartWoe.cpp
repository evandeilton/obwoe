// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace cartwoe {

// =============================================================================
// CONSTANTS & CONFIGURATION
// =============================================================================

constexpr double EPSILON = 1e-10;
constexpr double WOE_MIN = -20.0;
constexpr double WOE_MAX = 20.0;
constexpr double IMPURITY_THRESHOLD = 1e-8;
constexpr int MAX_TREE_DEPTH = 20;
constexpr int PARALLEL_THRESHOLD = 10000;

// =============================================================================
// LOGGER SYSTEM
// =============================================================================

class Logger {
private:
  int verbose_level_;
  std::string current_feature_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point feature_start_time_;
  bool use_color_;
  
  // ANSI color codes
  const std::string RESET = "\033[0m";
  const std::string BOLD = "\033[1m";
  const std::string RED = "\033[31m";
  const std::string GREEN = "\033[32m";
  const std::string YELLOW = "\033[33m";
  const std::string BLUE = "\033[34m";
  const std::string MAGENTA = "\033[35m";
  const std::string CYAN = "\033[36m";
  const std::string WHITE = "\033[37m";
  
public:
  Logger(int verbose_level = 0, bool use_color = true) 
    : verbose_level_(verbose_level), use_color_(use_color) {
    start_time_ = std::chrono::steady_clock::now();
  }
  
  void set_feature(const std::string& feature) {
    current_feature_ = feature;
    feature_start_time_ = std::chrono::steady_clock::now();
  }
  
  void set_verbose(int level) {
    verbose_level_ = level;
  }
  
  std::string get_timestamp() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    std::stringstream ss;
    ss << "[" << std::setfill('0') << std::setw(8) 
       << duration.count() << "ms]";
    return ss.str();
  }
  
  std::string get_feature_time() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - feature_start_time_);
    return std::to_string(duration.count()) + "ms";
  }
  
  void log(int level, const std::string& message, const std::string& color = "") {
    if (verbose_level_ >= level) {
      std::string prefix = get_timestamp();
      if (!current_feature_.empty()) {
        prefix += " [" + current_feature_ + "]";
      }
      
      std::string colored_msg = message;
      if (use_color_ && !color.empty()) {
        colored_msg = color + message + RESET;
      }
      
      Rcout << prefix << " " << colored_msg << std::endl;
    }
  }
  
  // Convenience methods for different log levels
  void info(const std::string& msg) {
    log(1, msg, CYAN);
  }
  
  void detail(const std::string& msg) {
    log(2, msg, WHITE);
  }
  
  void debug(const std::string& msg) {
    log(3, msg, YELLOW);
  }
  
  void success(const std::string& msg) {
    log(1, msg, GREEN);
  }
  
  void warning(const std::string& msg) {
    log(1, "WARNING: " + msg, YELLOW);
  }
  
  void error(const std::string& msg) {
    log(1, "ERROR: " + msg, RED);
  }
  
  void header(const std::string& msg) {
    if (verbose_level_ >= 1) {
      std::string line(60, '=');
      log(1, line, BLUE);
      log(1, msg, BOLD + BLUE);
      log(1, line, BLUE);
    }
  }
  
  void subheader(const std::string& msg) {
    if (verbose_level_ >= 2) {
      std::string line(40, '-');
      log(2, line, CYAN);
      log(2, msg, BOLD + CYAN);
    }
  }
  
  void progress(int current, int total, const std::string& task = "") {
    if (verbose_level_ >= 2) {
      std::stringstream ss;
      ss << "Progress: " << current << "/" << total;
      if (!task.empty()) {
        ss << " - " << task;
      }
      ss << " (" << std::fixed << std::setprecision(1) 
         << (100.0 * current / total) << "%)";
      log(2, ss.str(), MAGENTA);
    }
  }
  
  void metrics(const std::string& name, double value, int precision = 4) {
    if (verbose_level_ >= 2) {
      std::stringstream ss;
      ss << name << ": " << std::fixed << std::setprecision(precision) << value;
      log(2, ss.str(), GREEN);
    }
  }
  
  void tree_node(int depth, int id, int samples, double impurity, double gain = -1) {
    if (verbose_level_ >= 3) {
      std::string indent(depth * 2, ' ');
      std::stringstream ss;
      ss << indent << "Node " << id << ": samples=" << samples 
         << ", impurity=" << std::fixed << std::setprecision(4) << impurity;
      if (gain >= 0) {
        ss << ", gain=" << std::fixed << std::setprecision(4) << gain;
      }
      log(3, ss.str());
    }
  }
  
  void bin_info(int id, const std::string& label, int count, double woe, double iv) {
    if (verbose_level_ >= 2) {
      std::stringstream ss;
      ss << "Bin " << id << " [" << label << "]: "
         << "n=" << count 
         << ", WoE=" << std::fixed << std::setprecision(4) << woe
         << ", IV=" << std::fixed << std::setprecision(4) << iv;
      log(2, ss.str());
    }
  }
};

// Global logger instance
static Logger g_logger;

// =============================================================================
// ENUMS & TYPE DEFINITIONS
// =============================================================================

enum class VariableType {
  NUMERIC_CONTINUOUS,
  NUMERIC_DISCRETE,
  CATEGORICAL,
  BOOLEAN,
  DATE_TIME,
  UNSUPPORTED
};

enum class Criterion {
  GINI,
  ENTROPY
};

enum class MissPolicy {
  SEPARATE,
  REMOVE,
  IMPUTE,
  MERGE
};

enum class MonotonicTrend {
  INCREASING,
  DECREASING,
  AUTO,
  NONE
};

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct DataPoint {
  double value;
  double target;
  double weight;
  bool is_missing;
  bool is_special;
  double special_value;
  int original_index;
  std::string category_name;
  
  DataPoint() : value(0), target(0), weight(1), is_missing(false), 
  is_special(false), special_value(0), original_index(-1) {}
};

struct BinInfo {
  int id;
  std::string label;
  double lower_bound;
  double upper_bound;
  int total_count;
  int pos_count;
  int neg_count;
  double weight_sum;
  double pos_weight;
  double neg_weight;
  double woe;
  double iv;
  double ks;
  double gini;
  std::vector<std::string> categories;
  
  BinInfo() : id(0), lower_bound(-INFINITY), upper_bound(INFINITY),
  total_count(0), pos_count(0), neg_count(0), weight_sum(0),
  pos_weight(0), neg_weight(0), woe(0), iv(0), ks(0), gini(0) {}
};

struct TreeNode {
  int id;
  int depth;
  bool is_leaf;
  double threshold;
  double impurity;
  double gain;
  int n_samples;
  int n_pos;
  int n_neg;
  double weight_sum;
  double pos_weight;
  double neg_weight;
  std::vector<int> sample_indices;
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;
  
  TreeNode() : id(0), depth(0), is_leaf(true), threshold(0), impurity(0),
  gain(0), n_samples(0), n_pos(0), n_neg(0), weight_sum(0),
  pos_weight(0), neg_weight(0) {}
};

struct SplitCandidate {
  double threshold;
  double gain;
  double impurity_left;
  double impurity_right;
  int n_left;
  int n_right;
  double weight_left;
  double weight_right;
  bool valid;
  
  SplitCandidate() : threshold(0), gain(-1), impurity_left(0), impurity_right(0),
  n_left(0), n_right(0), weight_left(0), weight_right(0), valid(false) {}
};

// =============================================================================
// FAST BIN MAPPER FOR EFFICIENT TRANSFORM
// =============================================================================

class FastBinMapper {
private:
  VariableType var_type_;
  
  // For numeric variables
  std::vector<double> thresholds_;
  std::vector<int> bin_ids_;
  std::vector<double> woe_values_;
  std::vector<std::string> bin_labels_;
  
  // For categorical variables
  std::unordered_map<std::string, int> category_to_bin_;
  std::unordered_map<std::string, double> category_to_woe_;
  std::unordered_map<std::string, std::string> category_to_label_;
  
  // For special values
  std::unordered_map<double, int> special_to_bin_;
  std::unordered_map<double, double> special_to_woe_;
  std::unordered_map<double, std::string> special_to_label_;
  
  int missing_bin_id_;
  double missing_woe_;
  std::string missing_label_;
  bool has_missing_;
  
public:
  FastBinMapper(VariableType var_type) : var_type_(var_type), 
  missing_bin_id_(-1), 
  missing_woe_(0), 
  has_missing_(false) {}
  
  void build_from_bins(const std::vector<BinInfo>& bins) {
    // Reserve space
    thresholds_.reserve(bins.size());
    bin_ids_.reserve(bins.size());
    woe_values_.reserve(bins.size());
    bin_labels_.reserve(bins.size());
    
    for (const auto& bin : bins) {
      // Handle special bins
      if (bin.label == "[Missing]") {
        has_missing_ = true;
        missing_bin_id_ = bin.id;
        missing_woe_ = bin.woe;
        missing_label_ = bin.label;
        continue;
      }
      
      if (bin.label.find("[Special:") == 0) {
        // Parse special value
        std::string val_str = bin.label.substr(9, bin.label.length() - 10);
        double special_val = std::stod(val_str);
        special_to_bin_[special_val] = bin.id;
        special_to_woe_[special_val] = bin.woe;
        special_to_label_[special_val] = bin.label;
        continue;
      }
      
      // Regular bins
      if (var_type_ == VariableType::CATEGORICAL) {
        // Map each category to its bin
        for (const auto& cat : bin.categories) {
          category_to_bin_[cat] = bin.id;
          category_to_woe_[cat] = bin.woe;
          category_to_label_[cat] = bin.label;
        }
      } else {
        // For numeric, store thresholds
        thresholds_.push_back(bin.upper_bound);
        bin_ids_.push_back(bin.id);
        woe_values_.push_back(bin.woe);
        bin_labels_.push_back(bin.label);
      }
    }
  }
  
  inline void map_numeric(double value, bool is_missing, bool is_special,
                          std::string& bin_label, double& woe) const {
    if (is_missing && has_missing_) {
      bin_label = missing_label_;
      woe = missing_woe_;
      return;
    }
    
    if (is_special) {
      auto it = special_to_woe_.find(value);
      if (it != special_to_woe_.end()) {
        woe = it->second;
        bin_label = special_to_label_.at(value);
        return;
      }
    }
    
    // Binary search for the right bin
    auto it = std::lower_bound(thresholds_.begin(), thresholds_.end(), value);
    size_t idx = std::distance(thresholds_.begin(), it);
    
    if (idx < woe_values_.size()) {
      woe = woe_values_[idx];
      bin_label = bin_labels_[idx];
    } else if (!woe_values_.empty()) {
      woe = woe_values_.back();
      bin_label = bin_labels_.back();
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
// UTILITY FUNCTIONS
// =============================================================================

inline double safe_log(double x) {
  return std::log(std::max(x, EPSILON));
}

inline double clip_value(double x, double min_val, double max_val) {
  return std::max(min_val, std::min(max_val, x));
}

VariableType detect_variable_type(SEXP x) {
  if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") || 
      Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
    return VariableType::DATE_TIME;
  }
  
  if (Rf_isFactor(x)) {
    return VariableType::CATEGORICAL;
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
    return (unique_vals.size() <= 20) ? 
    VariableType::NUMERIC_DISCRETE : 
      VariableType::NUMERIC_CONTINUOUS;
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
    return (unique_vals.size() <= 20) ? 
    VariableType::NUMERIC_DISCRETE : 
      VariableType::NUMERIC_CONTINUOUS;
  }
  case LGLSXP:
    return VariableType::BOOLEAN;
  case STRSXP:
    return VariableType::CATEGORICAL;
  default:
    return VariableType::UNSUPPORTED;
  }
}

std::string variable_type_to_string(VariableType vt) {
  switch (vt) {
  case VariableType::NUMERIC_CONTINUOUS: return "numeric_continuous";
  case VariableType::NUMERIC_DISCRETE: return "numeric_discrete";
  case VariableType::CATEGORICAL: return "categorical";
  case VariableType::BOOLEAN: return "boolean";
  case VariableType::DATE_TIME: return "date_time";
  default: return "unsupported";
  }
}

// =============================================================================
// IMPURITY CALCULATIONS WITH VECTORIZATION
// =============================================================================

class ImpurityCalculator {
public:
  static double calculate(Criterion criterion, double pos_weight, 
                          double neg_weight, double total_weight) {
    if (total_weight <= EPSILON) return 0.0;
    
    double p1 = pos_weight / total_weight;
    double p0 = neg_weight / total_weight;
    
    switch (criterion) {
    case Criterion::GINI:
      return 1.0 - p1 * p1 - p0 * p0;
    case Criterion::ENTROPY:
      return -(p1 > EPSILON ? p1 * safe_log(p1) : 0.0) -
        (p0 > EPSILON ? p0 * safe_log(p0) : 0.0);
    default:
      return 0.0;
    }
  }
  
  static double calculate_gain(Criterion criterion, 
                               double parent_impurity,
                               double parent_weight,
                               double left_impurity,
                               double left_weight,
                               double right_impurity,
                               double right_weight) {
    if (parent_weight <= EPSILON) return 0.0;
    
    double weighted_impurity = (left_weight / parent_weight) * left_impurity +
      (right_weight / parent_weight) * right_impurity;
    
    return parent_impurity - weighted_impurity;
  }
};

// =============================================================================
// ISOTONIC REGRESSION (PAVA)
// =============================================================================

class IsotonicRegression {
public:
  static std::vector<double> fit(const std::vector<double>& y,
                                 const std::vector<double>& weights,
                                 bool increasing = true) {
    int n = y.size();
    if (n == 0) return {};
    
    g_logger.debug("Starting isotonic regression with " + std::to_string(n) + 
      " points, direction: " + (increasing ? "increasing" : "decreasing"));
    
    std::vector<double> result(n);
    std::vector<std::pair<double, double>> blocks;
    blocks.reserve(n);
    
    for (int i = 0; i < n; ++i) {
      double val = increasing ? y[i] : -y[i];
      blocks.push_back({val * weights[i], weights[i]});
      
      while (blocks.size() >= 2) {
        int last = blocks.size() - 1;
        double mean_last = blocks[last].first / blocks[last].second;
        double mean_prev = blocks[last-1].first / blocks[last-1].second;
        
        if (mean_prev <= mean_last + EPSILON) break;
        
        blocks[last-1].first += blocks[last].first;
        blocks[last-1].second += blocks[last].second;
        blocks.pop_back();
      }
    }
    
    int idx = 0;
    for (const auto& block : blocks) {
      double mean = block.first / block.second;
      if (!increasing) mean = -mean;
      
      int block_size = std::round(block.second / weights[idx]);
      for (int j = 0; j < block_size && idx < n; ++j, ++idx) {
        result[idx] = mean;
      }
    }
    
    g_logger.debug("Isotonic regression completed with " + 
      std::to_string(blocks.size()) + " blocks");
    
    return result;
  }
};

// =============================================================================
// OPTIMAL BINNING VIA CART
// =============================================================================

class OptimalBinningCART {
private:
  std::vector<DataPoint> data_;
  Criterion criterion_;
  double min_size_ratio_;
  int min_size_count_;
  int min_bins_;
  int max_bins_;
  bool use_pruning_;
  int cv_folds_;
  MonotonicTrend monotonic_trend_;
  VariableType var_type_;
  std::string cat_sep_;
  std::unique_ptr<TreeNode> root_;
  std::vector<BinInfo> bins_;
  std::unordered_map<double, std::string> category_map_;
  std::vector<std::pair<double, std::string>> ordered_categories_;
  bool use_parallel_;
  int unique_classes_count_;
  
public:
  OptimalBinningCART(std::vector<DataPoint>&& data,
                     Criterion criterion,
                     double min_size,
                     int min_bins,
                     int max_bins,
                     bool use_pruning,
                     int cv_folds,
                     MonotonicTrend monotonic_trend,
                     VariableType var_type,
                     const std::string& cat_sep,
                     int unique_classes_count)
    : data_(std::move(data)), criterion_(criterion), min_bins_(min_bins),
      max_bins_(max_bins), use_pruning_(use_pruning), cv_folds_(cv_folds),
      monotonic_trend_(monotonic_trend), var_type_(var_type), cat_sep_(cat_sep),
      unique_classes_count_(unique_classes_count) {
    
    double total_weight = 0;
    int total_count = 0;
    
    use_parallel_ = data_.size() > PARALLEL_THRESHOLD;
    
    if (use_parallel_) {
      g_logger.debug("Using parallel processing (data size > " + 
        std::to_string(PARALLEL_THRESHOLD) + ")");
#pragma omp parallel for reduction(+:total_weight,total_count)
      for (size_t i = 0; i < data_.size(); ++i) {
        if (!data_[i].is_missing && !data_[i].is_special) {
          total_weight += data_[i].weight;
          total_count++;
        }
      }
    } else {
      for (const auto& dp : data_) {
        if (!dp.is_missing && !dp.is_special) {
          total_weight += dp.weight;
          total_count++;
        }
      }
    }
    
    if (min_size <= 1.0) {
      min_size_ratio_ = min_size;
      min_size_count_ = std::max(1, static_cast<int>(min_size * total_count));
    } else {
      min_size_ratio_ = min_size / total_count;
      min_size_count_ = static_cast<int>(min_size);
    }
    
    g_logger.debug("Min size count: " + std::to_string(min_size_count_) + 
      ", Min size ratio: " + std::to_string(min_size_ratio_));
    
    bins_.reserve(max_bins_ + 2);
  }
  
  void fit() {
    if (data_.empty()) {
      g_logger.warning("No data to fit");
      return;
    }
    
    g_logger.detail("Starting CART binning fit");
    
    // If variable has 2 or fewer unique classes, create direct bins without optimization
    if (unique_classes_count_ <= 2) {
      g_logger.info("Variable has " + std::to_string(unique_classes_count_) + 
        " unique classes. Using direct binning.");
      create_direct_bins();
      return;
    }
    
    prepare_data();
    
    std::set<double> unique_values;
    
    for (const auto& dp : data_) {
      if (!dp.is_missing && !dp.is_special) {
        unique_values.insert(dp.value);
        if (unique_values.size() > static_cast<size_t>(max_bins_)) break;
      }
    }
    
    g_logger.detail("Found " + std::to_string(unique_values.size()) + " unique values");
    
    int actual_min_bins = std::min(min_bins_, static_cast<int>(unique_values.size()));
    int actual_max_bins = std::min(max_bins_, static_cast<int>(unique_values.size()));
    
    g_logger.detail("Adjusted bin range: [" + std::to_string(actual_min_bins) + 
      ", " + std::to_string(actual_max_bins) + "]");
    
    g_logger.subheader("Building CART Tree");
    root_ = build_tree(actual_max_bins);
    
    if (use_pruning_ && root_) {
      g_logger.subheader("Pruning Tree");
      prune_tree();
    }
    
    g_logger.subheader("Extracting Bins from Tree");
    extract_bins_from_tree();
    g_logger.detail("Extracted " + std::to_string(bins_.size()) + " initial bins");
    
    int attempts = 0;
    while (static_cast<int>(bins_.size()) < actual_min_bins && attempts < 10) {
      g_logger.debug("Forcing split to reach minimum bins (attempt " + 
        std::to_string(attempts+1) + ")");
      if (!force_split_largest_bin()) break;
      attempts++;
    }
    
    if (monotonic_trend_ != MonotonicTrend::NONE && bins_.size() > 1) {
      g_logger.subheader("Enforcing Monotonicity");
      enforce_monotonicity();
      g_logger.detail("Bins after monotonicity: " + std::to_string(bins_.size()));
    }
    
    attempts = 0;
    while (static_cast<int>(bins_.size()) < actual_min_bins && attempts < 5) {
      g_logger.debug("Additional split attempt after monotonicity (attempt " + 
        std::to_string(attempts+1) + ")");
      if (!force_split_largest_bin()) break;
      attempts++;
    }
    
    while (static_cast<int>(bins_.size()) > actual_max_bins && bins_.size() > 1) {
      g_logger.debug("Merging bins to respect maximum limit");
      merge_adjacent_bins_with_min_iv_loss();
    }
    
    g_logger.success("Final bin count: " + std::to_string(bins_.size()));
  }
  
  std::vector<BinInfo> get_bins() { return std::move(bins_); }
  std::vector<std::pair<double, std::string>> get_ordered_categories() const { 
    return ordered_categories_; 
  }
  
private:
  void create_direct_bins() {
    bins_.clear();
    
    if (var_type_ == VariableType::CATEGORICAL || var_type_ == VariableType::BOOLEAN) {
      std::unordered_map<std::string, std::pair<int, std::pair<int, int>>> category_stats;
      
      for (const auto& dp : data_) {
        if (!dp.is_missing && !dp.is_special) {
          auto& stats = category_stats[dp.category_name];
          stats.first++;
          if (dp.target > 0.5) {
            stats.second.second++;
          } else {
            stats.second.first++;
          }
        }
      }
      
      int bin_id = 1;
      for (const auto& [cat, stats] : category_stats) {
        BinInfo bin;
        bin.id = bin_id++;
        bin.categories.push_back(cat);
        bin.label = cat;
        bin.total_count = stats.first;
        bin.neg_count = stats.second.first;
        bin.pos_count = stats.second.second;
        bin.weight_sum = stats.first;
        bin.pos_weight = stats.second.second;
        bin.neg_weight = stats.second.first;
        bins_.push_back(std::move(bin));
      }
    } else {
      std::map<double, std::pair<int, std::pair<int, int>>> value_stats;
      
      for (const auto& dp : data_) {
        if (!dp.is_missing && !dp.is_special) {
          auto& stats = value_stats[dp.value];
          stats.first++;
          if (dp.target > 0.5) {
            stats.second.second++;
          } else {
            stats.second.first++;
          }
        }
      }
      
      int bin_id = 1;
      for (const auto& [val, stats] : value_stats) {
        BinInfo bin;
        bin.id = bin_id++;
        bin.lower_bound = val;
        bin.upper_bound = val;
        bin.label = std::to_string(val);
        bin.total_count = stats.first;
        bin.neg_count = stats.second.first;
        bin.pos_count = stats.second.second;
        bin.weight_sum = stats.first;
        bin.pos_weight = stats.second.second;
        bin.neg_weight = stats.second.first;
        bins_.push_back(std::move(bin));
      }
    }
    
    g_logger.detail("Created " + std::to_string(bins_.size()) + " direct bins");
  }
  
  void prepare_data() {
    g_logger.detail("Preparing data for binning");
    
    if (var_type_ == VariableType::CATEGORICAL) {
      prepare_categorical_data();
    } else {
      g_logger.debug("Sorting numeric data");
      std::sort(data_.begin(), data_.end(),
                [](const DataPoint& a, const DataPoint& b) {
                  if (a.is_missing != b.is_missing) return b.is_missing;
                  if (a.is_special != b.is_special) return b.is_special;
                  return a.value < b.value;
                });
    }
  }
  
  void prepare_categorical_data() {
    g_logger.detail("Preparing categorical data");
    
    std::unordered_map<std::string, std::pair<double, double>> category_stats;
    category_stats.reserve(100);
    
    for (const auto& dp : data_) {
      if (!dp.is_missing && !dp.is_special) {
        auto& stats = category_stats[dp.category_name];
        stats.first += dp.target * dp.weight;
        stats.second += dp.weight;
      }
    }
    
    g_logger.debug("Found " + std::to_string(category_stats.size()) + " categories");
    
    ordered_categories_.clear();
    ordered_categories_.reserve(category_stats.size());
    
    for (const auto& [cat, stats] : category_stats) {
      double event_rate = stats.first / (stats.second + EPSILON);
      ordered_categories_.push_back({event_rate, cat});
    }
    
    std::sort(ordered_categories_.begin(), ordered_categories_.end());
    
    g_logger.debug("Categories ordered by event rate");
    
    category_map_.clear();
    category_map_.reserve(ordered_categories_.size());
    
    for (size_t i = 0; i < ordered_categories_.size(); ++i) {
      category_map_[static_cast<double>(i)] = ordered_categories_[i].second;
    }
    
    if (use_parallel_) {
#pragma omp parallel for
      for (size_t j = 0; j < data_.size(); ++j) {
        if (!data_[j].is_missing && !data_[j].is_special) {
          for (size_t i = 0; i < ordered_categories_.size(); ++i) {
            if (ordered_categories_[i].second == data_[j].category_name) {
              data_[j].value = static_cast<double>(i);
              break;
            }
          }
        }
      }
    } else {
      for (auto& dp : data_) {
        if (!dp.is_missing && !dp.is_special) {
          for (size_t i = 0; i < ordered_categories_.size(); ++i) {
            if (ordered_categories_[i].second == dp.category_name) {
              dp.value = static_cast<double>(i);
              break;
            }
          }
        }
      }
    }
    
    std::sort(data_.begin(), data_.end(),
              [](const DataPoint& a, const DataPoint& b) {
                if (a.is_missing != b.is_missing) return b.is_missing;
                if (a.is_special != b.is_special) return b.is_special;
                return a.value < b.value;
              });
  }
  
  std::unique_ptr<TreeNode> build_tree(int target_leaves) {
    g_logger.detail("Building tree with target " + std::to_string(target_leaves) + " leaves");
    
    auto root = std::make_unique<TreeNode>();
    root->id = 0;
    root->depth = 0;
    
    root->sample_indices.reserve(data_.size());
    
    for (size_t i = 0; i < data_.size(); ++i) {
      if (!data_[i].is_missing && !data_[i].is_special) {
        root->sample_indices.push_back(i);
      }
    }
    
    if (root->sample_indices.empty()) {
      g_logger.warning("No valid samples for tree building");
      return root;
    }
    
    update_node_statistics(root.get());
    g_logger.tree_node(0, root->id, root->n_samples, root->impurity);
    
    if (root->sample_indices.size() < 2 * min_size_count_) {
      g_logger.debug("Root has insufficient samples for splitting");
      return root;
    }
    
    using NodePair = std::pair<double, TreeNode*>;
    std::priority_queue<NodePair> expansion_queue;
    
    auto best_split = find_best_split(root.get());
    if (best_split.valid && best_split.gain > IMPURITY_THRESHOLD) {
      root->gain = best_split.gain;
      expansion_queue.push({root->gain, root.get()});
      g_logger.debug("Root node added to expansion queue with gain: " + 
        std::to_string(root->gain));
    }
    
    int node_count = 1;
    int leaf_count = 1;
    
    while (!expansion_queue.empty() && leaf_count < target_leaves) {
      auto [gain, node] = expansion_queue.top();
      expansion_queue.pop();
      
      if (gain <= IMPURITY_THRESHOLD) {
        g_logger.debug("Stopping expansion: gain too small");
        break;
      }
      
      auto split = find_best_split(node);
      if (!split.valid) continue;
      
      node->is_leaf = false;
      node->threshold = split.threshold;
      node->gain = split.gain;
      
      node->left = std::make_unique<TreeNode>();
      node->right = std::make_unique<TreeNode>();
      
      node->left->id = node_count++;
      node->right->id = node_count++;
      node->left->depth = node->depth + 1;
      node->right->depth = node->depth + 1;
      
      node->left->sample_indices.reserve(split.n_left);
      node->right->sample_indices.reserve(split.n_right);
      
      for (int idx : node->sample_indices) {
        if (data_[idx].value < node->threshold) {
          node->left->sample_indices.push_back(idx);
        } else {
          node->right->sample_indices.push_back(idx);
        }
      }
      
      update_node_statistics(node->left.get());
      update_node_statistics(node->right.get());
      
      g_logger.tree_node(node->left->depth, node->left->id, 
                         node->left->n_samples, node->left->impurity);
      g_logger.tree_node(node->right->depth, node->right->id, 
                         node->right->n_samples, node->right->impurity);
      
      leaf_count++;
      g_logger.progress(leaf_count, target_leaves, "Building tree");
      
      if (leaf_count < target_leaves) {
        auto left_split = find_best_split(node->left.get());
        if (left_split.valid && left_split.gain > IMPURITY_THRESHOLD) {
          node->left->gain = left_split.gain;
          expansion_queue.push({left_split.gain, node->left.get()});
        }
        
        auto right_split = find_best_split(node->right.get());
        if (right_split.valid && right_split.gain > IMPURITY_THRESHOLD) {
          node->right->gain = right_split.gain;
          expansion_queue.push({right_split.gain, node->right.get()});
        }
      }
    }
    
    g_logger.detail("Tree construction complete with " + std::to_string(leaf_count) + " leaves");
    
    return root;
  }
  
  void update_node_statistics(TreeNode* node) {
    node->n_samples = node->sample_indices.size();
    node->n_pos = 0;
    node->n_neg = 0;
    node->weight_sum = 0;
    node->pos_weight = 0;
    node->neg_weight = 0;
    
    for (int idx : node->sample_indices) {
      const auto& dp = data_[idx];
      node->weight_sum += dp.weight;
      
      if (dp.target > 0.5) {
        node->n_pos++;
        node->pos_weight += dp.weight;
      } else {
        node->n_neg++;
        node->neg_weight += dp.weight;
      }
    }
    
    node->impurity = ImpurityCalculator::calculate(
      criterion_, node->pos_weight, node->neg_weight, node->weight_sum);
  }
  
  SplitCandidate find_best_split(TreeNode* node) {
    SplitCandidate best_split;
    best_split.valid = false;
    
    if (!node || node->sample_indices.size() < 2 * min_size_count_) {
      return best_split;
    }
    
    std::vector<int> sorted_indices = node->sample_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](int a, int b) { return data_[a].value < data_[b].value; });
    
    size_t n = sorted_indices.size();
    std::vector<double> values(n);
    std::vector<double> cum_pos(n + 1, 0);
    std::vector<double> cum_neg(n + 1, 0);
    std::vector<double> cum_weight(n + 1, 0);
    
    for (size_t i = 0; i < n; ++i) {
      const auto& dp = data_[sorted_indices[i]];
      values[i] = dp.value;
      
      cum_weight[i + 1] = cum_weight[i] + dp.weight;
      if (dp.target > 0.5) {
        cum_pos[i + 1] = cum_pos[i] + dp.weight;
        cum_neg[i + 1] = cum_neg[i];
      } else {
        cum_pos[i + 1] = cum_pos[i];
        cum_neg[i + 1] = cum_neg[i] + dp.weight;
      }
    }
    
    best_split.gain = -1;
    int splits_evaluated = 0;
    
    if (use_parallel_ && n > 1000) {
#ifdef _OPENMP
      std::vector<SplitCandidate> thread_best_splits(omp_get_max_threads());
      
#pragma omp parallel
{
  int tid = omp_get_thread_num();
  thread_best_splits[tid].gain = -1;
  
#pragma omp for schedule(dynamic, 100)
  for (size_t i = min_size_count_; i <= n - min_size_count_; ++i) {
    if (i > 0 && std::abs(values[i] - values[i-1]) < EPSILON) continue;
    
    double left_weight = cum_weight[i];
    double right_weight = cum_weight.back() - left_weight;
    
    if (left_weight < min_size_ratio_ * node->weight_sum) continue;
    if (right_weight < min_size_ratio_ * node->weight_sum) continue;
    
    double left_pos = cum_pos[i];
    double left_neg = cum_neg[i];
    double right_pos = cum_pos.back() - left_pos;
    double right_neg = cum_neg.back() - left_neg;
    
    double left_impurity = ImpurityCalculator::calculate(
      criterion_, left_pos, left_neg, left_weight);
    double right_impurity = ImpurityCalculator::calculate(
      criterion_, right_pos, right_neg, right_weight);
    
    double gain = ImpurityCalculator::calculate_gain(
      criterion_, node->impurity, node->weight_sum,
      left_impurity, left_weight, right_impurity, right_weight);
    
    if (gain > thread_best_splits[tid].gain) {
      thread_best_splits[tid].gain = gain;
      thread_best_splits[tid].threshold = (var_type_ == VariableType::CATEGORICAL) ?
      values[i-1] + 0.5 : (i > 0 ? (values[i-1] + values[i]) / 2.0 : values[i]);
      thread_best_splits[tid].impurity_left = left_impurity;
      thread_best_splits[tid].impurity_right = right_impurity;
      thread_best_splits[tid].n_left = i;
      thread_best_splits[tid].n_right = n - i;
      thread_best_splits[tid].weight_left = left_weight;
      thread_best_splits[tid].weight_right = right_weight;
      thread_best_splits[tid].valid = true;
    }
  }
}

for (const auto& split : thread_best_splits) {
  if (split.gain > best_split.gain) {
    best_split = split;
  }
}
#endif
    } else {
      for (size_t i = min_size_count_; i <= n - min_size_count_; ++i) {
        if (i > 0 && std::abs(values[i] - values[i-1]) < EPSILON) continue;
        
        double left_weight = cum_weight[i];
        double right_weight = cum_weight.back() - left_weight;
        
        if (left_weight < min_size_ratio_ * node->weight_sum) continue;
        if (right_weight < min_size_ratio_ * node->weight_sum) continue;
        
        splits_evaluated++;
        
        double left_pos = cum_pos[i];
        double left_neg = cum_neg[i];
        double right_pos = cum_pos.back() - left_pos;
        double right_neg = cum_neg.back() - left_neg;
        
        double left_impurity = ImpurityCalculator::calculate(
          criterion_, left_pos, left_neg, left_weight);
        double right_impurity = ImpurityCalculator::calculate(
          criterion_, right_pos, right_neg, right_weight);
        
        double gain = ImpurityCalculator::calculate_gain(
          criterion_, node->impurity, node->weight_sum,
          left_impurity, left_weight, right_impurity, right_weight);
        
        if (gain > best_split.gain) {
          best_split.gain = gain;
          best_split.threshold = (var_type_ == VariableType::CATEGORICAL) ?
          values[i-1] + 0.5 : (i > 0 ? (values[i-1] + values[i]) / 2.0 : values[i]);
          best_split.impurity_left = left_impurity;
          best_split.impurity_right = right_impurity;
          best_split.n_left = i;
          best_split.n_right = n - i;
          best_split.weight_left = left_weight;
          best_split.weight_right = right_weight;
          best_split.valid = true;
        }
      }
    }
    
    g_logger.debug("Evaluated " + std::to_string(splits_evaluated) + 
      " split points, best gain: " + std::to_string(best_split.gain));
    
    return best_split;
  }
  
  void prune_tree() {
    if (!root_) return;
    
    int leaf_count = count_leaves(root_.get());
    g_logger.detail("Starting pruning with " + std::to_string(leaf_count) + " leaves");
    
    if (leaf_count <= max_bins_) return;
    
    while (leaf_count > max_bins_) {
      prune_weakest_split(root_.get());
      leaf_count = count_leaves(root_.get());
      g_logger.debug("Pruned to " + std::to_string(leaf_count) + " leaves");
    }
    
    g_logger.detail("Pruning complete with " + std::to_string(leaf_count) + " leaves");
  }
  
  int count_leaves(TreeNode* node) {
    if (!node) return 0;
    if (node->is_leaf) return 1;
    return count_leaves(node->left.get()) + count_leaves(node->right.get());
  }
  
  void prune_weakest_split(TreeNode* node) {
    if (!node || node->is_leaf) return;
    
    TreeNode* weakest_node = nullptr;
    double min_gain = INFINITY;
    
    std::function<void(TreeNode*)> find_weakest = [&](TreeNode* n) {
      if (!n || n->is_leaf) return;
      
      if (n->gain < min_gain) {
        min_gain = n->gain;
        weakest_node = n;
      }
      
      find_weakest(n->left.get());
      find_weakest(n->right.get());
    };
    
    find_weakest(node);
    
    if (weakest_node) {
      g_logger.debug("Pruning node " + std::to_string(weakest_node->id) + 
        " with gain " + std::to_string(min_gain));
      weakest_node->is_leaf = true;
      weakest_node->left.reset();
      weakest_node->right.reset();
    }
  }
  
  void extract_bins_from_tree() {
    bins_.clear();
    if (!root_) return;
    
    std::vector<std::pair<TreeNode*, std::pair<double, double>>> leaves;
    leaves.reserve(max_bins_);
    
    std::function<void(TreeNode*, double, double)> collect_leaves = 
      [&](TreeNode* node, double lower, double upper) {
        if (!node) return;
        
        if (node->is_leaf) {
          leaves.push_back({node, {lower, upper}});
        } else {
          collect_leaves(node->left.get(), lower, node->threshold);
          collect_leaves(node->right.get(), node->threshold, upper);
        }
      };
      
      collect_leaves(root_.get(), -INFINITY, INFINITY);
      
      std::sort(leaves.begin(), leaves.end(),
                [](const auto& a, const auto& b) {
                  return a.second.first < b.second.first;
                });
      
      int bin_id = 1;
      for (size_t i = 0; i < leaves.size(); ++i) {
        BinInfo bin;
        bin.id = bin_id++;
        bin.lower_bound = leaves[i].second.first;
        bin.upper_bound = leaves[i].second.second;
        
        if (var_type_ == VariableType::CATEGORICAL) {
          for (double ord = std::floor(bin.lower_bound < 0 ? 0 : bin.lower_bound); 
               ord < bin.upper_bound && ord < ordered_categories_.size(); ord++) {
            if (category_map_.count(ord)) {
              bin.categories.push_back(category_map_[ord]);
            }
          }
        }
        
        bin.total_count = leaves[i].first->n_samples;
        bin.pos_count = leaves[i].first->n_pos;
        bin.neg_count = leaves[i].first->n_neg;
        bin.weight_sum = leaves[i].first->weight_sum;
        bin.pos_weight = leaves[i].first->pos_weight;
        bin.neg_weight = leaves[i].first->neg_weight;
        
        bins_.push_back(std::move(bin));
      }
  }
  
  bool force_split_largest_bin() {
    int largest_idx = -1;
    int max_count = 2 * min_size_count_;
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      if (bins_[i].total_count >= max_count) {
        max_count = bins_[i].total_count;
        largest_idx = i;
      }
    }
    
    if (largest_idx < 0) {
      g_logger.debug("No bin large enough to split");
      return false;
    }
    
    g_logger.debug("Forcing split of bin " + std::to_string(largest_idx) + 
      " with " + std::to_string(max_count) + " samples");
    
    std::vector<DataPoint> bin_data;
    bin_data.reserve(max_count);
    
    for (const auto& dp : data_) {
      if (!dp.is_missing && !dp.is_special &&
          dp.value >= bins_[largest_idx].lower_bound &&
          dp.value < bins_[largest_idx].upper_bound) {
        bin_data.push_back(dp);
      }
    }
    
    if (bin_data.size() < 2 * min_size_count_) {
      g_logger.debug("Insufficient data in bin for splitting");
      return false;
    }
    
    std::set<double> unique_vals;
    for (const auto& dp : bin_data) {
      unique_vals.insert(dp.value);
    }
    
    if (unique_vals.size() < 2) {
      g_logger.debug("Not enough unique values in bin for splitting");
      return false;
    }
    
    std::sort(bin_data.begin(), bin_data.end(),
              [](const auto& a, const auto& b) { return a.value < b.value; });
    
    double split_value = -1;
    for (size_t i = min_size_count_; i <= bin_data.size() - min_size_count_; ++i) {
      if (i > 0 && bin_data[i].value != bin_data[i-1].value) {
        split_value = (bin_data[i-1].value + bin_data[i].value) / 2.0;
        break;
      }
    }
    
    if (split_value < 0) {
      split_value = bin_data[bin_data.size() / 2].value;
    }
    
    BinInfo left_bin = bins_[largest_idx];
    BinInfo right_bin = bins_[largest_idx];
    
    left_bin.upper_bound = split_value;
    right_bin.lower_bound = split_value;
    right_bin.id = bins_.size() + 1;
    
    left_bin.categories.clear();
    right_bin.categories.clear();
    
    left_bin.total_count = 0;
    left_bin.pos_count = 0;
    left_bin.neg_count = 0;
    left_bin.weight_sum = 0;
    left_bin.pos_weight = 0;
    left_bin.neg_weight = 0;
    
    right_bin.total_count = 0;
    right_bin.pos_count = 0;
    right_bin.neg_count = 0;
    right_bin.weight_sum = 0;
    right_bin.pos_weight = 0;
    right_bin.neg_weight = 0;
    
    for (const auto& dp : bin_data) {
      if (dp.value < split_value) {
        left_bin.total_count++;
        left_bin.weight_sum += dp.weight;
        if (dp.target > 0.5) {
          left_bin.pos_count++;
          left_bin.pos_weight += dp.weight;
        } else {
          left_bin.neg_count++;
          left_bin.neg_weight += dp.weight;
        }
        
        if (var_type_ == VariableType::CATEGORICAL && category_map_.count(dp.value)) {
          if (std::find(left_bin.categories.begin(), left_bin.categories.end(), 
                        category_map_[dp.value]) == left_bin.categories.end()) {
            left_bin.categories.push_back(category_map_[dp.value]);
          }
        }
      } else {
        right_bin.total_count++;
        right_bin.weight_sum += dp.weight;
        if (dp.target > 0.5) {
          right_bin.pos_count++;
          right_bin.pos_weight += dp.weight;
        } else {
          right_bin.neg_count++;
          right_bin.neg_weight += dp.weight;
        }
        
        if (var_type_ == VariableType::CATEGORICAL && category_map_.count(dp.value)) {
          if (std::find(right_bin.categories.begin(), right_bin.categories.end(), 
                        category_map_[dp.value]) == right_bin.categories.end()) {
            right_bin.categories.push_back(category_map_[dp.value]);
          }
        }
      }
    }
    
    bins_[largest_idx] = std::move(left_bin);
    bins_.insert(bins_.begin() + largest_idx + 1, std::move(right_bin));
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      bins_[i].id = i + 1;
    }
    
    g_logger.debug("Split successful, now have " + std::to_string(bins_.size()) + " bins");
    
    return true;
  }
  
  void enforce_monotonicity() {
    if (bins_.size() <= 1) return;
    
    MonotonicTrend trend = monotonic_trend_;
    if (trend == MonotonicTrend::AUTO) {
      trend = detect_monotonic_trend();
      g_logger.detail("Auto-detected trend: " + 
        (trend == MonotonicTrend::INCREASING ? std::string("increasing") :
           trend == MonotonicTrend::DECREASING ? std::string("decreasing") :
           std::string("none")));
    }
    
    if (trend == MonotonicTrend::NONE) return;
    
    std::vector<double> event_rates(bins_.size());
    std::vector<double> weights(bins_.size());
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      event_rates[i] = bins_[i].pos_weight / (bins_[i].weight_sum + EPSILON);
      weights[i] = bins_[i].weight_sum;
    }
    
    auto monotonic_rates = IsotonicRegression::fit(
      event_rates, weights, trend == MonotonicTrend::INCREASING);
    
    std::vector<BinInfo> new_bins;
    new_bins.reserve(bins_.size());
    
    BinInfo current_bin = bins_[0];
    
    for (size_t i = 1; i < bins_.size(); ++i) {
      if (std::abs(monotonic_rates[i] - monotonic_rates[i-1]) < EPSILON) {
        g_logger.debug("Merging bins " + std::to_string(current_bin.id) + 
          " and " + std::to_string(bins_[i].id) + " for monotonicity");
        
        current_bin.upper_bound = bins_[i].upper_bound;
        current_bin.total_count += bins_[i].total_count;
        current_bin.pos_count += bins_[i].pos_count;
        current_bin.neg_count += bins_[i].neg_count;
        current_bin.weight_sum += bins_[i].weight_sum;
        current_bin.pos_weight += bins_[i].pos_weight;
        current_bin.neg_weight += bins_[i].neg_weight;
        
        for (const auto& cat : bins_[i].categories) {
          if (std::find(current_bin.categories.begin(), current_bin.categories.end(), cat) 
                == current_bin.categories.end()) {
            current_bin.categories.push_back(cat);
          }
        }
      } else {
        new_bins.push_back(std::move(current_bin));
        current_bin = bins_[i];
      }
    }
    
    new_bins.push_back(std::move(current_bin));
    
    for (size_t i = 0; i < new_bins.size(); ++i) {
      new_bins[i].id = i + 1;
    }
    
    bins_ = std::move(new_bins);
  }
  
  MonotonicTrend detect_monotonic_trend() {
    if (bins_.size() <= 1) return MonotonicTrend::NONE;
    
    std::vector<double> centroids(bins_.size());
    std::vector<double> event_rates(bins_.size());
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      double centroid = (bins_[i].lower_bound + bins_[i].upper_bound) / 2.0;
      if (!std::isfinite(centroid)) {
        centroid = std::isfinite(bins_[i].lower_bound) ? 
        bins_[i].lower_bound + 1.0 : bins_[i].upper_bound - 1.0;
      }
      
      centroids[i] = centroid;
      event_rates[i] = bins_[i].pos_weight / (bins_[i].weight_sum + EPSILON);
    }
    
    auto rank = [](std::vector<double>& v) {
      std::vector<std::pair<double, size_t>> indexed;
      indexed.reserve(v.size());
      
      for (size_t i = 0; i < v.size(); ++i) {
        indexed.push_back({v[i], i});
      }
      std::sort(indexed.begin(), indexed.end());
      
      std::vector<double> ranks(v.size());
      for (size_t i = 0; i < indexed.size(); ++i) {
        ranks[indexed[i].second] = i + 1;
      }
      return ranks;
    };
    
    auto rank_centroids = rank(centroids);
    auto rank_rates = rank(event_rates);
    
    double sum_sq_diff = 0;
    for (size_t i = 0; i < rank_centroids.size(); ++i) {
      double diff = rank_centroids[i] - rank_rates[i];
      sum_sq_diff += diff * diff;
    }
    
    int n = rank_centroids.size();
    double spearman = 1.0 - (6.0 * sum_sq_diff) / (n * (n * n - 1));
    
    g_logger.debug("Spearman correlation: " + std::to_string(spearman));
    
    if (spearman > 0.3) return MonotonicTrend::INCREASING;
    if (spearman < -0.3) return MonotonicTrend::DECREASING;
    
    return MonotonicTrend::NONE;
  }
  
  void merge_adjacent_bins_with_min_iv_loss() {
    if (bins_.size() <= 1) return;
    
    double min_loss = INFINITY;
    size_t merge_idx = 0;
    
    for (size_t i = 0; i < bins_.size() - 1; ++i) {
      double current_iv = bins_[i].iv + bins_[i+1].iv;
      double loss = std::abs(current_iv);
      
      if (loss < min_loss) {
        min_loss = loss;
        merge_idx = i;
      }
    }
    
    g_logger.debug("Merging bins " + std::to_string(merge_idx) + 
      " and " + std::to_string(merge_idx + 1) + 
      " with IV loss: " + std::to_string(min_loss));
    
    bins_[merge_idx].upper_bound = bins_[merge_idx + 1].upper_bound;
    bins_[merge_idx].total_count += bins_[merge_idx + 1].total_count;
    bins_[merge_idx].pos_count += bins_[merge_idx + 1].pos_count;
    bins_[merge_idx].neg_count += bins_[merge_idx + 1].neg_count;
    bins_[merge_idx].weight_sum += bins_[merge_idx + 1].weight_sum;
    bins_[merge_idx].pos_weight += bins_[merge_idx + 1].pos_weight;
    bins_[merge_idx].neg_weight += bins_[merge_idx + 1].neg_weight;
    
    for (const auto& cat : bins_[merge_idx + 1].categories) {
      if (std::find(bins_[merge_idx].categories.begin(), bins_[merge_idx].categories.end(), cat) 
            == bins_[merge_idx].categories.end()) {
        bins_[merge_idx].categories.push_back(cat);
      }
    }
    
    bins_.erase(bins_.begin() + merge_idx + 1);
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      bins_[i].id = i + 1;
    }
  }
};

// =============================================================================
// WOE/IV/KS/GINI CALCULATIONS
// =============================================================================

class MetricsCalculator {
public:
  static void calculate_metrics(std::vector<BinInfo>& bins, 
                                double smooth, 
                                int total_pos, 
                                int total_neg) {
    if (bins.empty() || total_pos == 0 || total_neg == 0) return;
    
    g_logger.detail("Calculating metrics for " + std::to_string(bins.size()) + " bins");
    g_logger.debug("Total positive: " + std::to_string(total_pos) + 
      ", Total negative: " + std::to_string(total_neg));
    
    int G = bins.size();
    double N0 = total_neg + G * smooth;
    double N1 = total_pos + G * smooth;
    
    double cum_pos = 0, cum_neg = 0;
    
    for (auto& bin : bins) {
      double p_pos = (bin.pos_count + smooth) / N1;
      double p_neg = (bin.neg_count + smooth) / N0;
      
      bin.woe = safe_log(p_pos / p_neg);
      bin.woe = clip_value(bin.woe, WOE_MIN, WOE_MAX);
      
      double actual_p_pos = static_cast<double>(bin.pos_count) / (total_pos + EPSILON);
      double actual_p_neg = static_cast<double>(bin.neg_count) / (total_neg + EPSILON);
      bin.iv = (actual_p_pos - actual_p_neg) * bin.woe;
      
      cum_pos += bin.pos_count;
      cum_neg += bin.neg_count;
      
      double cdf_pos = cum_pos / (total_pos + EPSILON);
      double cdf_neg = cum_neg / (total_neg + EPSILON);
      bin.ks = std::abs(cdf_pos - cdf_neg);
      
      double p0 = static_cast<double>(bin.neg_count) / (bin.total_count + EPSILON);
      double p1 = static_cast<double>(bin.pos_count) / (bin.total_count + EPSILON);
      bin.gini = std::abs(p1 - p0);
      
      g_logger.bin_info(bin.id, bin.label, bin.total_count, bin.woe, bin.iv);
    }
  }
  
  static double calculate_total_iv(const std::vector<BinInfo>& bins) {
    double total = 0;
    for (const auto& bin : bins) {
      total += bin.iv;
    }
    return total;
  }
};

// =============================================================================
// MAIN CART_WOE IMPLEMENTATION
// =============================================================================

List process_feature(
    const DataFrame& data,
    const std::string& target_col,
    const std::string& feature_col,
    int min_bins,
    int max_bins,
    const std::string& method,
    const std::string& miss_policy,
    const std::string& cat_sep,
    int digits,
    double smooth,
    Criterion criterion,
    double min_size,
    bool use_pruning,
    int cv_folds,
    MonotonicTrend monotonic_trend,
    const std::string& monotonic_mode,
    const NumericVector& weights,
    const std::set<double>& special_vals,
    int max_cat,
    double rare_pct,
    bool use_parallel) {
  
  g_logger.set_feature(feature_col);
  g_logger.info("Processing feature: " + feature_col);
  
  NumericVector target = as<NumericVector>(data[target_col]);
  SEXP feature_sexp = data[feature_col];
  
  VariableType var_type = detect_variable_type(feature_sexp);
  g_logger.detail("Variable type: " + variable_type_to_string(var_type));
  
  if (var_type == VariableType::DATE_TIME || var_type == VariableType::UNSUPPORTED) {
    g_logger.warning("Variable skipped: " + variable_type_to_string(var_type));
    
    return List::create(
      Named("bins") = DataFrame::create(),
      Named("total_iv") = 0.0,
      Named("variable_type") = variable_type_to_string(var_type),
      Named("method_used") = method,
      Named("n_bins") = 0,
      Named("total_samples") = 0,
      Named("event_rate") = 0.0,
      Named("messages") = "Variable skipped: " + variable_type_to_string(var_type),
      Named("transform") = DataFrame::create()
    );
  }
  
  // Count unique classes/values
  int unique_classes_count = 0;
  if (var_type == VariableType::BOOLEAN) {
    unique_classes_count = 2;
  } else if (var_type == VariableType::CATEGORICAL) {
    if (Rf_isFactor(feature_sexp)) {
      IntegerVector iv(feature_sexp);
      std::set<int> unique_vals;
      for (int i = 0; i < iv.size(); ++i) {
        if (!IntegerVector::is_na(iv[i])) {
          unique_vals.insert(iv[i]);
        }
      }
      unique_classes_count = unique_vals.size();
    } else {
      CharacterVector cv(feature_sexp);
      std::set<std::string> unique_vals;
      for (int i = 0; i < cv.size(); ++i) {
        if (!CharacterVector::is_na(cv[i])) {
          unique_vals.insert(as<std::string>(cv[i]));
        }
      }
      unique_classes_count = unique_vals.size();
    }
  } else {
    NumericVector nv(feature_sexp);
    std::set<double> unique_vals;
    int check_limit = std::min(1000, static_cast<int>(nv.size()));
    for (int i = 0; i < check_limit; ++i) {
      if (!NumericVector::is_na(nv[i]) && std::isfinite(nv[i])) {
        unique_vals.insert(nv[i]);
        if (unique_vals.size() > 2) break;
      }
    }
    unique_classes_count = unique_vals.size();
  }
  
  g_logger.detail("Unique classes/values: " + std::to_string(unique_classes_count));
  
  bool is_simple_variable = (unique_classes_count <= 2);
  
  // Prepare data points
  g_logger.subheader("Preparing Data Points");
  
  std::vector<DataPoint> data_points;
  data_points.reserve(target.size());
  
  int total_valid_pos = 0, total_valid_neg = 0;
  
  std::vector<std::string> original_categorical_values;
  std::vector<double> original_numeric_values;
  
  if (var_type == VariableType::CATEGORICAL) {
    original_categorical_values.resize(target.size());
  } else {
    original_numeric_values.resize(target.size());
  }
  
  int missing_count = 0;
  int special_count = 0;
  
  for (int i = 0; i < target.size(); ++i) {
    DataPoint dp;
    dp.target = target[i];
    dp.weight = weights[i];
    dp.original_index = i;
    
    bool is_missing = false;
    double value = 0;
    
    switch (var_type) {
    case VariableType::NUMERIC_CONTINUOUS:
    case VariableType::NUMERIC_DISCRETE: {
      NumericVector nv(feature_sexp);
      if (NumericVector::is_na(nv[i]) || !std::isfinite(nv[i])) {
        is_missing = true;
        missing_count++;
        original_numeric_values[i] = NA_REAL;
      } else {
        value = nv[i];
        original_numeric_values[i] = value;
        if (special_vals.count(value)) {
          dp.is_special = true;
          dp.special_value = value;
          special_count++;
        }
      }
      break;
    }
    case VariableType::CATEGORICAL: {
      if (Rf_isFactor(feature_sexp)) {
      IntegerVector iv(feature_sexp);
      if (IntegerVector::is_na(iv[i])) {
        is_missing = true;
        missing_count++;
        original_categorical_values[i] = "NA";
      } else {
        CharacterVector levels = iv.attr("levels");
        dp.category_name = as<std::string>(levels[iv[i] - 1]);
        original_categorical_values[i] = dp.category_name;
        value = iv[i];
      }
    } else {
      CharacterVector cv(feature_sexp);
      if (CharacterVector::is_na(cv[i])) {
        is_missing = true;
        missing_count++;
        original_categorical_values[i] = "NA";
      } else {
        dp.category_name = as<std::string>(cv[i]);
        original_categorical_values[i] = dp.category_name;
        std::hash<std::string> hasher;
        value = static_cast<double>(hasher(dp.category_name) % 1000000);
      }
    }
    break;
    }
    case VariableType::BOOLEAN: {
      LogicalVector lv(feature_sexp);
      if (LogicalVector::is_na(lv[i])) {
        is_missing = true;
        missing_count++;
        original_numeric_values[i] = NA_REAL;
      } else {
        value = lv[i] ? 1.0 : 0.0;
        original_numeric_values[i] = value;
        dp.category_name = lv[i] ? "TRUE" : "FALSE";
      }
      break;
    }
    default:
      break;
    }
    
    dp.is_missing = is_missing;
    dp.value = value;
    
    if (!is_missing && !dp.is_special) {
      if (dp.target > 0.5) {
        total_valid_pos++;
      } else {
        total_valid_neg++;
      }
    }
    
    data_points.push_back(std::move(dp));
  }
  
  g_logger.detail("Total samples: " + std::to_string(target.size()));
  g_logger.detail("Missing values: " + std::to_string(missing_count));
  g_logger.detail("Special values: " + std::to_string(special_count));
  g_logger.detail("Valid positive: " + std::to_string(total_valid_pos));
  g_logger.detail("Valid negative: " + std::to_string(total_valid_neg));
  
  // Handle missing values
  g_logger.subheader("Handling Missing Values");
  g_logger.detail("Missing policy: " + miss_policy);
  
  std::vector<DataPoint> processed_data;
  processed_data.reserve(data_points.size());
  
  std::vector<BinInfo> special_bins;
  special_bins.reserve(10);
  
  if (miss_policy == "separate") {
    BinInfo missing_bin;
    missing_bin.id = 0;
    missing_bin.label = "[Missing]";
    
    std::unordered_map<double, BinInfo> special_bin_map;
    
    for (const auto& dp : data_points) {
      if (dp.is_missing) {
        missing_bin.total_count++;
        if (dp.target > 0.5) {
          missing_bin.pos_count++;
          missing_bin.pos_weight += dp.weight;
        } else {
          missing_bin.neg_count++;
          missing_bin.neg_weight += dp.weight;
        }
        missing_bin.weight_sum += dp.weight;
      } else if (dp.is_special) {
        auto& sbin = special_bin_map[dp.special_value];
        sbin.total_count++;
        if (dp.target > 0.5) {
          sbin.pos_count++;
          sbin.pos_weight += dp.weight;
        } else {
          sbin.neg_count++;
          sbin.neg_weight += dp.weight;
        }
        sbin.weight_sum += dp.weight;
      } else {
        processed_data.push_back(dp);
      }
    }
    
    if (missing_bin.total_count > 0) {
      special_bins.push_back(std::move(missing_bin));
      g_logger.detail("Created missing bin with " + 
        std::to_string(missing_bin.total_count) + " samples");
    }
    
    for (auto& [val, sbin] : special_bin_map) {
      sbin.label = "[Special:" + std::to_string(val) + "]";
      special_bins.push_back(std::move(sbin));
      g_logger.detail("Created special bin for value " + std::to_string(val) + 
        " with " + std::to_string(sbin.total_count) + " samples");
    }
  } else {
    for (const auto& dp : data_points) {
      if (!dp.is_missing && !dp.is_special) {
        processed_data.push_back(dp);
      }
    }
  }
  
  g_logger.detail("Processed data size: " + std::to_string(processed_data.size()));
  
  // Build and fit the binning model
  std::vector<BinInfo> regular_bins;
  std::string message = "Success";
  
  if (!processed_data.empty()) {
    g_logger.subheader("Starting Binning Algorithm");
    
    if (is_simple_variable) {
      g_logger.info("Using direct binning for simple variable");
      
      OptimalBinningCART binner(
          std::move(processed_data),
          criterion, min_size, min_bins, max_bins,
          use_pruning, cv_folds, monotonic_trend, var_type, cat_sep,
          unique_classes_count
      );
      
      binner.fit();
      regular_bins = binner.get_bins();
      
      // Create appropriate labels
      for (auto& bin : regular_bins) {
        if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN) {
          if (!bin.categories.empty()) {
            bin.label = bin.categories[0];
          }
        } else {
          std::stringstream ss;
          ss.precision(digits);
          ss << std::fixed << bin.lower_bound;
          bin.label = ss.str();
        }
      }
      
      message = "Variable has " + std::to_string(unique_classes_count) + 
        " unique classes. Direct binning applied without optimization.";
    } else {
      g_logger.info("Using CART optimization for complex variable");
      
      OptimalBinningCART binner(
          std::move(processed_data),
          criterion, min_size, min_bins, max_bins,
          use_pruning, cv_folds, monotonic_trend, var_type, cat_sep,
          unique_classes_count
      );
      
      binner.fit();
      regular_bins = binner.get_bins();
      
      // Create labels for bins
      for (auto& bin : regular_bins) {
        std::stringstream ss;
        ss.precision(digits);
        
        if (var_type == VariableType::CATEGORICAL && !bin.categories.empty()) {
          for (size_t i = 0; i < bin.categories.size(); ++i) {
            if (i > 0) ss << cat_sep;
            ss << bin.categories[i];
          }
          bin.label = ss.str();
        } else {
          if (std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
            ss << "All";
          } else if (std::isinf(bin.lower_bound)) {
            ss << "(-inf, " << std::fixed << bin.upper_bound << ")";
          } else if (std::isinf(bin.upper_bound)) {
            ss << "[" << std::fixed << bin.lower_bound << ", +inf)";
          } else {
            ss << "[" << std::fixed << bin.lower_bound << ", " 
               << std::fixed << bin.upper_bound << ")";
          }
          bin.label = ss.str();
        }
      }
    }
    
    g_logger.detail("Created " + std::to_string(regular_bins.size()) + " regular bins");
  }
  
  // Combine special and regular bins
  g_logger.subheader("Combining Bins");
  
  std::vector<BinInfo> all_bins;
  all_bins.reserve(special_bins.size() + regular_bins.size());
  
  for (auto& bin : special_bins) {
    all_bins.push_back(std::move(bin));
  }
  for (auto& bin : regular_bins) {
    all_bins.push_back(std::move(bin));
  }
  
  g_logger.detail("Total bins: " + std::to_string(all_bins.size()));
  
  // Calculate metrics for all bins
  g_logger.subheader("Calculating Metrics");
  MetricsCalculator::calculate_metrics(all_bins, smooth, total_valid_pos, total_valid_neg);
  
  // Create output DataFrames
  IntegerVector bin_ids;
  CharacterVector bin_labels;
  IntegerVector total_counts;
  IntegerVector neg_counts;
  IntegerVector pos_counts;
  NumericVector woes;
  NumericVector ivs;
  NumericVector kss;
  NumericVector ginis;
  
  for (const auto& bin : all_bins) {
    bin_ids.push_back(bin.id);
    bin_labels.push_back(bin.label);
    total_counts.push_back(bin.total_count);
    neg_counts.push_back(bin.neg_count);
    pos_counts.push_back(bin.pos_count);
    woes.push_back(bin.woe);
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
    Named("iv") = ivs,
    Named("ks") = kss,
    Named("gini") = ginis
  );
  
  // FAST TRANSFORM using FastBinMapper
  g_logger.subheader("Creating Transformations");
  
  FastBinMapper mapper(var_type);
  mapper.build_from_bins(all_bins);
  
  CharacterVector transform_feature_values(target.size());
  CharacterVector feature_bins(target.size());
  NumericVector feature_woes(target.size());
  
  for (int i = 0; i < target.size(); ++i) {
    std::string value_str = "";
    std::string bin_label = "Unknown";
    double woe_value = 0;
    
    bool is_missing = false;
    
    if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN) {
      value_str = original_categorical_values[i];
      is_missing = (value_str == "NA");
      
      mapper.map_categorical(value_str, is_missing, bin_label, woe_value);
    } else {
      double numeric_value = original_numeric_values[i];
      is_missing = std::isnan(numeric_value) || numeric_value == NA_REAL;
      
      if (!is_missing) {
        value_str = std::to_string(numeric_value);
        value_str.erase(value_str.find_last_not_of('0') + 1, std::string::npos);
        value_str.erase(value_str.find_last_not_of('.') + 1, std::string::npos);
      } else {
        value_str = "NA";
      }
      
      bool is_special = special_vals.count(numeric_value) > 0;
      mapper.map_numeric(numeric_value, is_missing, is_special, bin_label, woe_value);
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
  
  double total_iv = MetricsCalculator::calculate_total_iv(all_bins);
  double event_rate = static_cast<double>(total_valid_pos) / 
    (total_valid_pos + total_valid_neg + EPSILON);
  
  g_logger.metrics("Total IV", total_iv);
  g_logger.metrics("Event Rate", event_rate);
  
  if (!is_simple_variable && static_cast<int>(regular_bins.size()) < min_bins && !processed_data.empty()) {
    message = "Warning: Could not achieve minimum " + std::to_string(min_bins) + 
      " bins. Created " + std::to_string(regular_bins.size()) + " bins.";
    g_logger.warning(message);
  }
  
  g_logger.success("Feature processing complete in " + g_logger.get_feature_time());
  
  return List::create(
    Named("bins") = bins_df,
    Named("total_iv") = total_iv,
    Named("variable_type") = variable_type_to_string(var_type),
    Named("method_used") = method,
    Named("n_bins") = all_bins.size(),
    Named("total_samples") = processed_data.size(),
    Named("event_rate") = event_rate,
    Named("messages") = message,
    Named("transform") = transform_df
  );
}

} // namespace cartwoe

// =============================================================================
// EXPORTED FUNCTION (WITH VERBOSE PARAMETER ADDED)
// =============================================================================

//' CART Optimal Binning with Weight of Evidence (WoE)
//'
//' @description
//' Performs optimal binning on numeric and categorical variables using Classification and Regression Trees (CART),
//' computing Weight of Evidence (WoE) and Information Value (IV) for each bin. Implements comprehensive binning
//' strategies including:
//' \itemize{
//'   \item \strong{CART Pure}: Standard Classification and Regression Trees
//'   \item \strong{CART+PAVA}: CART with isotonic regression for monotonicity enforcement
//'   \item \strong{CART+Merge}: CART with heuristic adjacent bin merging
//' }
//'
//' @details
//' \strong{1) CART Algorithm for Optimal Binning}
//'
//' The algorithm builds a binary tree by recursively partitioning the feature space to maximize impurity reduction:
//' \deqn{\Delta(t) = I(S) - \frac{W_L}{W}I(S_L) - \frac{W_R}{W}I(S_R)}
//' where:
//' \itemize{
//'   \item \eqn{I(S)} = impurity of parent node \eqn{S}
//'   \item \eqn{I(S_L), I(S_R)} = impurities of left and right child nodes
//'   \item \eqn{W, W_L, W_R} = total weights of parent, left child, and right child
//' }
//'
//' \strong{Impurity Measures:}
//' \itemize{
//'   \item \strong{Gini Impurity}: \eqn{I(S) = 1 - \sum_{i=1}^{C} p_i^2}
//'   \item \strong{Entropy}: \eqn{I(S) = -\sum_{i=1}^{C} p_i \log(p_i)}
//' }
//' where \eqn{p_i} is the probability of class \eqn{i} in the node.
//'
//' \strong{2) Weight of Evidence (WoE)}
//'
//' For bin \eqn{g}, WoE measures the strength of relationship with the target:
//' \deqn{\text{WoE}_g = \ln\left(\frac{P(Y=1|X \in \text{bin}_g)}{P(Y=0|X \in \text{bin}_g)}\right) = \ln\left(\frac{n_{1g}/N_1}{n_{0g}/N_0}\right)}
//' where:
//' \itemize{
//'   \item \eqn{n_{1g}} = number of positive events in bin \eqn{g}
//'   \item \eqn{n_{0g}} = number of negative events in bin \eqn{g}
//'   \item \eqn{N_1} = total positive events
//'   \item \eqn{N_0} = total negative events
//' }
//'
//' \strong{Laplace Smoothing}: To handle zero frequencies, applies smoothing factor \eqn{\lambda}:
//' \deqn{\text{WoE}_g = \ln\left(\frac{(n_{1g} + \lambda)/(N_1 + G\lambda)}{(n_{0g} + \lambda)/(N_0 + G\lambda)}\right)}
//' where \eqn{G} is the total number of bins.
//'
//' \strong{3) Information Value (IV)}
//'
//' Total IV quantifies the predictive power of the variable:
//' \deqn{\text{IV} = \sum_{g=1}^{G} \left(\frac{n_{1g}}{N_1} - \frac{n_{0g}}{N_0}\right) \times \text{WoE}_g}
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
//' When monotonic constraints are applied:
//' \itemize{
//'   \item \code{"cart+pava"}: Uses Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
//'   \item \code{"cart+merge"}: Heuristic merging of adjacent bins violating monotonicity
//' }
//'
//' \strong{PAVA Algorithm:}
//' Solves the isotonic regression problem:
//' \deqn{\min_{\theta_1 \leq \theta_2 \leq \cdots \leq \theta_G} \sum_{g=1}^{G} w_g(\theta_g - y_g)^2}
//' where \eqn{y_g} are target values (event rates or WoE) and \eqn{w_g} are weights.
//'
//' \strong{5) Cost-Complexity Pruning}
//'
//' To prevent overfitting, applies cost-complexity pruning:
//' \deqn{R_\alpha(T) = R(T) + \alpha|T|}
//' where:
//' \itemize{
//'   \item \eqn{R(T)} = total impurity of tree \eqn{T}
//'   \item \eqn{|T|} = number of terminal nodes (leaves)
//'   \item \eqn{\alpha} = complexity parameter
//' }
//'
//' \strong{6) Special Value Handling}
//'
//' Missing values and special codes are handled according to \code{miss_policy}:
//' \itemize{
//'   \item \code{"separate"}: Creates separate bin(s) for missing/special values
//'   \item \code{"remove"}: Excludes missing values from binning
//'   \item \code{"impute"}: Imputes with mode (categorical) or median (numeric)
//'   \item \code{"merge"}: Merges with nearest bin based on event rate
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
//'   Algorithm will force creation of at least this many bins when possible.
//'   Default: 2
//'
//' @param max_bins Integer specifying maximum number of bins allowed. Must be greater than or equal to
//'   min_bins. Controls model complexity and prevents overfitting.
//'   Default: 6
//'
//' @param method String specifying the binning algorithm to use. Options:
//'   \itemize{
//'     \item \code{"cart"}: Pure CART algorithm
//'     \item \code{"cart+pava"}: CART with PAVA monotonicity enforcement (default)
//'     \item \code{"cart+merge"}: CART with heuristic merging for monotonicity
//'   }
//'   Default: "cart"
//'
//' @param miss_policy String specifying how to handle missing values. Options:
//'   \itemize{
//'     \item \code{"separate"}: Create separate bin(s) for missing values (default)
//'     \item \code{"remove"}: Exclude missing values from analysis
//'     \item \code{"impute"}: Impute with mode (categorical) or median (numeric)
//'     \item \code{"merge"}: Merge with nearest bin based on event rate
//'   }
//'   Default: "separate"
//'
//' @param cat_sep String separator used when combining multiple categories into a single bin label.
//'   Should be a pattern unlikely to appear in actual category names.
//'   Default: "\%;\%"
//'
//' @param digits Integer number of decimal places for numeric bin boundaries in labels.
//'   Must be between 0 and 10. Affects display only, not calculation precision.
//'   Default: 4
//'
//' @param smooth Numeric Laplace smoothing factor for WoE calculation. Must be non-negative.
//'   Prevents undefined WoE values when bins have zero counts. Higher values provide more smoothing.
//'   Default: 0.5
//'
//' @param criterion String specifying the impurity criterion for CART splits. Options:
//'   \itemize{
//'     \item \code{"gini"}: Gini impurity (default)
//'     \item \code{"entropy"}: Information gain (entropy-based)
//'   }
//'   Default: "gini"
//'
//' @param min_size Numeric minimum bin size specification. Can be:
//'   \itemize{
//'     \item Proportion (0,1]: Minimum proportion of total observations per bin
//'     \item Count (>1): Minimum absolute count per bin
//'   }
//'   Prevents creation of very small bins that may be unstable.
//'   Default: 0.05
//'
//' @param use_pruning Logical indicating whether to apply cost-complexity pruning to the CART tree.
//'   Helps prevent overfitting by removing branches that provide little predictive power.
//'   Default: TRUE
//'
//' @param cv_folds Integer number of cross-validation folds for pruning parameter selection.
//'   Must be at least 2 when pruning is enabled. Higher values provide more robust pruning.
//'   Default: 5
//'
//' @param monotonic_trend String specifying the desired monotonic relationship. Options:
//'   \itemize{
//'     \item \code{"auto"}: Automatically detect trend based on Spearman correlation (default)
//'     \item \code{"increasing"}: Enforce increasing WoE trend
//'     \item \code{"decreasing"}: Enforce decreasing WoE trend
//'     \item \code{"none"}: No monotonic constraint
//'   }
//'   Default: "auto"
//'
//' @param monotonic_mode String specifying the method for enforcing monotonicity. Options:
//'   \itemize{
//'     \item \code{"pava"}: Isotonic regression using PAVA algorithm (default)
//'     \item \code{"merge"}: Heuristic adjacent bin merging
//'   }
//'   Default: "pava"
//'
//' @param parallel Logical indicating whether to use parallel processing via OpenMP for multiple features.
//'   Requires OpenMP support in compilation. Speeds up processing of many features.
//'   Default: FALSE
//'
//' @param weights Optional NumericVector of observation weights for weighted binning. Length must equal
//'   number of rows in data. All weights must be non-negative. NULL indicates equal weights.
//'   Default: NULL
//'
//' @param special_vals NumericVector of special values to handle separately (e.g., -999, -888 for
//'   special codes). These values are isolated in separate bins when encountered.
//'   Default: empty vector (no special values)
//'
//' @param max_cat Integer maximum number of categories before automatic grouping for categorical variables.
//'   Must be at least 2. Categories beyond this limit are grouped based on event rate similarity.
//'   Default: 50
//'
//' @param rare_pct Numeric threshold for rare category grouping as proportion of total.
//'   Categories with frequency below this threshold are combined. Must be between 0 and 0.5.
//'   Default: 0.01 (1\%)
//'
//' @param verbose Integer controlling log verbosity level (0-3):
//'   \itemize{
//'     \item 0: Silent mode - no logs
//'     \item 1: Basic logs - key steps and results  
//'     \item 2: Detailed logs - includes algorithm progress
//'     \item 3: Debug logs - full algorithm details
//'   }
//'   Default: 0
//'
//' @return
//' Named list with one element per processed feature, each containing:
//' \itemize{
//'   \item \code{bins}: data.frame with columns:
//'     \itemize{
//'       \item \code{bin_id}: Sequential bin identifier (1, 2, 3, ...)
//'       \item \code{bin_label}: Formatted bin range/categories (e.g., "(-inf, 25.5]", "[25.5, 40.0]", "(40.0, +inf)")
//'       \item \code{total_count}: Total observations in bin
//'       \item \code{neg_count}: Count of negative class (Y=0)
//'       \item \code{pos_count}: Count of positive class (Y=1)
//'       \item \code{woe}: Weight of Evidence value for the bin
//'       \item \code{iv}: Information Value contribution of the bin
//'       \item \code{ks}: Kolmogorov-Smirnov statistic up to this bin
//'       \item \code{gini}: Gini coefficient for the bin
//'     }
//'   \item \code{total_iv}: Total Information Value for the feature
//'   \item \code{variable_type}: Detected type ("numeric_continuous", "numeric_discrete", "categorical", "boolean")
//'   \item \code{method_used}: Algorithm actually used for binning
//'   \item \code{n_bins}: Final number of bins created (including special bins if separate)
//'   \item \code{total_samples}: Total valid observations processed
//'   \item \code{event_rate}: Overall positive class rate in processed data
//'   \item \code{messages}: Processing messages or warnings
//'   \item \code{transform}: data.frame with original values and transformations:
//'     \itemize{
//'       \item \code{[feature]}: Original feature values
//'       \item \code{[feature]_bin}: Assigned bin labels
//'       \item \code{[feature]_woe}: Assigned WoE values
//'     }
//' }
//'
//' @section Algorithm Pipeline:
//' The complete binning process follows these steps:
//' \enumerate{
//'   \item \strong{Data Preparation}:
//'     \itemize{
//'       \item Detect variable type (numeric, categorical, boolean, date)
//'       \item Handle missing values according to \code{miss_policy}
//'       \item Process special values into separate bins
//'       \item For categorical variables: order by event rate and map to ordinal scale
//'     }
//'   \item \strong{CART Tree Construction}:
//'     \itemize{
//'       \item Sort data by feature value (numeric) or ordinal mapping (categorical)
//'       \item Build binary tree using best-first search with impurity gain criterion
//'       \item Respect \code{min_size} constraints during splitting
//'       \item Stop when reaching \code{max_bins} leaves or no beneficial splits
//'     }
//'   \item \strong{Tree Pruning} (if enabled):
//'     \itemize{
//'       \item Generate sequence of subtrees via cost-complexity pruning
//'       \item Select optimal subtree using cross-validation
//'       \item Ensure result respects \code{min_bins} and \code{max_bins} constraints
//'     }
//'   \item \strong{Bin Adjustment}:
//'     \itemize{
//'       \item Force minimum bin count if below \code{min_bins}
//'       \item Merge bins if above \code{max_bins}
//'       \item Apply monotonicity constraints if specified
//'     }
//'   \item \strong{Metric Calculation}:
//'     \itemize{
//'       \item Compute WoE and IV with Laplace smoothing
//'       \item Calculate KS statistics and Gini coefficients
//'       \item Generate transformation mappings
//'     }
//' }
//'
//' @section Special Cases Handling:
//' \itemize{
//'   \item \strong{Variables with \eqn{\le 2} unique classes}: Direct binning without optimization
//'   \item \strong{Date/Time variables}: Automatically skipped with warning message
//'   \item \strong{Unsupported types}: Skipped with appropriate message
//'   \item \strong{Constant variables}: Single bin created with warning
//'   \item \strong{Insufficient data}: Processing skipped with informative message
//' }
//'
//' @note
//' \itemize{
//'   \item WoE values are clipped to [-20, 20] for numerical stability
//'   \item Variables with 2 or fewer unique classes are processed directly without CART optimization
//'   \item Factor variables are processed as categorical with proper level handling
//'   \item Integer variables with \eqn{\leq 20} unique values are treated as discrete numeric
//'   \item Special values (-999, -888, etc.) can be isolated in separate bins
//'   \item Uses "(-inf,x)" and "[x,+inf)" notation for unbounded intervals
//'   \item Categorical bins combine multiple categories using \code{cat_sep} separator
//' }
//'
//' @references
//' \itemize{
//'   \item Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
//'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//'   \item Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. Journal of the Royal Statistical Society: Series A (Statistics in Society), 160(3), 523-541.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//'   \item Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). Order restricted statistical inference. Wiley.
//' }
//'
//' @examples
//' \dontrun{
//' # Load credit data
//' data <- scorecard::germancredit
//' data$target <- ifelse(data$creditability == "good", 1, 0)
//'
//' # Basic binning with all defaults
//' result <- cart_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
//' )
//'
//' # Advanced binning with custom parameters
//' result_custom <- cart_woe(
//'   data = data,
//'   target_col = "target",
//'   feature_cols = names(data)[1:10],
//'   min_bins = 3,              # Minimum 3 bins
//'   max_bins = 8,              # Maximum 8 bins
//'   method = "cart+pava",      # CART with PAVA monotonicity
//'   miss_policy = "separate",  # Missing values in separate bin
//'   cat_sep = " | ",           # Use pipe separator for categories
//'   digits = 2,                # 2 decimal places in labels
//'   smooth = 1.0,              # Higher smoothing for WoE
//'   criterion = "entropy",     # Use entropy instead of Gini
//'   min_size = 0.03,           # Each bin at least 3% of data
//'   use_pruning = TRUE,        # Enable cost-complexity pruning
//'   cv_folds = 10,             # 10-fold CV for pruning
//'   monotonic_trend = "auto",  # Auto-detect monotonic trend
//'   monotonic_mode = "pava",   # Use PAVA for monotonicity
//'   parallel = TRUE,           # Parallel processing for speed
//'   special_vals = c(-999, -888), # Special codes to isolate
//'   max_cat = 25,              # Group if >25 categories
//'   rare_pct = 0.02            # Group categories <2% frequency
//' )
//'
//' # Extract results for a specific feature
//' duration_bins <- result$duration.in.month$bins
//' duration_iv <- result$duration.in.month$total_iv
//' duration_transform <- result$duration.in.month$transform
//'
//' # View binning results
//' print(duration_bins)
//' print(paste("Total IV:", duration_iv))
//'
//' # Check transformation
//' head(duration_transform)
//' }
//'
//' @export
// [[Rcpp::export]]
List cart_woe(
   DataFrame data,
   std::string target_col,
   CharacterVector feature_cols,
   int min_bins = 2,
   int max_bins = 5,
   std::string method = "cart",
   std::string miss_policy = "separate",
   std::string cat_sep = "%;%",
   int digits = 4,
   double smooth = 0.0,
   std::string criterion = "gini",
   double min_size = 0.05,
   bool use_pruning = true,
   int cv_folds = 5,
   std::string monotonic_trend = "auto",
   std::string monotonic_mode = "pava",
   bool parallel = false,
   Nullable<NumericVector> weights = R_NilValue,
   NumericVector special_vals = NumericVector::create(),
   int max_cat = 50,
   double rare_pct = 0.01,
   int verbose = 0) {
 
 // Initialize logger with verbose level
 cartwoe::g_logger.set_verbose(verbose);
 
 if (verbose > 0) {
   cartwoe::g_logger.header("CART-WOE Optimal Binning");
   cartwoe::g_logger.info("Processing " + std::to_string(feature_cols.size()) + " features");
   cartwoe::g_logger.detail("Parameters:");
   cartwoe::g_logger.detail("  Bins: [" + std::to_string(min_bins) + ", " + 
     std::to_string(max_bins) + "]");
   cartwoe::g_logger.detail("  Method: " + method);
   cartwoe::g_logger.detail("  Criterion: " + criterion);
   cartwoe::g_logger.detail("  Monotonic: " + monotonic_trend);
   cartwoe::g_logger.detail("  Missing Policy: " + miss_policy);
 }
 
 // Input validation
 if (min_bins < 2) {
   stop("min_bins must be >= 2");
 }
 if (max_bins < min_bins) {
   stop("max_bins must be >= min_bins");
 }
 if (min_size <= 0) {
   stop("min_size must be > 0");
 }
 if (smooth < 0) {
   stop("smooth must be >= 0");
 }
 if (criterion != "gini" && criterion != "entropy") {
   stop("criterion must be 'gini' or 'entropy'");
 }
 if (use_pruning && cv_folds < 2) {
   stop("cv_folds must be >= 2 when use_pruning is TRUE");
 }
 
 cartwoe::Criterion crit = (criterion == "gini") ? 
 cartwoe::Criterion::GINI : cartwoe::Criterion::ENTROPY;
 
 cartwoe::MonotonicTrend mono_trend = cartwoe::MonotonicTrend::AUTO;
 if (monotonic_trend == "increasing") {
   mono_trend = cartwoe::MonotonicTrend::INCREASING;
 } else if (monotonic_trend == "decreasing") {
   mono_trend = cartwoe::MonotonicTrend::DECREASING;
 } else if (monotonic_trend == "none") {
   mono_trend = cartwoe::MonotonicTrend::NONE;
 }
 
 int n = data.nrows();
 NumericVector w(n, 1.0);
 if (!weights.isNull()) {
   w = as<NumericVector>(weights);
   if (w.size() != n) {
     stop("weights must have same length as data");
   }
 }
 
 std::set<double> special_set;
 for (int i = 0; i < special_vals.size(); ++i) {
   special_set.insert(special_vals[i]);
 }
 
 List results;
 
#ifdef _OPENMP
 if (parallel && verbose > 0) {
   int max_threads = omp_get_max_threads();
   omp_set_num_threads(max_threads);
   cartwoe::g_logger.info("Parallel processing enabled with " + 
     std::to_string(max_threads) + " threads");
 }
#endif
 
 for (int i = 0; i < feature_cols.size(); ++i) {
   std::string feature_name = as<std::string>(feature_cols[i]);
   
   if (verbose > 0) {
     cartwoe::g_logger.progress(i + 1, feature_cols.size(), feature_name);
   }
   
   List feature_result = cartwoe::process_feature(
     data, target_col, feature_name, min_bins, max_bins,
     method, miss_policy, cat_sep, digits, smooth,
     crit, min_size, use_pruning, cv_folds,
     mono_trend, monotonic_mode, w, special_set,
     max_cat, rare_pct, parallel
   );
   
   results[feature_name] = feature_result;
 }
 
 if (verbose > 0) {
   cartwoe::g_logger.header("Processing Complete");
 }
 
 return results;
}















// // [[Rcpp::plugins(cpp17)]]
// // [[Rcpp::plugins(openmp)]]
// #include <Rcpp.h>
// #include <algorithm>
// #include <cmath>
// #include <functional>
// #include <limits>
// #include <memory>
// #include <numeric>
// #include <queue>
// #include <random>
// #include <set>
// #include <sstream>
// #include <stack>
// #include <string>
// #include <unordered_map>
// #include <vector>
// 
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// 
// using namespace Rcpp;
// 
// namespace cartwoe {
// 
// // =============================================================================
// // CONSTANTS & CONFIGURATION
// // =============================================================================
// 
// constexpr double EPSILON = 1e-10;
// constexpr double WOE_MIN = -20.0;
// constexpr double WOE_MAX = 20.0;
// constexpr double IMPURITY_THRESHOLD = 1e-8;
// constexpr int MAX_TREE_DEPTH = 20;
// constexpr int PARALLEL_THRESHOLD = 10000;  // Use parallel processing for datasets larger than this
// 
// // =============================================================================
// // ENUMS & TYPE DEFINITIONS
// // =============================================================================
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
// enum class Criterion {
//   GINI,
//   ENTROPY
// };
// 
// enum class MissPolicy {
//   SEPARATE,
//   REMOVE,
//   IMPUTE,
//   MERGE
// };
// 
// enum class MonotonicTrend {
//   INCREASING,
//   DECREASING,
//   AUTO,
//   NONE
// };
// 
// // =============================================================================
// // DATA STRUCTURES
// // =============================================================================
// 
// struct DataPoint {
//   double value;
//   double target;
//   double weight;
//   bool is_missing;
//   bool is_special;
//   double special_value;
//   int original_index;
//   std::string category_name;
//   
//   DataPoint() : value(0), target(0), weight(1), is_missing(false), 
//   is_special(false), special_value(0), original_index(-1) {}
// };
// 
// struct BinInfo {
//   int id;
//   std::string label;
//   double lower_bound;
//   double upper_bound;
//   int total_count;
//   int pos_count;
//   int neg_count;
//   double weight_sum;
//   double pos_weight;
//   double neg_weight;
//   double woe;
//   double iv;
//   double ks;
//   double gini;
//   std::vector<std::string> categories;
//   
//   BinInfo() : id(0), lower_bound(-INFINITY), upper_bound(INFINITY),
//   total_count(0), pos_count(0), neg_count(0), weight_sum(0),
//   pos_weight(0), neg_weight(0), woe(0), iv(0), ks(0), gini(0) {}
// };
// 
// struct TreeNode {
//   int id;
//   int depth;
//   bool is_leaf;
//   double threshold;
//   double impurity;
//   double gain;
//   int n_samples;
//   int n_pos;
//   int n_neg;
//   double weight_sum;
//   double pos_weight;
//   double neg_weight;
//   std::vector<int> sample_indices;
//   std::unique_ptr<TreeNode> left;
//   std::unique_ptr<TreeNode> right;
//   
//   TreeNode() : id(0), depth(0), is_leaf(true), threshold(0), impurity(0),
//   gain(0), n_samples(0), n_pos(0), n_neg(0), weight_sum(0),
//   pos_weight(0), neg_weight(0) {}
// };
// 
// struct SplitCandidate {
//   double threshold;
//   double gain;
//   double impurity_left;
//   double impurity_right;
//   int n_left;
//   int n_right;
//   double weight_left;
//   double weight_right;
//   bool valid;
//   
//   SplitCandidate() : threshold(0), gain(-1), impurity_left(0), impurity_right(0),
//   n_left(0), n_right(0), weight_left(0), weight_right(0), valid(false) {}
// };
// 
// // =============================================================================
// // FAST BIN MAPPER FOR EFFICIENT TRANSFORM
// // =============================================================================
// 
// class FastBinMapper {
// private:
//   VariableType var_type_;
//   
//   // For numeric variables
//   std::vector<double> thresholds_;
//   std::vector<int> bin_ids_;
//   std::vector<double> woe_values_;
//   std::vector<std::string> bin_labels_;
//   
//   // For categorical variables
//   std::unordered_map<std::string, int> category_to_bin_;
//   std::unordered_map<std::string, double> category_to_woe_;
//   std::unordered_map<std::string, std::string> category_to_label_;
//   
//   // For special values
//   std::unordered_map<double, int> special_to_bin_;
//   std::unordered_map<double, double> special_to_woe_;
//   std::unordered_map<double, std::string> special_to_label_;
//   
//   int missing_bin_id_;
//   double missing_woe_;
//   std::string missing_label_;
//   bool has_missing_;
//   
// public:
//   FastBinMapper(VariableType var_type) : var_type_(var_type), 
//   missing_bin_id_(-1), 
//   missing_woe_(0), 
//   has_missing_(false) {}
//   
//   void build_from_bins(const std::vector<BinInfo>& bins) {
//     // Reserve space
//     thresholds_.reserve(bins.size());
//     bin_ids_.reserve(bins.size());
//     woe_values_.reserve(bins.size());
//     bin_labels_.reserve(bins.size());
//     
//     for (const auto& bin : bins) {
//       // Handle special bins
//       if (bin.label == "[Missing]") {
//         has_missing_ = true;
//         missing_bin_id_ = bin.id;
//         missing_woe_ = bin.woe;
//         missing_label_ = bin.label;
//         continue;
//       }
//       
//       if (bin.label.find("[Special:") == 0) {
//         // Parse special value
//         std::string val_str = bin.label.substr(9, bin.label.length() - 10);
//         double special_val = std::stod(val_str);
//         special_to_bin_[special_val] = bin.id;
//         special_to_woe_[special_val] = bin.woe;
//         special_to_label_[special_val] = bin.label;
//         continue;
//       }
//       
//       // Regular bins
//       if (var_type_ == VariableType::CATEGORICAL) {
//         // Map each category to its bin
//         for (const auto& cat : bin.categories) {
//           category_to_bin_[cat] = bin.id;
//           category_to_woe_[cat] = bin.woe;
//           category_to_label_[cat] = bin.label;
//         }
//       } else {
//         // For numeric, store thresholds
//         thresholds_.push_back(bin.upper_bound);
//         bin_ids_.push_back(bin.id);
//         woe_values_.push_back(bin.woe);
//         bin_labels_.push_back(bin.label);
//       }
//     }
//   }
//   
//   inline void map_numeric(double value, bool is_missing, bool is_special,
//                           std::string& bin_label, double& woe) const {
//     if (is_missing && has_missing_) {
//       bin_label = missing_label_;
//       woe = missing_woe_;
//       return;
//     }
//     
//     if (is_special) {
//       auto it = special_to_woe_.find(value);
//       if (it != special_to_woe_.end()) {
//         woe = it->second;
//         bin_label = special_to_label_.at(value);
//         return;
//       }
//     }
//     
//     // Binary search for the right bin
//     auto it = std::lower_bound(thresholds_.begin(), thresholds_.end(), value);
//     size_t idx = std::distance(thresholds_.begin(), it);
//     
//     if (idx < woe_values_.size()) {
//       woe = woe_values_[idx];
//       bin_label = bin_labels_[idx];
//     } else if (!woe_values_.empty()) {
//       woe = woe_values_.back();
//       bin_label = bin_labels_.back();
//     }
//   }
//   
//   inline void map_categorical(const std::string& category, bool is_missing,
//                               std::string& bin_label, double& woe) const {
//     if (is_missing && has_missing_) {
//       bin_label = missing_label_;
//       woe = missing_woe_;
//       return;
//     }
//     
//     auto it = category_to_woe_.find(category);
//     if (it != category_to_woe_.end()) {
//       woe = it->second;
//       bin_label = category_to_label_.at(category);
//     } else if (!category_to_woe_.empty()) {
//       // Fallback to first bin
//       woe = category_to_woe_.begin()->second;
//       bin_label = category_to_label_.begin()->second;
//     }
//   }
// };
// 
// // =============================================================================
// // UTILITY FUNCTIONS
// // =============================================================================
// 
// inline double safe_log(double x) {
//   return std::log(std::max(x, EPSILON));
// }
// 
// inline double clip_value(double x, double min_val, double max_val) {
//   return std::max(min_val, std::min(max_val, x));
// }
// 
// VariableType detect_variable_type(SEXP x) {
//   if (Rf_inherits(x, "Date") || Rf_inherits(x, "POSIXct") || 
//       Rf_inherits(x, "POSIXlt") || Rf_inherits(x, "POSIXt")) {
//     return VariableType::DATE_TIME;
//   }
//   
//   if (Rf_isFactor(x)) {
//     return VariableType::CATEGORICAL;
//   }
//   
//   switch (TYPEOF(x)) {
//   case REALSXP: {
//     NumericVector nv(x);
//     int check_limit = std::min(1000, static_cast<int>(nv.size()));
//     std::set<double> unique_vals;
//     for (int i = 0; i < check_limit && unique_vals.size() <= 20; ++i) {
//       if (!NumericVector::is_na(nv[i]) && std::isfinite(nv[i])) {
//         unique_vals.insert(nv[i]);
//       }
//     }
//     return (unique_vals.size() <= 20) ? 
//     VariableType::NUMERIC_DISCRETE : 
//       VariableType::NUMERIC_CONTINUOUS;
//   }
//   case INTSXP: {
//     IntegerVector iv(x);
//     int check_limit = std::min(1000, static_cast<int>(iv.size()));
//     std::set<int> unique_vals;
//     for (int i = 0; i < check_limit && unique_vals.size() <= 20; ++i) {
//       if (!IntegerVector::is_na(iv[i])) {
//         unique_vals.insert(iv[i]);
//       }
//     }
//     return (unique_vals.size() <= 20) ? 
//     VariableType::NUMERIC_DISCRETE : 
//       VariableType::NUMERIC_CONTINUOUS;
//   }
//   case LGLSXP:
//     return VariableType::BOOLEAN;
//   case STRSXP:
//     return VariableType::CATEGORICAL;
//   default:
//     return VariableType::UNSUPPORTED;
//   }
// }
// 
// std::string variable_type_to_string(VariableType vt) {
//   switch (vt) {
//   case VariableType::NUMERIC_CONTINUOUS: return "numeric_continuous";
//   case VariableType::NUMERIC_DISCRETE: return "numeric_discrete";
//   case VariableType::CATEGORICAL: return "categorical";
//   case VariableType::BOOLEAN: return "boolean";
//   case VariableType::DATE_TIME: return "date_time";
//   default: return "unsupported";
//   }
// }
// 
// // =============================================================================
// // IMPURITY CALCULATIONS WITH VECTORIZATION
// // =============================================================================
// 
// class ImpurityCalculator {
// public:
//   static double calculate(Criterion criterion, double pos_weight, 
//                           double neg_weight, double total_weight) {
//     if (total_weight <= EPSILON) return 0.0;
//     
//     double p1 = pos_weight / total_weight;
//     double p0 = neg_weight / total_weight;
//     
//     switch (criterion) {
//     case Criterion::GINI:
//       return 1.0 - p1 * p1 - p0 * p0;
//     case Criterion::ENTROPY:
//       return -(p1 > EPSILON ? p1 * safe_log(p1) : 0.0) -
//         (p0 > EPSILON ? p0 * safe_log(p0) : 0.0);
//     default:
//       return 0.0;
//     }
//   }
//   
//   static double calculate_gain(Criterion criterion, 
//                                double parent_impurity,
//                                double parent_weight,
//                                double left_impurity,
//                                double left_weight,
//                                double right_impurity,
//                                double right_weight) {
//     if (parent_weight <= EPSILON) return 0.0;
//     
//     double weighted_impurity = (left_weight / parent_weight) * left_impurity +
//       (right_weight / parent_weight) * right_impurity;
//     
//     return parent_impurity - weighted_impurity;
//   }
// };
// 
// // =============================================================================
// // ISOTONIC REGRESSION (PAVA)
// // =============================================================================
// 
// class IsotonicRegression {
// public:
//   static std::vector<double> fit(const std::vector<double>& y,
//                                  const std::vector<double>& weights,
//                                  bool increasing = true) {
//     int n = y.size();
//     if (n == 0) return {};
//     
//     std::vector<double> result(n);
//     std::vector<std::pair<double, double>> blocks;
//     blocks.reserve(n);
//     
//     for (int i = 0; i < n; ++i) {
//       double val = increasing ? y[i] : -y[i];
//       blocks.push_back({val * weights[i], weights[i]});
//       
//       while (blocks.size() >= 2) {
//         int last = blocks.size() - 1;
//         double mean_last = blocks[last].first / blocks[last].second;
//         double mean_prev = blocks[last-1].first / blocks[last-1].second;
//         
//         if (mean_prev <= mean_last + EPSILON) break;
//         
//         blocks[last-1].first += blocks[last].first;
//         blocks[last-1].second += blocks[last].second;
//         blocks.pop_back();
//       }
//     }
//     
//     int idx = 0;
//     for (const auto& block : blocks) {
//       double mean = block.first / block.second;
//       if (!increasing) mean = -mean;
//       
//       int block_size = std::round(block.second / weights[idx]);
//       for (int j = 0; j < block_size && idx < n; ++j, ++idx) {
//         result[idx] = mean;
//       }
//     }
//     
//     return result;
//   }
// };
// 
// // =============================================================================
// // OPTIMAL BINNING VIA CART
// // =============================================================================
// 
// class OptimalBinningCART {
// private:
//   std::vector<DataPoint> data_;
//   Criterion criterion_;
//   double min_size_ratio_;
//   int min_size_count_;
//   int min_bins_;
//   int max_bins_;
//   bool use_pruning_;
//   int cv_folds_;
//   MonotonicTrend monotonic_trend_;
//   VariableType var_type_;
//   std::string cat_sep_;
//   std::unique_ptr<TreeNode> root_;
//   std::vector<BinInfo> bins_;
//   std::unordered_map<double, std::string> category_map_;
//   std::vector<std::pair<double, std::string>> ordered_categories_;
//   bool use_parallel_;
//   int unique_classes_count_;
//   
// public:
//   OptimalBinningCART(std::vector<DataPoint>&& data,
//                      Criterion criterion,
//                      double min_size,
//                      int min_bins,
//                      int max_bins,
//                      bool use_pruning,
//                      int cv_folds,
//                      MonotonicTrend monotonic_trend,
//                      VariableType var_type,
//                      const std::string& cat_sep,
//                      int unique_classes_count)
//     : data_(std::move(data)), criterion_(criterion), min_bins_(min_bins),
//       max_bins_(max_bins), use_pruning_(use_pruning), cv_folds_(cv_folds),
//       monotonic_trend_(monotonic_trend), var_type_(var_type), cat_sep_(cat_sep),
//       unique_classes_count_(unique_classes_count) {
//     
//     double total_weight = 0;
//     int total_count = 0;
//     
//     use_parallel_ = data_.size() > PARALLEL_THRESHOLD;
//     
//     if (use_parallel_) {
// #pragma omp parallel for reduction(+:total_weight,total_count)
//       for (size_t i = 0; i < data_.size(); ++i) {
//         if (!data_[i].is_missing && !data_[i].is_special) {
//           total_weight += data_[i].weight;
//           total_count++;
//         }
//       }
//     } else {
//       for (const auto& dp : data_) {
//         if (!dp.is_missing && !dp.is_special) {
//           total_weight += dp.weight;
//           total_count++;
//         }
//       }
//     }
//     
//     if (min_size <= 1.0) {
//       min_size_ratio_ = min_size;
//       min_size_count_ = std::max(1, static_cast<int>(min_size * total_count));
//     } else {
//       min_size_ratio_ = min_size / total_count;
//       min_size_count_ = static_cast<int>(min_size);
//     }
//     
//     bins_.reserve(max_bins_ + 2);
//   }
//   
//   void fit() {
//     if (data_.empty()) return;
//     
//     // If variable has 2 or fewer unique classes, create direct bins without optimization
//     if (unique_classes_count_ <= 2) {
//       create_direct_bins();
//       return;
//     }
//     
//     prepare_data();
//     
//     std::set<double> unique_values;
//     
//     for (const auto& dp : data_) {
//       if (!dp.is_missing && !dp.is_special) {
//         unique_values.insert(dp.value);
//         if (unique_values.size() > static_cast<size_t>(max_bins_)) break;
//       }
//     }
//     
//     int actual_min_bins = std::min(min_bins_, static_cast<int>(unique_values.size()));
//     int actual_max_bins = std::min(max_bins_, static_cast<int>(unique_values.size()));
//     
//     root_ = build_tree(actual_max_bins);
//     
//     if (use_pruning_ && root_) {
//       prune_tree();
//     }
//     
//     extract_bins_from_tree();
//     
//     int attempts = 0;
//     while (static_cast<int>(bins_.size()) < actual_min_bins && attempts < 10) {
//       if (!force_split_largest_bin()) break;
//       attempts++;
//     }
//     
//     if (monotonic_trend_ != MonotonicTrend::NONE && bins_.size() > 1) {
//       enforce_monotonicity();
//     }
//     
//     attempts = 0;
//     while (static_cast<int>(bins_.size()) < actual_min_bins && attempts < 5) {
//       if (!force_split_largest_bin()) break;
//       attempts++;
//     }
//     
//     while (static_cast<int>(bins_.size()) > actual_max_bins && bins_.size() > 1) {
//       merge_adjacent_bins_with_min_iv_loss();
//     }
//   }
//   
//   std::vector<BinInfo> get_bins() { return std::move(bins_); }
//   std::vector<std::pair<double, std::string>> get_ordered_categories() const { 
//     return ordered_categories_; 
//   }
//   
// private:
//   void create_direct_bins() {
//     // For variables with 2 or fewer unique classes, create bins directly
//     bins_.clear();
//     
//     if (var_type_ == VariableType::CATEGORICAL || var_type_ == VariableType::BOOLEAN) {
//       // Create one bin per category
//       std::unordered_map<std::string, std::pair<int, std::pair<int, int>>> category_stats; // cat -> {count, {neg, pos}}
//       
//       for (const auto& dp : data_) {
//         if (!dp.is_missing && !dp.is_special) {
//           auto& stats = category_stats[dp.category_name];
//           stats.first++; // count
//           if (dp.target > 0.5) {
//             stats.second.second++; // pos
//           } else {
//             stats.second.first++; // neg
//           }
//         }
//       }
//       
//       int bin_id = 1;
//       for (const auto& [cat, stats] : category_stats) {
//         BinInfo bin;
//         bin.id = bin_id++;
//         bin.categories.push_back(cat);
//         bin.label = cat;
//         bin.total_count = stats.first;
//         bin.neg_count = stats.second.first;
//         bin.pos_count = stats.second.second;
//         bin.weight_sum = stats.first; // Assuming unit weights for simplicity
//         bin.pos_weight = stats.second.second;
//         bin.neg_weight = stats.second.first;
//         bins_.push_back(std::move(bin));
//       }
//     } else {
//       // For numeric with few unique values, create bins for each unique value
//       std::map<double, std::pair<int, std::pair<int, int>>> value_stats; // value -> {count, {neg, pos}}
//       
//       for (const auto& dp : data_) {
//         if (!dp.is_missing && !dp.is_special) {
//           auto& stats = value_stats[dp.value];
//           stats.first++; // count
//           if (dp.target > 0.5) {
//             stats.second.second++; // pos
//           } else {
//             stats.second.first++; // neg
//           }
//         }
//       }
//       
//       int bin_id = 1;
//       for (const auto& [val, stats] : value_stats) {
//         BinInfo bin;
//         bin.id = bin_id++;
//         bin.lower_bound = val;
//         bin.upper_bound = val;
//         bin.label = std::to_string(val);
//         bin.total_count = stats.first;
//         bin.neg_count = stats.second.first;
//         bin.pos_count = stats.second.second;
//         bin.weight_sum = stats.first;
//         bin.pos_weight = stats.second.second;
//         bin.neg_weight = stats.second.first;
//         bins_.push_back(std::move(bin));
//       }
//     }
//   }
//   
//   void prepare_data() {
//     if (var_type_ == VariableType::CATEGORICAL) {
//       prepare_categorical_data();
//     } else {
//       std::sort(data_.begin(), data_.end(),
//                 [](const DataPoint& a, const DataPoint& b) {
//                   if (a.is_missing != b.is_missing) return b.is_missing;
//                   if (a.is_special != b.is_special) return b.is_special;
//                   return a.value < b.value;
//                 });
//     }
//   }
//   
//   void prepare_categorical_data() {
//     std::unordered_map<std::string, std::pair<double, double>> category_stats;
//     category_stats.reserve(100);
//     
//     for (const auto& dp : data_) {
//       if (!dp.is_missing && !dp.is_special) {
//         auto& stats = category_stats[dp.category_name];
//         stats.first += dp.target * dp.weight;
//         stats.second += dp.weight;
//       }
//     }
//     
//     ordered_categories_.clear();
//     ordered_categories_.reserve(category_stats.size());
//     
//     for (const auto& [cat, stats] : category_stats) {
//       double event_rate = stats.first / (stats.second + EPSILON);
//       ordered_categories_.push_back({event_rate, cat});
//     }
//     
//     std::sort(ordered_categories_.begin(), ordered_categories_.end());
//     
//     category_map_.clear();
//     category_map_.reserve(ordered_categories_.size());
//     
//     for (size_t i = 0; i < ordered_categories_.size(); ++i) {
//       category_map_[static_cast<double>(i)] = ordered_categories_[i].second;
//     }
//     
//     if (use_parallel_) {
// #pragma omp parallel for
//       for (size_t j = 0; j < data_.size(); ++j) {
//         if (!data_[j].is_missing && !data_[j].is_special) {
//           for (size_t i = 0; i < ordered_categories_.size(); ++i) {
//             if (ordered_categories_[i].second == data_[j].category_name) {
//               data_[j].value = static_cast<double>(i);
//               break;
//             }
//           }
//         }
//       }
//     } else {
//       for (auto& dp : data_) {
//         if (!dp.is_missing && !dp.is_special) {
//           for (size_t i = 0; i < ordered_categories_.size(); ++i) {
//             if (ordered_categories_[i].second == dp.category_name) {
//               dp.value = static_cast<double>(i);
//               break;
//             }
//           }
//         }
//       }
//     }
//     
//     std::sort(data_.begin(), data_.end(),
//               [](const DataPoint& a, const DataPoint& b) {
//                 if (a.is_missing != b.is_missing) return b.is_missing;
//                 if (a.is_special != b.is_special) return b.is_special;
//                 return a.value < b.value;
//               });
//   }
//   
//   std::unique_ptr<TreeNode> build_tree(int target_leaves) {
//     auto root = std::make_unique<TreeNode>();
//     root->id = 0;
//     root->depth = 0;
//     
//     root->sample_indices.reserve(data_.size());
//     
//     for (size_t i = 0; i < data_.size(); ++i) {
//       if (!data_[i].is_missing && !data_[i].is_special) {
//         root->sample_indices.push_back(i);
//       }
//     }
//     
//     if (root->sample_indices.empty()) return root;
//     
//     update_node_statistics(root.get());
//     
//     if (root->sample_indices.size() < 2 * min_size_count_) {
//       return root;
//     }
//     
//     using NodePair = std::pair<double, TreeNode*>;
//     std::priority_queue<NodePair> expansion_queue;
//     
//     auto best_split = find_best_split(root.get());
//     if (best_split.valid && best_split.gain > IMPURITY_THRESHOLD) {
//       root->gain = best_split.gain;
//       expansion_queue.push({root->gain, root.get()});
//     }
//     
//     int node_count = 1;
//     int leaf_count = 1;
//     
//     while (!expansion_queue.empty() && leaf_count < target_leaves) {
//       auto [gain, node] = expansion_queue.top();
//       expansion_queue.pop();
//       
//       if (gain <= IMPURITY_THRESHOLD) break;
//       
//       auto split = find_best_split(node);
//       if (!split.valid) continue;
//       
//       node->is_leaf = false;
//       node->threshold = split.threshold;
//       node->gain = split.gain;
//       
//       node->left = std::make_unique<TreeNode>();
//       node->right = std::make_unique<TreeNode>();
//       
//       node->left->id = node_count++;
//       node->right->id = node_count++;
//       node->left->depth = node->depth + 1;
//       node->right->depth = node->depth + 1;
//       
//       node->left->sample_indices.reserve(split.n_left);
//       node->right->sample_indices.reserve(split.n_right);
//       
//       for (int idx : node->sample_indices) {
//         if (data_[idx].value < node->threshold) {
//           node->left->sample_indices.push_back(idx);
//         } else {
//           node->right->sample_indices.push_back(idx);
//         }
//       }
//       
//       update_node_statistics(node->left.get());
//       update_node_statistics(node->right.get());
//       
//       leaf_count++;
//       
//       if (leaf_count < target_leaves) {
//         auto left_split = find_best_split(node->left.get());
//         if (left_split.valid && left_split.gain > IMPURITY_THRESHOLD) {
//           node->left->gain = left_split.gain;
//           expansion_queue.push({left_split.gain, node->left.get()});
//         }
//         
//         auto right_split = find_best_split(node->right.get());
//         if (right_split.valid && right_split.gain > IMPURITY_THRESHOLD) {
//           node->right->gain = right_split.gain;
//           expansion_queue.push({right_split.gain, node->right.get()});
//         }
//       }
//     }
//     
//     return root;
//   }
//   
//   void update_node_statistics(TreeNode* node) {
//     node->n_samples = node->sample_indices.size();
//     node->n_pos = 0;
//     node->n_neg = 0;
//     node->weight_sum = 0;
//     node->pos_weight = 0;
//     node->neg_weight = 0;
//     
//     for (int idx : node->sample_indices) {
//       const auto& dp = data_[idx];
//       node->weight_sum += dp.weight;
//       
//       if (dp.target > 0.5) {
//         node->n_pos++;
//         node->pos_weight += dp.weight;
//       } else {
//         node->n_neg++;
//         node->neg_weight += dp.weight;
//       }
//     }
//     
//     node->impurity = ImpurityCalculator::calculate(
//       criterion_, node->pos_weight, node->neg_weight, node->weight_sum);
//   }
//   
//   SplitCandidate find_best_split(TreeNode* node) {
//     SplitCandidate best_split;
//     best_split.valid = false;
//     
//     if (!node || node->sample_indices.size() < 2 * min_size_count_) {
//       return best_split;
//     }
//     
//     std::vector<int> sorted_indices = node->sample_indices;
//     std::sort(sorted_indices.begin(), sorted_indices.end(),
//               [this](int a, int b) { return data_[a].value < data_[b].value; });
//     
//     size_t n = sorted_indices.size();
//     std::vector<double> values(n);
//     std::vector<double> cum_pos(n + 1, 0);
//     std::vector<double> cum_neg(n + 1, 0);
//     std::vector<double> cum_weight(n + 1, 0);
//     
//     for (size_t i = 0; i < n; ++i) {
//       const auto& dp = data_[sorted_indices[i]];
//       values[i] = dp.value;
//       
//       cum_weight[i + 1] = cum_weight[i] + dp.weight;
//       if (dp.target > 0.5) {
//         cum_pos[i + 1] = cum_pos[i] + dp.weight;
//         cum_neg[i + 1] = cum_neg[i];
//       } else {
//         cum_pos[i + 1] = cum_pos[i];
//         cum_neg[i + 1] = cum_neg[i] + dp.weight;
//       }
//     }
//     
//     best_split.gain = -1;
//     
//     if (use_parallel_ && n > 1000) {
// #ifdef _OPENMP
//       std::vector<SplitCandidate> thread_best_splits(omp_get_max_threads());
//       
// #pragma omp parallel
// {
//   int tid = omp_get_thread_num();
//   thread_best_splits[tid].gain = -1;
//   
// #pragma omp for schedule(dynamic, 100)
//   for (size_t i = min_size_count_; i <= n - min_size_count_; ++i) {
//     if (i > 0 && std::abs(values[i] - values[i-1]) < EPSILON) continue;
//     
//     double left_weight = cum_weight[i];
//     double right_weight = cum_weight.back() - left_weight;
//     
//     if (left_weight < min_size_ratio_ * node->weight_sum) continue;
//     if (right_weight < min_size_ratio_ * node->weight_sum) continue;
//     
//     double left_pos = cum_pos[i];
//     double left_neg = cum_neg[i];
//     double right_pos = cum_pos.back() - left_pos;
//     double right_neg = cum_neg.back() - left_neg;
//     
//     double left_impurity = ImpurityCalculator::calculate(
//       criterion_, left_pos, left_neg, left_weight);
//     double right_impurity = ImpurityCalculator::calculate(
//       criterion_, right_pos, right_neg, right_weight);
//     
//     double gain = ImpurityCalculator::calculate_gain(
//       criterion_, node->impurity, node->weight_sum,
//       left_impurity, left_weight, right_impurity, right_weight);
//     
//     if (gain > thread_best_splits[tid].gain) {
//       thread_best_splits[tid].gain = gain;
//       thread_best_splits[tid].threshold = (var_type_ == VariableType::CATEGORICAL) ?
//       values[i-1] + 0.5 : (i > 0 ? (values[i-1] + values[i]) / 2.0 : values[i]);
//       thread_best_splits[tid].impurity_left = left_impurity;
//       thread_best_splits[tid].impurity_right = right_impurity;
//       thread_best_splits[tid].n_left = i;
//       thread_best_splits[tid].n_right = n - i;
//       thread_best_splits[tid].weight_left = left_weight;
//       thread_best_splits[tid].weight_right = right_weight;
//       thread_best_splits[tid].valid = true;
//     }
//   }
// }
// 
// for (const auto& split : thread_best_splits) {
//   if (split.gain > best_split.gain) {
//     best_split = split;
//   }
// }
// #endif
//     } else {
//       for (size_t i = min_size_count_; i <= n - min_size_count_; ++i) {
//         if (i > 0 && std::abs(values[i] - values[i-1]) < EPSILON) continue;
//         
//         double left_weight = cum_weight[i];
//         double right_weight = cum_weight.back() - left_weight;
//         
//         if (left_weight < min_size_ratio_ * node->weight_sum) continue;
//         if (right_weight < min_size_ratio_ * node->weight_sum) continue;
//         
//         double left_pos = cum_pos[i];
//         double left_neg = cum_neg[i];
//         double right_pos = cum_pos.back() - left_pos;
//         double right_neg = cum_neg.back() - left_neg;
//         
//         double left_impurity = ImpurityCalculator::calculate(
//           criterion_, left_pos, left_neg, left_weight);
//         double right_impurity = ImpurityCalculator::calculate(
//           criterion_, right_pos, right_neg, right_weight);
//         
//         double gain = ImpurityCalculator::calculate_gain(
//           criterion_, node->impurity, node->weight_sum,
//           left_impurity, left_weight, right_impurity, right_weight);
//         
//         if (gain > best_split.gain) {
//           best_split.gain = gain;
//           best_split.threshold = (var_type_ == VariableType::CATEGORICAL) ?
//           values[i-1] + 0.5 : (i > 0 ? (values[i-1] + values[i]) / 2.0 : values[i]);
//           best_split.impurity_left = left_impurity;
//           best_split.impurity_right = right_impurity;
//           best_split.n_left = i;
//           best_split.n_right = n - i;
//           best_split.weight_left = left_weight;
//           best_split.weight_right = right_weight;
//           best_split.valid = true;
//         }
//       }
//     }
//     
//     return best_split;
//   }
//   
//   void prune_tree() {
//     if (!root_) return;
//     
//     int leaf_count = count_leaves(root_.get());
//     if (leaf_count <= max_bins_) return;
//     
//     while (leaf_count > max_bins_) {
//       prune_weakest_split(root_.get());
//       leaf_count = count_leaves(root_.get());
//     }
//   }
//   
//   int count_leaves(TreeNode* node) {
//     if (!node) return 0;
//     if (node->is_leaf) return 1;
//     return count_leaves(node->left.get()) + count_leaves(node->right.get());
//   }
//   
//   void prune_weakest_split(TreeNode* node) {
//     if (!node || node->is_leaf) return;
//     
//     TreeNode* weakest_node = nullptr;
//     double min_gain = INFINITY;
//     
//     std::function<void(TreeNode*)> find_weakest = [&](TreeNode* n) {
//       if (!n || n->is_leaf) return;
//       
//       if (n->gain < min_gain) {
//         min_gain = n->gain;
//         weakest_node = n;
//       }
//       
//       find_weakest(n->left.get());
//       find_weakest(n->right.get());
//     };
//     
//     find_weakest(node);
//     
//     if (weakest_node) {
//       weakest_node->is_leaf = true;
//       weakest_node->left.reset();
//       weakest_node->right.reset();
//     }
//   }
//   
//   void extract_bins_from_tree() {
//     bins_.clear();
//     if (!root_) return;
//     
//     std::vector<std::pair<TreeNode*, std::pair<double, double>>> leaves;
//     leaves.reserve(max_bins_);
//     
//     std::function<void(TreeNode*, double, double)> collect_leaves = 
//       [&](TreeNode* node, double lower, double upper) {
//         if (!node) return;
//         
//         if (node->is_leaf) {
//           leaves.push_back({node, {lower, upper}});
//         } else {
//           collect_leaves(node->left.get(), lower, node->threshold);
//           collect_leaves(node->right.get(), node->threshold, upper);
//         }
//       };
//       
//       collect_leaves(root_.get(), -INFINITY, INFINITY);
//       
//       std::sort(leaves.begin(), leaves.end(),
//                 [](const auto& a, const auto& b) {
//                   return a.second.first < b.second.first;
//                 });
//       
//       int bin_id = 1;
//       for (size_t i = 0; i < leaves.size(); ++i) {
//         BinInfo bin;
//         bin.id = bin_id++;
//         bin.lower_bound = leaves[i].second.first;
//         bin.upper_bound = leaves[i].second.second;
//         
//         if (var_type_ == VariableType::CATEGORICAL) {
//           for (double ord = std::floor(bin.lower_bound < 0 ? 0 : bin.lower_bound); 
//                ord < bin.upper_bound && ord < ordered_categories_.size(); ord++) {
//             if (category_map_.count(ord)) {
//               bin.categories.push_back(category_map_[ord]);
//             }
//           }
//         }
//         
//         bin.total_count = leaves[i].first->n_samples;
//         bin.pos_count = leaves[i].first->n_pos;
//         bin.neg_count = leaves[i].first->n_neg;
//         bin.weight_sum = leaves[i].first->weight_sum;
//         bin.pos_weight = leaves[i].first->pos_weight;
//         bin.neg_weight = leaves[i].first->neg_weight;
//         
//         bins_.push_back(std::move(bin));
//       }
//   }
//   
//   bool force_split_largest_bin() {
//     int largest_idx = -1;
//     int max_count = 2 * min_size_count_;
//     
//     for (size_t i = 0; i < bins_.size(); ++i) {
//       if (bins_[i].total_count >= max_count) {
//         max_count = bins_[i].total_count;
//         largest_idx = i;
//       }
//     }
//     
//     if (largest_idx < 0) return false;
//     
//     std::vector<DataPoint> bin_data;
//     bin_data.reserve(max_count);
//     
//     for (const auto& dp : data_) {
//       if (!dp.is_missing && !dp.is_special &&
//           dp.value >= bins_[largest_idx].lower_bound &&
//           dp.value < bins_[largest_idx].upper_bound) {
//         bin_data.push_back(dp);
//       }
//     }
//     
//     if (bin_data.size() < 2 * min_size_count_) return false;
//     
//     std::set<double> unique_vals;
//     for (const auto& dp : bin_data) {
//       unique_vals.insert(dp.value);
//     }
//     
//     if (unique_vals.size() < 2) return false;
//     
//     std::sort(bin_data.begin(), bin_data.end(),
//               [](const auto& a, const auto& b) { return a.value < b.value; });
//     
//     double split_value = -1;
//     for (size_t i = min_size_count_; i <= bin_data.size() - min_size_count_; ++i) {
//       if (i > 0 && bin_data[i].value != bin_data[i-1].value) {
//         split_value = (bin_data[i-1].value + bin_data[i].value) / 2.0;
//         break;
//       }
//     }
//     
//     if (split_value < 0) {
//       split_value = bin_data[bin_data.size() / 2].value;
//     }
//     
//     BinInfo left_bin = bins_[largest_idx];
//     BinInfo right_bin = bins_[largest_idx];
//     
//     left_bin.upper_bound = split_value;
//     right_bin.lower_bound = split_value;
//     right_bin.id = bins_.size() + 1;
//     
//     left_bin.categories.clear();
//     right_bin.categories.clear();
//     
//     left_bin.total_count = 0;
//     left_bin.pos_count = 0;
//     left_bin.neg_count = 0;
//     left_bin.weight_sum = 0;
//     left_bin.pos_weight = 0;
//     left_bin.neg_weight = 0;
//     
//     right_bin.total_count = 0;
//     right_bin.pos_count = 0;
//     right_bin.neg_count = 0;
//     right_bin.weight_sum = 0;
//     right_bin.pos_weight = 0;
//     right_bin.neg_weight = 0;
//     
//     for (const auto& dp : bin_data) {
//       if (dp.value < split_value) {
//         left_bin.total_count++;
//         left_bin.weight_sum += dp.weight;
//         if (dp.target > 0.5) {
//           left_bin.pos_count++;
//           left_bin.pos_weight += dp.weight;
//         } else {
//           left_bin.neg_count++;
//           left_bin.neg_weight += dp.weight;
//         }
//         
//         if (var_type_ == VariableType::CATEGORICAL && category_map_.count(dp.value)) {
//           if (std::find(left_bin.categories.begin(), left_bin.categories.end(), 
//                         category_map_[dp.value]) == left_bin.categories.end()) {
//             left_bin.categories.push_back(category_map_[dp.value]);
//           }
//         }
//       } else {
//         right_bin.total_count++;
//         right_bin.weight_sum += dp.weight;
//         if (dp.target > 0.5) {
//           right_bin.pos_count++;
//           right_bin.pos_weight += dp.weight;
//         } else {
//           right_bin.neg_count++;
//           right_bin.neg_weight += dp.weight;
//         }
//         
//         if (var_type_ == VariableType::CATEGORICAL && category_map_.count(dp.value)) {
//           if (std::find(right_bin.categories.begin(), right_bin.categories.end(), 
//                         category_map_[dp.value]) == right_bin.categories.end()) {
//             right_bin.categories.push_back(category_map_[dp.value]);
//           }
//         }
//       }
//     }
//     
//     bins_[largest_idx] = std::move(left_bin);
//     bins_.insert(bins_.begin() + largest_idx + 1, std::move(right_bin));
//     
//     for (size_t i = 0; i < bins_.size(); ++i) {
//       bins_[i].id = i + 1;
//     }
//     
//     return true;
//   }
//   
//   void enforce_monotonicity() {
//     if (bins_.size() <= 1) return;
//     
//     MonotonicTrend trend = monotonic_trend_;
//     if (trend == MonotonicTrend::AUTO) {
//       trend = detect_monotonic_trend();
//     }
//     
//     if (trend == MonotonicTrend::NONE) return;
//     
//     std::vector<double> event_rates(bins_.size());
//     std::vector<double> weights(bins_.size());
//     
//     for (size_t i = 0; i < bins_.size(); ++i) {
//       event_rates[i] = bins_[i].pos_weight / (bins_[i].weight_sum + EPSILON);
//       weights[i] = bins_[i].weight_sum;
//     }
//     
//     auto monotonic_rates = IsotonicRegression::fit(
//       event_rates, weights, trend == MonotonicTrend::INCREASING);
//     
//     std::vector<BinInfo> new_bins;
//     new_bins.reserve(bins_.size());
//     
//     BinInfo current_bin = bins_[0];
//     
//     for (size_t i = 1; i < bins_.size(); ++i) {
//       if (std::abs(monotonic_rates[i] - monotonic_rates[i-1]) < EPSILON) {
//         current_bin.upper_bound = bins_[i].upper_bound;
//         current_bin.total_count += bins_[i].total_count;
//         current_bin.pos_count += bins_[i].pos_count;
//         current_bin.neg_count += bins_[i].neg_count;
//         current_bin.weight_sum += bins_[i].weight_sum;
//         current_bin.pos_weight += bins_[i].pos_weight;
//         current_bin.neg_weight += bins_[i].neg_weight;
//         
//         for (const auto& cat : bins_[i].categories) {
//           if (std::find(current_bin.categories.begin(), current_bin.categories.end(), cat) 
//                 == current_bin.categories.end()) {
//             current_bin.categories.push_back(cat);
//           }
//         }
//       } else {
//         new_bins.push_back(std::move(current_bin));
//         current_bin = bins_[i];
//       }
//     }
//     
//     new_bins.push_back(std::move(current_bin));
//     
//     for (size_t i = 0; i < new_bins.size(); ++i) {
//       new_bins[i].id = i + 1;
//     }
//     
//     bins_ = std::move(new_bins);
//   }
//   
//   MonotonicTrend detect_monotonic_trend() {
//     if (bins_.size() <= 1) return MonotonicTrend::NONE;
//     
//     std::vector<double> centroids(bins_.size());
//     std::vector<double> event_rates(bins_.size());
//     
//     for (size_t i = 0; i < bins_.size(); ++i) {
//       double centroid = (bins_[i].lower_bound + bins_[i].upper_bound) / 2.0;
//       if (!std::isfinite(centroid)) {
//         centroid = std::isfinite(bins_[i].lower_bound) ? 
//         bins_[i].lower_bound + 1.0 : bins_[i].upper_bound - 1.0;
//       }
//       
//       centroids[i] = centroid;
//       event_rates[i] = bins_[i].pos_weight / (bins_[i].weight_sum + EPSILON);
//     }
//     
//     auto rank = [](std::vector<double>& v) {
//       std::vector<std::pair<double, size_t>> indexed;
//       indexed.reserve(v.size());
//       
//       for (size_t i = 0; i < v.size(); ++i) {
//         indexed.push_back({v[i], i});
//       }
//       std::sort(indexed.begin(), indexed.end());
//       
//       std::vector<double> ranks(v.size());
//       for (size_t i = 0; i < indexed.size(); ++i) {
//         ranks[indexed[i].second] = i + 1;
//       }
//       return ranks;
//     };
//     
//     auto rank_centroids = rank(centroids);
//     auto rank_rates = rank(event_rates);
//     
//     double sum_sq_diff = 0;
//     for (size_t i = 0; i < rank_centroids.size(); ++i) {
//       double diff = rank_centroids[i] - rank_rates[i];
//       sum_sq_diff += diff * diff;
//     }
//     
//     int n = rank_centroids.size();
//     double spearman = 1.0 - (6.0 * sum_sq_diff) / (n * (n * n - 1));
//     
//     if (spearman > 0.3) return MonotonicTrend::INCREASING;
//     if (spearman < -0.3) return MonotonicTrend::DECREASING;
//     
//     return MonotonicTrend::NONE;
//   }
//   
//   void merge_adjacent_bins_with_min_iv_loss() {
//     if (bins_.size() <= 1) return;
//     
//     double min_loss = INFINITY;
//     size_t merge_idx = 0;
//     
//     for (size_t i = 0; i < bins_.size() - 1; ++i) {
//       double current_iv = bins_[i].iv + bins_[i+1].iv;
//       double loss = std::abs(current_iv);
//       
//       if (loss < min_loss) {
//         min_loss = loss;
//         merge_idx = i;
//       }
//     }
//     
//     bins_[merge_idx].upper_bound = bins_[merge_idx + 1].upper_bound;
//     bins_[merge_idx].total_count += bins_[merge_idx + 1].total_count;
//     bins_[merge_idx].pos_count += bins_[merge_idx + 1].pos_count;
//     bins_[merge_idx].neg_count += bins_[merge_idx + 1].neg_count;
//     bins_[merge_idx].weight_sum += bins_[merge_idx + 1].weight_sum;
//     bins_[merge_idx].pos_weight += bins_[merge_idx + 1].pos_weight;
//     bins_[merge_idx].neg_weight += bins_[merge_idx + 1].neg_weight;
//     
//     for (const auto& cat : bins_[merge_idx + 1].categories) {
//       if (std::find(bins_[merge_idx].categories.begin(), bins_[merge_idx].categories.end(), cat) 
//             == bins_[merge_idx].categories.end()) {
//         bins_[merge_idx].categories.push_back(cat);
//       }
//     }
//     
//     bins_.erase(bins_.begin() + merge_idx + 1);
//     
//     for (size_t i = 0; i < bins_.size(); ++i) {
//       bins_[i].id = i + 1;
//     }
//   }
// };
// 
// // =============================================================================
// // WOE/IV/KS/GINI CALCULATIONS
// // =============================================================================
// 
// class MetricsCalculator {
// public:
//   static void calculate_metrics(std::vector<BinInfo>& bins, 
//                                 double smooth, 
//                                 int total_pos, 
//                                 int total_neg) {
//     if (bins.empty() || total_pos == 0 || total_neg == 0) return;
//     
//     int G = bins.size();
//     double N0 = total_neg + G * smooth;
//     double N1 = total_pos + G * smooth;
//     
//     double cum_pos = 0, cum_neg = 0;
//     
//     for (auto& bin : bins) {
//       double p_pos = (bin.pos_count + smooth) / N1;
//       double p_neg = (bin.neg_count + smooth) / N0;
//       
//       bin.woe = safe_log(p_pos / p_neg);
//       bin.woe = clip_value(bin.woe, WOE_MIN, WOE_MAX);
//       
//       double actual_p_pos = static_cast<double>(bin.pos_count) / (total_pos + EPSILON);
//       double actual_p_neg = static_cast<double>(bin.neg_count) / (total_neg + EPSILON);
//       bin.iv = (actual_p_pos - actual_p_neg) * bin.woe;
//       
//       cum_pos += bin.pos_count;
//       cum_neg += bin.neg_count;
//       
//       double cdf_pos = cum_pos / (total_pos + EPSILON);
//       double cdf_neg = cum_neg / (total_neg + EPSILON);
//       bin.ks = std::abs(cdf_pos - cdf_neg);
//       
//       double p0 = static_cast<double>(bin.neg_count) / (bin.total_count + EPSILON);
//       double p1 = static_cast<double>(bin.pos_count) / (bin.total_count + EPSILON);
//       bin.gini = std::abs(p1 - p0);
//     }
//   }
//   
//   static double calculate_total_iv(const std::vector<BinInfo>& bins) {
//     double total = 0;
//     for (const auto& bin : bins) {
//       total += bin.iv;
//     }
//     return total;
//   }
// };
// 
// // =============================================================================
// // MAIN CART_WOE IMPLEMENTATION
// // =============================================================================
// 
// List process_feature(
//     const DataFrame& data,
//     const std::string& target_col,
//     const std::string& feature_col,
//     int min_bins,
//     int max_bins,
//     const std::string& method,
//     const std::string& miss_policy,
//     const std::string& cat_sep,
//     int digits,
//     double smooth,
//     Criterion criterion,
//     double min_size,
//     bool use_pruning,
//     int cv_folds,
//     MonotonicTrend monotonic_trend,
//     const std::string& monotonic_mode,
//     const NumericVector& weights,
//     const std::set<double>& special_vals,
//     int max_cat,
//     double rare_pct,
//     bool use_parallel) {
//   
//   NumericVector target = as<NumericVector>(data[target_col]);
//   SEXP feature_sexp = data[feature_col];
//   
//   VariableType var_type = detect_variable_type(feature_sexp);
//   
//   if (var_type == VariableType::DATE_TIME || var_type == VariableType::UNSUPPORTED) {
//     return List::create(
//       Named("bins") = DataFrame::create(),
//       Named("total_iv") = 0.0,
//       Named("variable_type") = variable_type_to_string(var_type),
//       Named("method_used") = method,
//       Named("n_bins") = 0,
//       Named("total_samples") = 0,
//       Named("event_rate") = 0.0,
//       Named("messages") = "Variable skipped: " + variable_type_to_string(var_type),
//       Named("transform") = DataFrame::create()
//     );
//   }
//   
//   // Count unique classes/values
//   int unique_classes_count = 0;
//   if (var_type == VariableType::BOOLEAN) {
//     unique_classes_count = 2; // Always 2 for boolean
//   } else if (var_type == VariableType::CATEGORICAL) {
//     if (Rf_isFactor(feature_sexp)) {
//       IntegerVector iv(feature_sexp);
//       std::set<int> unique_vals;
//       for (int i = 0; i < iv.size(); ++i) {
//         if (!IntegerVector::is_na(iv[i])) {
//           unique_vals.insert(iv[i]);
//         }
//       }
//       unique_classes_count = unique_vals.size();
//     } else {
//       CharacterVector cv(feature_sexp);
//       std::set<std::string> unique_vals;
//       for (int i = 0; i < cv.size(); ++i) {
//         if (!CharacterVector::is_na(cv[i])) {
//           unique_vals.insert(as<std::string>(cv[i]));
//         }
//       }
//       unique_classes_count = unique_vals.size();
//     }
//   } else {
//     // For numeric variables
//     NumericVector nv(feature_sexp);
//     std::set<double> unique_vals;
//     int check_limit = std::min(1000, static_cast<int>(nv.size()));
//     for (int i = 0; i < check_limit; ++i) {
//       if (!NumericVector::is_na(nv[i]) && std::isfinite(nv[i])) {
//         unique_vals.insert(nv[i]);
//         if (unique_vals.size() > 2) break; // Early exit if more than 2
//       }
//     }
//     unique_classes_count = unique_vals.size();
//   }
//   
//   // If 2 or fewer unique classes, create direct bins without optimization
//   bool is_simple_variable = (unique_classes_count <= 2);
//   
//   // Prepare data points
//   std::vector<DataPoint> data_points;
//   data_points.reserve(target.size());
//   
//   int total_valid_pos = 0, total_valid_neg = 0;
//   
//   std::vector<std::string> original_categorical_values;
//   std::vector<double> original_numeric_values;
//   
//   if (var_type == VariableType::CATEGORICAL) {
//     original_categorical_values.resize(target.size());
//   } else {
//     original_numeric_values.resize(target.size());
//   }
//   
//   for (int i = 0; i < target.size(); ++i) {
//     DataPoint dp;
//     dp.target = target[i];
//     dp.weight = weights[i];
//     dp.original_index = i;
//     
//     bool is_missing = false;
//     double value = 0;
//     
//     switch (var_type) {
//     case VariableType::NUMERIC_CONTINUOUS:
//     case VariableType::NUMERIC_DISCRETE: {
//       NumericVector nv(feature_sexp);
//       if (NumericVector::is_na(nv[i]) || !std::isfinite(nv[i])) {
//         is_missing = true;
//         original_numeric_values[i] = NA_REAL;
//       } else {
//         value = nv[i];
//         original_numeric_values[i] = value;
//         if (special_vals.count(value)) {
//           dp.is_special = true;
//           dp.special_value = value;
//         }
//       }
//       break;
//     }
//     case VariableType::CATEGORICAL: {
//       if (Rf_isFactor(feature_sexp)) {
//       IntegerVector iv(feature_sexp);
//       if (IntegerVector::is_na(iv[i])) {
//         is_missing = true;
//         original_categorical_values[i] = "NA";
//       } else {
//         CharacterVector levels = iv.attr("levels");
//         dp.category_name = as<std::string>(levels[iv[i] - 1]);
//         original_categorical_values[i] = dp.category_name;
//         value = iv[i];
//       }
//     } else {
//       CharacterVector cv(feature_sexp);
//       if (CharacterVector::is_na(cv[i])) {
//         is_missing = true;
//         original_categorical_values[i] = "NA";
//       } else {
//         dp.category_name = as<std::string>(cv[i]);
//         original_categorical_values[i] = dp.category_name;
//         std::hash<std::string> hasher;
//         value = static_cast<double>(hasher(dp.category_name) % 1000000);
//       }
//     }
//     break;
//     }
//     case VariableType::BOOLEAN: {
//       LogicalVector lv(feature_sexp);
//       if (LogicalVector::is_na(lv[i])) {
//         is_missing = true;
//         original_numeric_values[i] = NA_REAL;
//       } else {
//         value = lv[i] ? 1.0 : 0.0;
//         original_numeric_values[i] = value;
//         // Also store category name for boolean
//         dp.category_name = lv[i] ? "TRUE" : "FALSE";
//       }
//       break;
//     }
//     default:
//       break;
//     }
//     
//     dp.is_missing = is_missing;
//     dp.value = value;
//     
//     if (!is_missing && !dp.is_special) {
//       if (dp.target > 0.5) {
//         total_valid_pos++;
//       } else {
//         total_valid_neg++;
//       }
//     }
//     
//     data_points.push_back(std::move(dp));
//   }
//   
//   // Handle missing values
//   std::vector<DataPoint> processed_data;
//   processed_data.reserve(data_points.size());
//   
//   std::vector<BinInfo> special_bins;
//   special_bins.reserve(10);
//   
//   if (miss_policy == "separate") {
//     BinInfo missing_bin;
//     missing_bin.id = 0;
//     missing_bin.label = "[Missing]";
//     
//     std::unordered_map<double, BinInfo> special_bin_map;
//     
//     for (const auto& dp : data_points) {
//       if (dp.is_missing) {
//         missing_bin.total_count++;
//         if (dp.target > 0.5) {
//           missing_bin.pos_count++;
//           missing_bin.pos_weight += dp.weight;
//         } else {
//           missing_bin.neg_count++;
//           missing_bin.neg_weight += dp.weight;
//         }
//         missing_bin.weight_sum += dp.weight;
//       } else if (dp.is_special) {
//         auto& sbin = special_bin_map[dp.special_value];
//         sbin.total_count++;
//         if (dp.target > 0.5) {
//           sbin.pos_count++;
//           sbin.pos_weight += dp.weight;
//         } else {
//           sbin.neg_count++;
//           sbin.neg_weight += dp.weight;
//         }
//         sbin.weight_sum += dp.weight;
//       } else {
//         processed_data.push_back(dp);
//       }
//     }
//     
//     if (missing_bin.total_count > 0) {
//       special_bins.push_back(std::move(missing_bin));
//     }
//     
//     for (auto& [val, sbin] : special_bin_map) {
//       sbin.label = "[Special:" + std::to_string(val) + "]";
//       special_bins.push_back(std::move(sbin));
//     }
//   } else {
//     for (const auto& dp : data_points) {
//       if (!dp.is_missing && !dp.is_special) {
//         processed_data.push_back(dp);
//       }
//     }
//   }
//   
//   // Build and fit the binning model
//   std::vector<BinInfo> regular_bins;
//   std::string message = "Success";
//   
//   if (!processed_data.empty()) {
//     if (is_simple_variable) {
//       // For simple variables, create direct bins
//       OptimalBinningCART binner(
//           std::move(processed_data),
//           criterion, min_size, min_bins, max_bins,
//           use_pruning, cv_folds, monotonic_trend, var_type, cat_sep,
//           unique_classes_count
//       );
//       
//       binner.fit();
//       regular_bins = binner.get_bins();
//       
//       // Create appropriate labels
//       for (auto& bin : regular_bins) {
//         if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN) {
//           if (!bin.categories.empty()) {
//             bin.label = bin.categories[0];
//           }
//         } else {
//           std::stringstream ss;
//           ss.precision(digits);
//           ss << std::fixed << bin.lower_bound;
//           bin.label = ss.str();
//         }
//       }
//       
//       message = "Variable has " + std::to_string(unique_classes_count) + 
//         " unique classes. Direct binning applied without optimization.";
//     } else {
//       // For complex variables, use full optimization
//       OptimalBinningCART binner(
//           std::move(processed_data),
//           criterion, min_size, min_bins, max_bins,
//           use_pruning, cv_folds, monotonic_trend, var_type, cat_sep,
//           unique_classes_count
//       );
//       
//       binner.fit();
//       regular_bins = binner.get_bins();
//       
//       // Create labels for bins
//       for (auto& bin : regular_bins) {
//         std::stringstream ss;
//         ss.precision(digits);
//         
//         if (var_type == VariableType::CATEGORICAL && !bin.categories.empty()) {
//           for (size_t i = 0; i < bin.categories.size(); ++i) {
//             if (i > 0) ss << cat_sep;
//             ss << bin.categories[i];
//           }
//           bin.label = ss.str();
//         } else {
//           if (std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
//             ss << "All";
//           } else if (std::isinf(bin.lower_bound)) {
//             ss << "(-inf, " << std::fixed << bin.upper_bound << ")";
//           } else if (std::isinf(bin.upper_bound)) {
//             ss << "[" << std::fixed << bin.lower_bound << ", +inf)";
//           } else {
//             ss << "[" << std::fixed << bin.lower_bound << ", " 
//                << std::fixed << bin.upper_bound << ")";
//           }
//           bin.label = ss.str();
//         }
//       }
//     }
//   }
//   
//   // Combine special and regular bins
//   std::vector<BinInfo> all_bins;
//   all_bins.reserve(special_bins.size() + regular_bins.size());
//   
//   for (auto& bin : special_bins) {
//     all_bins.push_back(std::move(bin));
//   }
//   for (auto& bin : regular_bins) {
//     all_bins.push_back(std::move(bin));
//   }
//   
//   // Calculate metrics for all bins
//   MetricsCalculator::calculate_metrics(all_bins, smooth, total_valid_pos, total_valid_neg);
//   
//   // Create output DataFrames
//   IntegerVector bin_ids;
//   CharacterVector bin_labels;
//   IntegerVector total_counts;
//   IntegerVector neg_counts;
//   IntegerVector pos_counts;
//   NumericVector woes;
//   NumericVector ivs;
//   NumericVector kss;
//   NumericVector ginis;
//   
//   for (const auto& bin : all_bins) {
//     bin_ids.push_back(bin.id);
//     bin_labels.push_back(bin.label);
//     total_counts.push_back(bin.total_count);
//     neg_counts.push_back(bin.neg_count);
//     pos_counts.push_back(bin.pos_count);
//     woes.push_back(bin.woe);
//     ivs.push_back(bin.iv);
//     kss.push_back(bin.ks);
//     ginis.push_back(bin.gini);
//   }
//   
//   DataFrame bins_df = DataFrame::create(
//     Named("bin_id") = bin_ids,
//     Named("bin_label") = bin_labels,
//     Named("total_count") = total_counts,
//     Named("neg_count") = neg_counts,
//     Named("pos_count") = pos_counts,
//     Named("woe") = woes,
//     Named("iv") = ivs,
//     Named("ks") = kss,
//     Named("gini") = ginis
//   );
//   
//   // FAST TRANSFORM using FastBinMapper
//   FastBinMapper mapper(var_type);
//   mapper.build_from_bins(all_bins);
//   
//   CharacterVector transform_feature_values(target.size());
//   CharacterVector feature_bins(target.size());
//   NumericVector feature_woes(target.size());
//   
//   for (int i = 0; i < target.size(); ++i) {
//     std::string value_str = "";
//     std::string bin_label = "Unknown";
//     double woe_value = 0;
//     
//     bool is_missing = false;
//     
//     if (var_type == VariableType::CATEGORICAL || var_type == VariableType::BOOLEAN) {
//       value_str = original_categorical_values[i];
//       is_missing = (value_str == "NA");
//       
//       mapper.map_categorical(value_str, is_missing, bin_label, woe_value);
//     } else {
//       double numeric_value = original_numeric_values[i];
//       is_missing = std::isnan(numeric_value) || numeric_value == NA_REAL;
//       
//       if (!is_missing) {
//         value_str = std::to_string(numeric_value);
//         value_str.erase(value_str.find_last_not_of('0') + 1, std::string::npos);
//         value_str.erase(value_str.find_last_not_of('.') + 1, std::string::npos);
//       } else {
//         value_str = "NA";
//       }
//       
//       bool is_special = special_vals.count(numeric_value) > 0;
//       mapper.map_numeric(numeric_value, is_missing, is_special, bin_label, woe_value);
//     }
//     
//     transform_feature_values[i] = value_str;
//     feature_bins[i] = bin_label;
//     feature_woes[i] = woe_value;
//   }
//   
//   DataFrame transform_df = DataFrame::create(
//     Named(feature_col) = transform_feature_values,
//     Named(feature_col + "_bin") = feature_bins,
//     Named(feature_col + "_woe") = feature_woes
//   );
//   
//   double total_iv = MetricsCalculator::calculate_total_iv(all_bins);
//   double event_rate = static_cast<double>(total_valid_pos) / 
//     (total_valid_pos + total_valid_neg + EPSILON);
//   
//   if (!is_simple_variable && static_cast<int>(regular_bins.size()) < min_bins && !processed_data.empty()) {
//     message = "Warning: Could not achieve minimum " + std::to_string(min_bins) + 
//       " bins. Created " + std::to_string(regular_bins.size()) + " bins.";
//   }
//   
//   return List::create(
//     Named("bins") = bins_df,
//     Named("total_iv") = total_iv,
//     Named("variable_type") = variable_type_to_string(var_type),
//     Named("method_used") = method,
//     Named("n_bins") = all_bins.size(),
//     Named("total_samples") = processed_data.size(),
//     Named("event_rate") = event_rate,
//     Named("messages") = message,
//     Named("transform") = transform_df
//   );
// }
// 
// } // namespace cartwoe
// 
// // =============================================================================
// // EXPORTED FUNCTION
// // =============================================================================
// 
// //' CART Optimal Binning with Weight of Evidence (WoE)
// //'
// //' @description
// //' Performs optimal binning on numeric and categorical variables using Classification and Regression Trees (CART),
// //' computing Weight of Evidence (WoE) and Information Value (IV) for each bin. Implements comprehensive binning
// //' strategies including:
// //' \itemize{
// //'   \item \strong{CART Pure}: Standard Classification and Regression Trees
// //'   \item \strong{CART+PAVA}: CART with isotonic regression for monotonicity enforcement
// //'   \item \strong{CART+Merge}: CART with heuristic adjacent bin merging
// //' }
// //'
// //' @details
// //' \strong{1) CART Algorithm for Optimal Binning}
// //'
// //' The algorithm builds a binary tree by recursively partitioning the feature space to maximize impurity reduction:
// //' \deqn{\Delta(t) = I(S) - \frac{W_L}{W}I(S_L) - \frac{W_R}{W}I(S_R)}
// //' where:
// //' \itemize{
// //'   \item \eqn{I(S)} = impurity of parent node \eqn{S}
// //'   \item \eqn{I(S_L), I(S_R)} = impurities of left and right child nodes
// //'   \item \eqn{W, W_L, W_R} = total weights of parent, left child, and right child
// //' }
// //'
// //' \strong{Impurity Measures:}
// //' \itemize{
// //'   \item \strong{Gini Impurity}: \eqn{I(S) = 1 - \sum_{i=1}^{C} p_i^2}
// //'   \item \strong{Entropy}: \eqn{I(S) = -\sum_{i=1}^{C} p_i \log(p_i)}
// //' }
// //' where \eqn{p_i} is the probability of class \eqn{i} in the node.
// //'
// //' \strong{2) Weight of Evidence (WoE)}
// //'
// //' For bin \eqn{g}, WoE measures the strength of relationship with the target:
// //' \deqn{\text{WoE}_g = \ln\left(\frac{P(Y=1|X \in \text{bin}_g)}{P(Y=0|X \in \text{bin}_g)}\right) = \ln\left(\frac{n_{1g}/N_1}{n_{0g}/N_0}\right)}
// //' where:
// //' \itemize{
// //'   \item \eqn{n_{1g}} = number of positive events in bin \eqn{g}
// //'   \item \eqn{n_{0g}} = number of negative events in bin \eqn{g}
// //'   \item \eqn{N_1} = total positive events
// //'   \item \eqn{N_0} = total negative events
// //' }
// //'
// //' \strong{Laplace Smoothing}: To handle zero frequencies, applies smoothing factor \eqn{\lambda}:
// //' \deqn{\text{WoE}_g = \ln\left(\frac{(n_{1g} + \lambda)/(N_1 + G\lambda)}{(n_{0g} + \lambda)/(N_0 + G\lambda)}\right)}
// //' where \eqn{G} is the total number of bins.
// //'
// //' \strong{3) Information Value (IV)}
// //'
// //' Total IV quantifies the predictive power of the variable:
// //' \deqn{\text{IV} = \sum_{g=1}^{G} \left(\frac{n_{1g}}{N_1} - \frac{n_{0g}}{N_0}\right) \times \text{WoE}_g}
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
// //' When monotonic constraints are applied:
// //' \itemize{
// //'   \item \code{"cart+pava"}: Uses Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
// //'   \item \code{"cart+merge"}: Heuristic merging of adjacent bins violating monotonicity
// //' }
// //'
// //' \strong{PAVA Algorithm:}
// //' Solves the isotonic regression problem:
// //' \deqn{\min_{\theta_1 \leq \theta_2 \leq \cdots \leq \theta_G} \sum_{g=1}^{G} w_g(\theta_g - y_g)^2}
// //' where \eqn{y_g} are target values (event rates or WoE) and \eqn{w_g} are weights.
// //'
// //' \strong{5) Cost-Complexity Pruning}
// //'
// //' To prevent overfitting, applies cost-complexity pruning:
// //' \deqn{R_\alpha(T) = R(T) + \alpha|T|}
// //' where:
// //' \itemize{
// //'   \item \eqn{R(T)} = total impurity of tree \eqn{T}
// //'   \item \eqn{|T|} = number of terminal nodes (leaves)
// //'   \item \eqn{\alpha} = complexity parameter
// //' }
// //'
// //' \strong{6) Special Value Handling}
// //'
// //' Missing values and special codes are handled according to \code{miss_policy}:
// //' \itemize{
// //'   \item \code{"separate"}: Creates separate bin(s) for missing/special values
// //'   \item \code{"remove"}: Excludes missing values from binning
// //'   \item \code{"impute"}: Imputes with mode (categorical) or median (numeric)
// //'   \item \code{"merge"}: Merges with nearest bin based on event rate
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
// //'   Algorithm will force creation of at least this many bins when possible.
// //'   Default: 2
// //'
// //' @param max_bins Integer specifying maximum number of bins allowed. Must be greater than or equal to
// //'   min_bins. Controls model complexity and prevents overfitting.
// //'   Default: 6
// //'
// //' @param method String specifying the binning algorithm to use. Options:
// //'   \itemize{
// //'     \item \code{"cart"}: Pure CART algorithm
// //'     \item \code{"cart+pava"}: CART with PAVA monotonicity enforcement (default)
// //'     \item \code{"cart+merge"}: CART with heuristic merging for monotonicity
// //'   }
// //'   Default: "cart"
// //'
// //' @param miss_policy String specifying how to handle missing values. Options:
// //'   \itemize{
// //'     \item \code{"separate"}: Create separate bin(s) for missing values (default)
// //'     \item \code{"remove"}: Exclude missing values from analysis
// //'     \item \code{"impute"}: Impute with mode (categorical) or median (numeric)
// //'     \item \code{"merge"}: Merge with nearest bin based on event rate
// //'   }
// //'   Default: "separate"
// //'
// //' @param cat_sep String separator used when combining multiple categories into a single bin label.
// //'   Should be a pattern unlikely to appear in actual category names.
// //'   Default: "\%;\%"
// //'
// //' @param digits Integer number of decimal places for numeric bin boundaries in labels.
// //'   Must be between 0 and 10. Affects display only, not calculation precision.
// //'   Default: 4
// //'
// //' @param smooth Numeric Laplace smoothing factor for WoE calculation. Must be non-negative.
// //'   Prevents undefined WoE values when bins have zero counts. Higher values provide more smoothing.
// //'   Default: 0.5
// //'
// //' @param criterion String specifying the impurity criterion for CART splits. Options:
// //'   \itemize{
// //'     \item \code{"gini"}: Gini impurity (default)
// //'     \item \code{"entropy"}: Information gain (entropy-based)
// //'   }
// //'   Default: "gini"
// //'
// //' @param min_size Numeric minimum bin size specification. Can be:
// //'   \itemize{
// //'     \item Proportion (0,1]: Minimum proportion of total observations per bin
// //'     \item Count (>1): Minimum absolute count per bin
// //'   }
// //'   Prevents creation of very small bins that may be unstable.
// //'   Default: 0.05
// //'
// //' @param use_pruning Logical indicating whether to apply cost-complexity pruning to the CART tree.
// //'   Helps prevent overfitting by removing branches that provide little predictive power.
// //'   Default: TRUE
// //'
// //' @param cv_folds Integer number of cross-validation folds for pruning parameter selection.
// //'   Must be at least 2 when pruning is enabled. Higher values provide more robust pruning.
// //'   Default: 5
// //'
// //' @param monotonic_trend String specifying the desired monotonic relationship. Options:
// //'   \itemize{
// //'     \item \code{"auto"}: Automatically detect trend based on Spearman correlation (default)
// //'     \item \code{"increasing"}: Enforce increasing WoE trend
// //'     \item \code{"decreasing"}: Enforce decreasing WoE trend
// //'     \item \code{"none"}: No monotonic constraint
// //'   }
// //'   Default: "auto"
// //'
// //' @param monotonic_mode String specifying the method for enforcing monotonicity. Options:
// //'   \itemize{
// //'     \item \code{"pava"}: Isotonic regression using PAVA algorithm (default)
// //'     \item \code{"merge"}: Heuristic adjacent bin merging
// //'   }
// //'   Default: "pava"
// //'
// //' @param parallel Logical indicating whether to use parallel processing via OpenMP for multiple features.
// //'   Requires OpenMP support in compilation. Speeds up processing of many features.
// //'   Default: FALSE
// //'
// //' @param weights Optional NumericVector of observation weights for weighted binning. Length must equal
// //'   number of rows in data. All weights must be non-negative. NULL indicates equal weights.
// //'   Default: NULL
// //'
// //' @param special_vals NumericVector of special values to handle separately (e.g., -999, -888 for
// //'   special codes). These values are isolated in separate bins when encountered.
// //'   Default: empty vector (no special values)
// //'
// //' @param max_cat Integer maximum number of categories before automatic grouping for categorical variables.
// //'   Must be at least 2. Categories beyond this limit are grouped based on event rate similarity.
// //'   Default: 50
// //'
// //' @param rare_pct Numeric threshold for rare category grouping as proportion of total.
// //'   Categories with frequency below this threshold are combined. Must be between 0 and 0.5.
// //'   Default: 0.01 (1\%)
// //'
// //' @return
// //' Named list with one element per processed feature, each containing:
// //' \itemize{
// //'   \item \code{bins}: data.frame with columns:
// //'     \itemize{
// //'       \item \code{bin_id}: Sequential bin identifier (1, 2, 3, ...)
// //'       \item \code{bin_label}: Formatted bin range/categories (e.g., "(-inf, 25.5]", "[25.5, 40.0]", "(40.0, +inf)")
// //'       \item \code{total_count}: Total observations in bin
// //'       \item \code{neg_count}: Count of negative class (Y=0)
// //'       \item \code{pos_count}: Count of positive class (Y=1)
// //'       \item \code{woe}: Weight of Evidence value for the bin
// //'       \item \code{iv}: Information Value contribution of the bin
// //'       \item \code{ks}: Kolmogorov-Smirnov statistic up to this bin
// //'       \item \code{gini}: Gini coefficient for the bin
// //'     }
// //'   \item \code{total_iv}: Total Information Value for the feature
// //'   \item \code{variable_type}: Detected type ("numeric_continuous", "numeric_discrete", "categorical", "boolean")
// //'   \item \code{method_used}: Algorithm actually used for binning
// //'   \item \code{n_bins}: Final number of bins created (including special bins if separate)
// //'   \item \code{total_samples}: Total valid observations processed
// //'   \item \code{event_rate}: Overall positive class rate in processed data
// //'   \item \code{messages}: Processing messages or warnings
// //'   \item \code{transform}: data.frame with original values and transformations:
// //'     \itemize{
// //'       \item \code{[feature]}: Original feature values
// //'       \item \code{[feature]_bin}: Assigned bin labels
// //'       \item \code{[feature]_woe}: Assigned WoE values
// //'     }
// //' }
// //'
// //' @section Algorithm Pipeline:
// //' The complete binning process follows these steps:
// //' \enumerate{
// //'   \item \strong{Data Preparation}:
// //'     \itemize{
// //'       \item Detect variable type (numeric, categorical, boolean, date)
// //'       \item Handle missing values according to \code{miss_policy}
// //'       \item Process special values into separate bins
// //'       \item For categorical variables: order by event rate and map to ordinal scale
// //'     }
// //'   \item \strong{CART Tree Construction}:
// //'     \itemize{
// //'       \item Sort data by feature value (numeric) or ordinal mapping (categorical)
// //'       \item Build binary tree using best-first search with impurity gain criterion
// //'       \item Respect \code{min_size} constraints during splitting
// //'       \item Stop when reaching \code{max_bins} leaves or no beneficial splits
// //'     }
// //'   \item \strong{Tree Pruning} (if enabled):
// //'     \itemize{
// //'       \item Generate sequence of subtrees via cost-complexity pruning
// //'       \item Select optimal subtree using cross-validation
// //'       \item Ensure result respects \code{min_bins} and \code{max_bins} constraints
// //'     }
// //'   \item \strong{Bin Adjustment}:
// //'     \itemize{
// //'       \item Force minimum bin count if below \code{min_bins}
// //'       \item Merge bins if above \code{max_bins}
// //'       \item Apply monotonicity constraints if specified
// //'     }
// //'   \item \strong{Metric Calculation}:
// //'     \itemize{
// //'       \item Compute WoE and IV with Laplace smoothing
// //'       \item Calculate KS statistics and Gini coefficients
// //'       \item Generate transformation mappings
// //'     }
// //' }
// //'
// //' @section Special Cases Handling:
// //' \itemize{
// //'   \item \strong{Variables with \eqn{\le 2} unique classes}: Direct binning without optimization
// //'   \item \strong{Date/Time variables}: Automatically skipped with warning message
// //'   \item \strong{Unsupported types}: Skipped with appropriate message
// //'   \item \strong{Constant variables}: Single bin created with warning
// //'   \item \strong{Insufficient data}: Processing skipped with informative message
// //' }
// //'
// //' @note
// //' \itemize{
// //'   \item WoE values are clipped to [-20, 20] for numerical stability
// //'   \item Variables with 2 or fewer unique classes are processed directly without CART optimization
// //'   \item Factor variables are processed as categorical with proper level handling
// //'   \item Integer variables with \eqn{\leq 20} unique values are treated as discrete numeric
// //'   \item Special values (-999, -888, etc.) can be isolated in separate bins
// //'   \item Uses "(-inf,x)" and "[x,+inf)" notation for unbounded intervals
// //'   \item Categorical bins combine multiple categories using \code{cat_sep} separator
// //' }
// //'
// //' @references
// //' \itemize{
// //'   \item Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //'   \item Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. Journal of the Royal Statistical Society: Series A (Statistics in Society), 160(3), 523-541.
// //'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
// //'   \item Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). Order restricted statistical inference. Wiley.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Load credit data
// //' data <- scorecard::germancredit
// //' data$target <- ifelse(data$creditability == "good", 1, 0)
// //'
// //' # Basic binning with all defaults
// //' result <- cart_woe(
// //'   data = data,
// //'   target_col = "target",
// //'   feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
// //' )
// //'
// //' # Advanced binning with custom parameters
// //' result_custom <- cart_woe(
// //'   data = data,
// //'   target_col = "target",
// //'   feature_cols = names(data)[1:10],
// //'   min_bins = 3,              # Minimum 3 bins
// //'   max_bins = 8,              # Maximum 8 bins
// //'   method = "cart+pava",      # CART with PAVA monotonicity
// //'   miss_policy = "separate",  # Missing values in separate bin
// //'   cat_sep = " | ",           # Use pipe separator for categories
// //'   digits = 2,                # 2 decimal places in labels
// //'   smooth = 1.0,              # Higher smoothing for WoE
// //'   criterion = "entropy",     # Use entropy instead of Gini
// //'   min_size = 0.03,           # Each bin at least 3% of data
// //'   use_pruning = TRUE,        # Enable cost-complexity pruning
// //'   cv_folds = 10,             # 10-fold CV for pruning
// //'   monotonic_trend = "auto",  # Auto-detect monotonic trend
// //'   monotonic_mode = "pava",   # Use PAVA for monotonicity
// //'   parallel = TRUE,           # Parallel processing for speed
// //'   special_vals = c(-999, -888), # Special codes to isolate
// //'   max_cat = 25,              # Group if >25 categories
// //'   rare_pct = 0.02            # Group categories <2% frequency
// //' )
// //'
// //' # Extract results for a specific feature
// //' duration_bins <- result$duration.in.month$bins
// //' duration_iv <- result$duration.in.month$total_iv
// //' duration_transform <- result$duration.in.month$transform
// //'
// //' # View binning results
// //' print(duration_bins)
// //' print(paste("Total IV:", duration_iv))
// //'
// //' # Check transformation
// //' head(duration_transform)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List cart_woe(
//     DataFrame data,
//     std::string target_col,
//     CharacterVector feature_cols,
//     int min_bins = 2,
//     int max_bins = 5,
//     std::string method = "cart",
//     std::string miss_policy = "separate",
//     std::string cat_sep = "%;%",
//     int digits = 4,
//     double smooth = 0.0,
//     std::string criterion = "gini",
//     double min_size = 0.05,
//     bool use_pruning = true,
//     int cv_folds = 5,
//     std::string monotonic_trend = "auto",
//     std::string monotonic_mode = "pava",
//     bool parallel = false,
//     Nullable<NumericVector> weights = R_NilValue,
//     NumericVector special_vals = NumericVector::create(),
//     int max_cat = 50,
//     double rare_pct = 0.01) {
//   
//   // Input validation
//   if (min_bins < 2) {
//     stop("min_bins must be >= 2");
//   }
//   if (max_bins < min_bins) {
//     stop("max_bins must be >= min_bins");
//   }
//   if (min_size <= 0) {
//     stop("min_size must be > 0");
//   }
//   if (smooth < 0) {
//     stop("smooth must be >= 0");
//   }
//   if (criterion != "gini" && criterion != "entropy") {
//     stop("criterion must be 'gini' or 'entropy'");
//   }
//   if (use_pruning && cv_folds < 2) {
//     stop("cv_folds must be >= 2 when use_pruning is TRUE");
//   }
//   
//   cartwoe::Criterion crit = (criterion == "gini") ? 
//   cartwoe::Criterion::GINI : cartwoe::Criterion::ENTROPY;
//   
//   cartwoe::MonotonicTrend mono_trend = cartwoe::MonotonicTrend::AUTO;
//   if (monotonic_trend == "increasing") {
//     mono_trend = cartwoe::MonotonicTrend::INCREASING;
//   } else if (monotonic_trend == "decreasing") {
//     mono_trend = cartwoe::MonotonicTrend::DECREASING;
//   } else if (monotonic_trend == "none") {
//     mono_trend = cartwoe::MonotonicTrend::NONE;
//   }
//   
//   int n = data.nrows();
//   NumericVector w(n, 1.0);
//   if (!weights.isNull()) {
//     w = as<NumericVector>(weights);
//     if (w.size() != n) {
//       stop("weights must have same length as data");
//     }
//   }
//   
//   std::set<double> special_set;
//   for (int i = 0; i < special_vals.size(); ++i) {
//     special_set.insert(special_vals[i]);
//   }
//   
//   List results;
//   
// #ifdef _OPENMP
//   if (parallel) {
//     int max_threads = omp_get_max_threads();
//     omp_set_num_threads(max_threads);
//   }
// #endif
//   
//   for (int i = 0; i < feature_cols.size(); ++i) {
//     std::string feature_name = as<std::string>(feature_cols[i]);
//     
//     List feature_result = cartwoe::process_feature(
//       data, target_col, feature_name, min_bins, max_bins,
//       method, miss_policy, cat_sep, digits, smooth,
//       crit, min_size, use_pruning, cv_folds,
//       mono_trend, monotonic_mode, w, special_set,
//       max_cat, rare_pct, parallel
//     );
//     
//     results[feature_name] = feature_result;
//   }
//   
//   return results;
// }
