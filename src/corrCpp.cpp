// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>

namespace CorrCppInternal {

// Enumeration for correlation methods
enum class CorrMethod {
  PEARSON,
  SPEARMAN,
  KENDALL,
  ALPHA
};

// Enumeration for missing data handling
enum class MissingMethod {
  PAIRWISE,
  LISTWISE
};

// Convert string to CorrMethod
CorrMethod parseMethod(const std::string& method_str) {
  if (method_str == "pearson") return CorrMethod::PEARSON;
  if (method_str == "spearman") return CorrMethod::SPEARMAN;
  if (method_str == "kendall") return CorrMethod::KENDALL;
  if (method_str == "alpha") return CorrMethod::ALPHA;
  throw std::invalid_argument("Invalid correlation method: " + method_str);
}

// Convert string to MissingMethod
MissingMethod parseMissing(const std::string& use_str) {
  if (use_str == "pairwise" || use_str == "pairwise.complete.obs") return MissingMethod::PAIRWISE;
  if (use_str == "listwise" || use_str == "complete.obs") return MissingMethod::LISTWISE;
  throw std::invalid_argument("Invalid missing data method: " + use_str);
}

// Calculate p-value for Pearson correlation using t-distribution
double calculatePearsonPValue(double r, int n) {
  if (n <= 2 || std::abs(r) >= 1.0) {
    return 1.0;
  }
  
  // t-statistic: t = r * sqrt((n-2)/(1-r^2))
  double t_stat = r * std::sqrt(static_cast<double>(n - 2) / (1.0 - r * r));
  
  // Two-tailed p-value using R's pt function
  double p_val = 2.0 * R::pt(std::abs(t_stat), n - 2, false, false);
  
  return std::min(p_val, 1.0);
}

// Calculate p-value for Spearman correlation (approximate using normal distribution for large n)
double calculateSpearmanPValue(double rho, int n) {
  if (n <= 3) {
    return 1.0;
  }
  
  // For large n, Spearman's rho approximately follows normal distribution
  // t = rho * sqrt((n-2)/(1-rho^2))
  if (n >= 30) {
    return calculatePearsonPValue(rho, n); // Same formula for large samples
  }
  
  // For smaller samples, use t-distribution approximation
  double t_stat = rho * std::sqrt(static_cast<double>(n - 2) / (1.0 - rho * rho));
  double p_val = 2.0 * R::pt(std::abs(t_stat), n - 2, false, false);
  
  return std::min(p_val, 1.0);
}

// Rank vector with midrank for ties (average rank for tied values)
arma::vec rankVector(const arma::vec& x) {
  arma::uword n = x.n_elem;
  if (n == 0) return arma::vec();
  
  // Create pairs of (value, original_index)
  std::vector<std::pair<double, arma::uword>> value_index_pairs;
  value_index_pairs.reserve(n);
  
  for (arma::uword i = 0; i < n; ++i) {
    value_index_pairs.emplace_back(x[i], i);
  }
  
  // Sort by value
  std::sort(value_index_pairs.begin(), value_index_pairs.end());
  
  arma::vec ranks(n);
  
  // Assign ranks with midrank for ties
  arma::uword i = 0;
  while (i < n) {
    arma::uword j = i;
    
    // Find the end of the current tie group
    while (j < n - 1 && value_index_pairs[j].first == value_index_pairs[j + 1].first) {
      j++;
    }
    
    // Calculate midrank for the tie group
    double midrank = static_cast<double>(i + j + 2) / 2.0; // +1 for 1-based ranking, +1 for inclusive
    
    // Assign midrank to all tied values
    for (arma::uword k = i; k <= j; ++k) {
      ranks[value_index_pairs[k].second] = midrank;
    }
    
    i = j + 1;
  }
  
  return ranks;
}

// Calculate Kendall's tau-b coefficient
double calculateKendallTauB(const arma::vec& x, const arma::vec& y) {
  arma::uword n = x.n_elem;
  if (n != y.n_elem || n < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  
  int concordant = 0;
  int discordant = 0;
  int ties_x = 0;
  int ties_y = 0;
  int ties_both = 0;
  
  // Count concordant, discordant pairs, and ties
  for (arma::uword i = 0; i < n - 1; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      
      if (dx > 0 && dy > 0) {
        concordant++;
      } else if (dx < 0 && dy < 0) {
        concordant++;
      } else if (dx > 0 && dy < 0) {
        discordant++;
      } else if (dx < 0 && dy > 0) {
        discordant++;
      } else if (dx == 0 && dy == 0) {
        ties_both++;
      } else if (dx == 0) {
        ties_x++;
      } else if (dy == 0) {
        ties_y++;
      }
    }
  }
  
  // Total pairs
  int total_pairs = n * (n - 1) / 2;
  
  // Kendall's tau-b formula with tie correction
  double denominator = std::sqrt(static_cast<double>(total_pairs - ties_x) * (total_pairs - ties_y));
  
  if (denominator == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  
  return static_cast<double>(concordant - discordant) / denominator;
}

// Calculate p-value for Kendall's tau using normal approximation
double calculateKendallPValue(double tau, int n) {
  if (n <= 3) {
    return 1.0;
  }
  
  // Standard error of tau for large samples
  double se = std::sqrt((2.0 * (2 * n + 5)) / (9.0 * n * (n - 1)));
  
  // Z-score
  double z = tau / se;
  
  // Two-tailed p-value using normal approximation
  double p_val = 2.0 * (1.0 - R::pnorm(std::abs(z), 0.0, 1.0, true, false));
  
  return std::min(p_val, 1.0);
}

// Calculate Cronbach's alpha coefficient
double calculateCronbachAlpha(const arma::mat& X) {
  arma::uword n_items = X.n_cols;
  if (n_items < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  
  // Calculate item variances
  arma::vec item_variances = arma::var(X, 0, 0); // ddof=0 for population variance
  double sum_item_variances = arma::sum(item_variances);
  
  // Calculate total score variance
  arma::vec total_scores = arma::sum(X, 1);
  double total_variance = arma::var(total_scores, 0); // ddof=0 for population variance
  
  // Cronbach's alpha formula
  if (total_variance <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  
  double alpha = (static_cast<double>(n_items) / (n_items - 1.0)) * 
    (1.0 - sum_item_variances / total_variance);
  
  return alpha;
}

// Main correlation processor class
class CorrelationProcessor {
private:
  CorrMethod method;
  MissingMethod missing_method;
  
  // Remove rows with any missing values (listwise deletion)
  arma::mat removeIncompleteRows(const arma::mat& X) const {
    std::vector<arma::uword> complete_rows;
    
    for (arma::uword i = 0; i < X.n_rows; ++i) {
      bool row_complete = true;
      for (arma::uword j = 0; j < X.n_cols; ++j) {
        if (!arma::is_finite(X(i, j))) {
          row_complete = false;
          break;
        }
      }
      if (row_complete) {
        complete_rows.push_back(i);
      }
    }
    
    if (complete_rows.empty()) {
      return arma::mat(); // Return empty matrix
    }
    
    // Extract complete rows
    arma::mat X_complete(complete_rows.size(), X.n_cols);
    for (size_t i = 0; i < complete_rows.size(); ++i) {
      X_complete.row(i) = X.row(complete_rows[i]);
    }
    
    return X_complete;
  }
  
  // Extract valid pairs for pairwise analysis
  std::pair<arma::vec, arma::vec> extractValidPairs(const arma::vec& x, const arma::vec& y) const {
    std::vector<double> valid_x, valid_y;
    
    for (arma::uword i = 0; i < x.n_elem && i < y.n_elem; ++i) {
      if (arma::is_finite(x[i]) && arma::is_finite(y[i])) {
        valid_x.push_back(x[i]);
        valid_y.push_back(y[i]);
      }
    }
    
    arma::vec vec_x(valid_x);
    arma::vec vec_y(valid_y);
    
    return std::make_pair(vec_x, vec_y);
  }
  
  // Calculate Pearson correlation between two vectors
  double calculatePearsonCorr(const arma::vec& x, const arma::vec& y) const {
    if (x.n_elem != y.n_elem || x.n_elem < 2) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Use Armadillo's built-in correlation function for numerical stability
    arma::mat corr_mat = arma::cor(arma::join_rows(x, y));
    
    if (corr_mat.n_rows >= 2 && corr_mat.n_cols >= 2) {
      return corr_mat(0, 1);
    }
    
    return std::numeric_limits<double>::quiet_NaN();
  }
  
public:
  CorrelationProcessor(CorrMethod m, MissingMethod mm) : method(m), missing_method(mm) {}
  
  // Main processing function
  Rcpp::List process(const arma::mat& X) const {
    arma::uword n_vars = X.n_cols;
    
    // Initialize result matrices
    arma::mat R(n_vars, n_vars, arma::fill::eye);  // Correlation matrix
    arma::mat N(n_vars, n_vars, arma::fill::zeros); // Sample size matrix
    arma::mat P(n_vars, n_vars, arma::fill::ones);  // P-value matrix
    
    // Handle Cronbach's alpha as special case
    if (method == CorrMethod::ALPHA) {
      arma::mat X_complete = (missing_method == MissingMethod::LISTWISE) ? 
      removeIncompleteRows(X) : X;
      
      double alpha = calculateCronbachAlpha(X_complete);
      
      // Return alpha as a 1x1 matrix
      arma::mat alpha_mat(1, 1);
      alpha_mat(0, 0) = alpha;
      
      arma::mat n_mat(1, 1);
      n_mat(0, 0) = X_complete.n_rows;
      
      arma::mat p_mat(1, 1);
      p_mat(0, 0) = std::numeric_limits<double>::quiet_NaN();
      
      return Rcpp::List::create(
        Rcpp::Named("r") = alpha_mat,
        Rcpp::Named("n") = n_mat,
        Rcpp::Named("p") = p_mat
      );
    }
    
    // Handle listwise deletion
    arma::mat X_work = X;
    if (missing_method == MissingMethod::LISTWISE) {
      X_work = removeIncompleteRows(X);
      if (X_work.n_rows == 0) {
        // All rows had missing values
        R.fill(std::numeric_limits<double>::quiet_NaN());
        N.fill(0);
        P.fill(std::numeric_limits<double>::quiet_NaN());
        
        return Rcpp::List::create(
          Rcpp::Named("r") = R,
          Rcpp::Named("n") = N,
          Rcpp::Named("p") = P
        );
      }
    }
    
    // Calculate correlations for each pair of variables
    for (arma::uword i = 0; i < n_vars; ++i) {
      for (arma::uword j = i; j < n_vars; ++j) {
        if (i == j) {
          // Diagonal elements
          if (missing_method == MissingMethod::LISTWISE) {
            N(i, j) = X_work.n_rows;
          } else {
            arma::uvec finite_mask = arma::find_finite(X.col(i));
            N(i, j) = finite_mask.n_elem;
          }
          P(i, j) = 0.0;
          continue;
        }
        
        arma::vec x_vec, y_vec;
        
        if (missing_method == MissingMethod::LISTWISE) {
          x_vec = X_work.col(i);
          y_vec = X_work.col(j);
        } else {
          // Pairwise deletion
          auto valid_pair = extractValidPairs(X.col(i), X.col(j));
          x_vec = valid_pair.first;
          y_vec = valid_pair.second;
        }
        
        arma::uword n_pair = x_vec.n_elem;
        N(i, j) = N(j, i) = n_pair;
        
        if (n_pair < 2) {
          R(i, j) = R(j, i) = std::numeric_limits<double>::quiet_NaN();
          P(i, j) = P(j, i) = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        
        double corr_val = std::numeric_limits<double>::quiet_NaN();
        double p_val = 1.0;
        
        // Calculate correlation based on method
        switch (method) {
        case CorrMethod::PEARSON: {
          corr_val = calculatePearsonCorr(x_vec, y_vec);
          if (std::isfinite(corr_val)) {
            p_val = calculatePearsonPValue(corr_val, n_pair);
          }
          break;
        }
          
        case CorrMethod::SPEARMAN: {
          arma::vec x_ranks = rankVector(x_vec);
          arma::vec y_ranks = rankVector(y_vec);
          corr_val = calculatePearsonCorr(x_ranks, y_ranks);
          if (std::isfinite(corr_val)) {
            p_val = calculateSpearmanPValue(corr_val, n_pair);
          }
          break;
        }
          
        case CorrMethod::KENDALL: {
          corr_val = calculateKendallTauB(x_vec, y_vec);
          if (std::isfinite(corr_val)) {
            p_val = calculateKendallPValue(corr_val, n_pair);
          }
          break;
        }
          
        default:
          break;
        }
        
        R(i, j) = R(j, i) = corr_val;
        P(i, j) = P(j, i) = p_val;
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("r") = R,
      Rcpp::Named("n") = N,
      Rcpp::Named("p") = P
    );
  }
};

} // namespace CorrCppInternal


//' High-Performance Correlation Analysis (SAS PROC CORR equivalent)
//'
//' This function provides comprehensive correlation analysis with multiple association 
//' measures, hypothesis testing, and flexible missing data handling. It replicates
//' and extends the functionality of SAS PROC CORR using optimized C++ algorithms.
//'
//' @param x A numeric matrix or data frame containing the variables for correlation analysis
//' @param method Character string specifying the correlation method:
//'   \itemize{
//'     \item "pearson" - Pearson product-moment correlation (default)
//'     \item "spearman" - Spearman rank correlation (non-parametric)
//'     \item "kendall" - Kendall's tau-b (robust to outliers)
//'     \item "alpha" - Cronbach's alpha for reliability analysis
//'   }
//' @param use Character string specifying missing data handling:
//'   \itemize{
//'     \item "pairwise" - Pairwise deletion (default, matches SAS PROC CORR default)
//'     \item "listwise" - Listwise deletion (complete cases only, matches SAS NOMISS option)
//'   }
//' @param min_periods Integer minimum number of valid observations required for a correlation
//'   (default: 2). Correlations with fewer observations return NA.
//'
//' @return A list with three named components (following Hmisc::rcorr structure):
//'   \itemize{
//'     \item r - Matrix of correlation coefficients
//'     \item n - Matrix of sample sizes used for each pairwise correlation
//'     \item p - Matrix of p-values for significance tests
//'   }
//'
//' @details
//' The function implements numerically stable algorithms optimized for performance:
//' 
//' \strong{Pearson Correlation:} Uses BLAS-optimized routines via RcppArmadillo for
//' numerical stability. Significance testing via t-distribution with (n-2) degrees of freedom.
//' 
//' \strong{Spearman Correlation:} Implements midrank averaging for tied values, then
//' applies Pearson correlation to ranks. For large samples (n >= 30), uses normal
//' approximation for p-values.
//' 
//' \strong{Kendall's tau-b:} Counts concordant/discordant pairs with tie corrections.
//' Uses normal approximation for significance testing in large samples.
//' 
//' \strong{Cronbach's Alpha:} Measures internal consistency reliability using the
//' variance-based formula. Returns single value in 1×1 matrix format.
//' 
//' \strong{Missing Data Handling:}
//' \itemize{
//'   \item Pairwise: Maximizes available data but may produce non-positive-semidefinite matrices
//'   \item Listwise: Ensures mathematical validity for downstream multivariate procedures
//' }
//'
//' @examples
//' \dontrun{
//' # Basic Pearson correlation
//' result <- corrCpp(mtcars[, c("mpg", "hp", "wt")])
//' print(result$r)
//' print(result$p)
//' 
//' # Spearman correlation with listwise deletion
//' result <- corrCpp(mtcars, method = "spearman", use = "listwise")
//' 
//' # Kendall's tau for robust correlation
//' result <- corrCpp(mtcars, method = "kendall")
//' 
//' # Cronbach's alpha for reliability
//' # (typically used with survey/scale items)
//' alpha_result <- corrCpp(scale_items, method = "alpha")
//' cat("Cronbach's Alpha:", alpha_result$r[1,1], "\n")
//' }
//'
//' @references
//' Pearson, K. (1895). Mathematical contributions to the theory of evolution.
//' Philosophical Transactions of the Royal Society A, 186, 343-414.
//' 
//' Spearman, C. (1904). The proof and measurement of association between two things.
//' American Journal of Psychology, 15(1), 72-101.
//' 
//' Kendall, M.G. (1938). A new measure of rank correlation. Biometrika, 30(1-2), 81-93.
//' 
//' Cronbach, L.J. (1951). Coefficient alpha and the internal structure of tests.
//' Psychometrika, 16(3), 297-334.
//'
//' @note This implementation prioritizes computational efficiency while maintaining
//' statistical accuracy. For very large matrices, consider using specialized
//' big data correlation libraries.
//' 
//' @author Statistical Computing Implementation
//' @export
// [[Rcpp::export]]
Rcpp::List corrCpp(
   SEXP x,
   std::string method = "pearson",
   std::string use = "pairwise",
   int min_periods = 2) {
 
 try {
   // Convert input to Armadillo matrix
   arma::mat X;
   
   if (Rf_isMatrix(x)) {
     Rcpp::NumericMatrix r_mat(x);
     X = arma::mat(r_mat.begin(), r_mat.nrow(), r_mat.ncol(), false);
   } else if (Rf_isNewList(x)) {
     // Handle data.frame input
     Rcpp::DataFrame df(x);
     
     // Extract numeric columns only
     std::vector<std::string> numeric_cols;
     Rcpp::CharacterVector col_names = df.names();
     
     for (int i = 0; i < df.size(); ++i) {
       SEXP col = df[i];
       if (TYPEOF(col) == REALSXP || TYPEOF(col) == INTSXP) {
         numeric_cols.push_back(Rcpp::as<std::string>(col_names[i]));
       }
     }
     
     if (numeric_cols.empty()) {
       Rcpp::stop("No numeric columns found in data.frame");
     }
     
     // Create numeric matrix from selected columns
     X.set_size(df.nrows(), numeric_cols.size());
     for (size_t j = 0; j < numeric_cols.size(); ++j) {
       SEXP col = df[numeric_cols[j]];
       if (TYPEOF(col) == REALSXP) {
         Rcpp::NumericVector num_col(col);
         for (int i = 0; i < df.nrows(); ++i) {
           X(i, j) = num_col[i];
         }
       } else if (TYPEOF(col) == INTSXP) {
         Rcpp::IntegerVector int_col(col);
         for (int i = 0; i < df.nrows(); ++i) {
           X(i, j) = Rcpp::IntegerVector::is_na(int_col[i]) ? 
           std::numeric_limits<double>::quiet_NaN() : 
           static_cast<double>(int_col[i]);
         }
       }
     }
   } else {
     Rcpp::stop("Input must be a numeric matrix or data.frame");
   }
   
   if (X.n_rows == 0 || X.n_cols == 0) {
     Rcpp::stop("Input matrix is empty");
   }
   
   if (X.n_cols == 1 && method != "alpha") {
     Rcpp::stop("Need at least 2 variables for correlation analysis (except for alpha)");
   }
   
   // Parse method and missing data handling
   CorrCppInternal::CorrMethod corr_method = CorrCppInternal::parseMethod(method);
   CorrCppInternal::MissingMethod missing_method = CorrCppInternal::parseMissing(use);
   
   // Create processor and compute correlations
   CorrCppInternal::CorrelationProcessor processor(corr_method, missing_method);
   Rcpp::List result = processor.process(X);
   
   // Apply min_periods filter
   if (min_periods > 2) {
     arma::mat n_mat = Rcpp::as<arma::mat>(result["n"]);
     arma::mat r_mat = Rcpp::as<arma::mat>(result["r"]);
     arma::mat p_mat = Rcpp::as<arma::mat>(result["p"]);
     
     // Set correlations to NA where sample size is below threshold
     for (arma::uword i = 0; i < n_mat.n_rows; ++i) {
       for (arma::uword j = 0; j < n_mat.n_cols; ++j) {
         if (n_mat(i, j) < min_periods) {
           r_mat(i, j) = std::numeric_limits<double>::quiet_NaN();
           p_mat(i, j) = std::numeric_limits<double>::quiet_NaN();
         }
       }
     }
     
     result["r"] = r_mat;
     result["p"] = p_mat;
   }
   
   return result;
   
 } catch (const std::exception& e) {
   Rcpp::stop("corrCpp error: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("corrCpp: Unknown error occurred during processing");
 }
}
