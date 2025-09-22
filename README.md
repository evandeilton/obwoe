# obwoe: Comprehensive Optimal Binning and WoE Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R](https://img.shields.io/badge/R-%3E%3D%203.5.0-blue)](https://www.r-project.org/)
[![CRAN](https://www.r-pkg.org/badges/version/obwoe)](https://cran.r-project.org/package=obwoe)

## Overview

The **obwoe** package provides a comprehensive toolkit for optimal binning and Weight of Evidence (WoE) analysis in R. It implements state-of-the-art algorithms for transforming continuous and categorical variables into discrete bins that maximize predictive power for binary classification problems.

This package is particularly useful for credit scoring, risk modeling, and feature engineering applications where binning variables and calculating WoE/IV metrics are essential steps in the modeling process.

## Features

### 1. Multiple Optimal Binning Algorithms

The package offers three distinct approaches to optimal binning:

1. **ChiMerge-based Binning** (`chimerge_woe`)
   - Original ChiMerge algorithm
   - Modified ChiMerge with Yates correction
   - Optimized ChiMerge with caching
   - D2 entropy-based algorithm
   - Hybrid approach combining multiple methods

2. **Divergence Measures and Information Value Binning** (`dmiv_woe`)
   - Multi-divergence optimization (L2, KL, Hellinger, etc.)
   - Regularized Information Value (L1/L2 regularization)
   - Adaptive monotonicity enforcement
   - Fast transform mapping for O(log n) bin assignment

3. **CART-based Binning** (`cart_woe`)
   - Standard Classification and Regression Trees
   - CART with isotonic regression for monotonicity
   - CART with heuristic adjacent bin merging
   - Cost-complexity pruning to prevent overfitting

### 2. Advanced WoE and IV Analysis

- **Weight of Evidence (WoE)** calculation with configurable smoothing
- **Information Value (IV)** computation with interpretation guidelines
- **Laplace smoothing** to handle zero frequencies
- **Monotonicity enforcement** using isotonic regression
- **Special value handling** for missing values and custom codes

### 3. Comprehensive Gains Table Analysis

- **woe_gains**: Detailed gains table metrics with over 20 statistical measures
- **woe_gains_compare**: Feature comparison and ranking based on performance metrics
- Metrics include KS statistic, AUC, Gini coefficient, PSI, and more

### 4. Performance and Usability Features

- **High-performance C++ implementation** with R integration
- **Parallel processing** support via OpenMP
- **Extensive parameter customization** for fine-tuning
- **Comprehensive error handling** and informative messages
- **Memory-efficient** algorithms for large datasets

## Installation

You can install the development version of obwoe from GitHub:

```r
# Install devtools if not already installed
if (!require(devtools)) {
  install.packages("devtools")
}

# Install obwoe from GitHub
devtools::install_github("evandeilton/obwoe")
```

## Quick Start

Here's a basic example of how to use the package:

```r
library(obwoe)

# Load sample data (using scorecard package data)
library(scorecard)
data(germancredit)
data$target <- ifelse(germancredit$creditability == "good", 1, 0)

# Perform ChiMerge-based binning
result <- chimerge_woe(
  data = data,
  target_col = "target",
  feature_cols = c("duration.in.month", "credit.amount", "age.in.years")
)

# View binning results for a specific feature
print(result$duration.in.month$bins)

# Calculate gains table metrics
gains_result <- woe_gains(result)

# Compare features by KS statistic
comparison <- woe_gains_compare(gains_result)
head(comparison)
```

## Detailed Functionality

### ChiMerge-based Binning (`chimerge_woe`)

Implements multiple ChiMerge algorithm variations for optimal binning:

- Bottom-up merging based on chi-square independence test
- Handles small expected frequencies with Yates' continuity correction
- Includes performance optimizations and caching
- Supports various binning strategies (original, modified, optimized, D2, hybrid)

### Divergence Measures and Information Value Binning (`dmiv_woe`)

Advanced binning using divergence measures:

- Supports multiple divergence measures (L2, KL, Hellinger, Chi-square, etc.)
- Regularized Information Value with L1/L2 penalties
- Isotonic regression with automatic trend detection
- Wilson score confidence intervals for WoE values
- Fast bin mapping for efficient transformations

### CART-based Binning (`cart_woe`)

Classification and Regression Trees approach:

- Standard CART algorithm for binning
- PAVA monotonicity enforcement
- Heuristic merging for monotonicity
- Cost-complexity pruning to prevent overfitting
- Support for both Gini impurity and entropy criteria

### Gains Table Analysis (`woe_gains`)

Comprehensive evaluation metrics:

- Over 20 statistical measures for binning quality
- KS statistic, AUC, Gini coefficient
- Lift metrics, entropy measures
- Classification performance metrics
- Statistical tests for bin significance

### Feature Comparison (`woe_gains_compare`)

Feature ranking and comparison:

- Rank features by various performance metrics
- Support for KS, AUC, Gini, IV, and PSI
- Customizable ranking criteria
- Useful for feature selection and model validation

## Key Parameters and Options

### Common Binning Parameters

- `min_bins` / `max_bins`: Control the number of bins
- `smooth`: Laplace smoothing factor for WoE calculation
- `miss_policy`: Handling of missing values (separate, remove, impute, merge)
- `special_vals`: Custom special values to isolate
- `parallel`: Enable parallel processing for multiple features

### Algorithm-Specific Parameters

#### ChiMerge
- `sig_level`: Significance level for chi-square test
- `method`: Algorithm variant (chimerge, chimerge_mod, etc.)

#### DMIV
- `divergence_method`: Divergence measure to optimize
- `woe_method`: WoE calculation method
- `enforce_monotonicity`: Enable monotonicity constraints
- `l1_regularization` / `l2_regularization`: Regularization parameters

#### CART
- `criterion`: Impurity criterion (gini, entropy)
- `use_pruning`: Enable cost-complexity pruning
- `monotonic_trend`: Desired monotonic relationship

## Performance Considerations

- **C++ Implementation**: Core algorithms implemented in C++ for speed
- **Parallel Processing**: OpenMP support for multi-core systems
- **Memory Efficiency**: Optimized data structures and algorithms
- **Scalability**: Designed to handle large datasets efficiently

## Use Cases

1. **Credit Scoring**: Transform financial variables for credit risk models
2. **Marketing Analytics**: Bin customer attributes for targeting models
3. **Healthcare**: Process medical data for predictive models
4. **General Feature Engineering**: Preprocessing for machine learning models

## Contributing

We welcome contributions to the obwoe package. Please feel free to submit issues, feature requests, or pull requests on our GitHub repository.

## License

This package is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

José Lopes (LOPES, J.E.) - <evandeilton@gmail.com>

## References

- Kerber, R. (1992). ChiMerge: Discretization of numeric attributes.
- Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees.
- Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring.

## See Also

- [scorecard](https://cran.r-project.org/package=scorecard): Traditional credit scorecard development
- [woe](https://cran.r-project.org/package=woe): Basic WoE transformation utilities