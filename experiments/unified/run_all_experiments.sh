#!/bin/bash
# =============================================================================
# ROAD Benchmark - Run All Experiments
# =============================================================================
# This script runs the complete ROAD benchmark pipeline for all datasets
# with all imputation methods (linear, telea, ns) and both ranking strategies
# (sort-based and threshold-based).
#
# Usage:
#   ./run_all_experiments.sh                    # Run with defaults
#   TEST_SIZE=500 ./run_all_experiments.sh      # Custom test size
#   DATASETS="cifar10" ./run_all_experiments.sh # Single dataset
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration (can be overridden via environment variables)
# =============================================================================

# Datasets to process
DATASETS="${DATASETS:-cifar10 food101}"

# Imputation methods
IMPUTATIONS="${IMPUTATIONS:-linear telea ns}"

# Ranking methods (both sort and threshold)
RANKINGS="${RANKINGS:-sort threshold}"

# Removal orders
ORDERS="${ORDERS:-morf lerf}"

# Explanation methods
EXPL_METHODS="${EXPL_METHODS:-ig gb ig_sg gb_sg}"

# Test subset size (None = use all, set a number for faster runs)
TEST_SIZE="${TEST_SIZE:-}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Figure format
FIG_FORMAT="${FIG_FORMAT:-png}"

# Enable imputation visualization
VISUALIZE="${VISUALIZE:-true}"

# =============================================================================
# Setup
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

# Create logs directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_pipeline() {
    local dataset=$1
    local log_file="$LOG_DIR/${dataset}_${TIMESTAMP}.log"
    
    log "Starting $dataset pipeline..."
    log "Log file: $log_file"
    
    # Build command
    local cmd="python -m experiments.unified.run_pipeline"
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --imputations $IMPUTATIONS"
    cmd="$cmd --rankings $RANKINGS"
    cmd="$cmd --orders $ORDERS"
    cmd="$cmd --explanation_methods $EXPL_METHODS"
    cmd="$cmd --stages explain benchmark analyze"
    cmd="$cmd --output_dir $OUTPUT_DIR"
    cmd="$cmd --figure_format $FIG_FORMAT"
    
    # Add test size if specified
    if [ -n "$TEST_SIZE" ]; then
        cmd="$cmd --test_subset_size $TEST_SIZE"
    fi
    
    # Add visualization flag
    if [ "$VISUALIZE" = "true" ]; then
        cmd="$cmd --visualize_imputations"
    fi
    
    log "Command: $cmd"
    
    # Run and log
    if $cmd 2>&1 | tee "$log_file"; then
        log "✓ $dataset completed successfully"
        return 0
    else
        log "✗ $dataset failed - check $log_file"
        return 1
    fi
}

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================================="
echo "ROAD Benchmark - Run All Experiments"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Project Root:   $PROJECT_ROOT"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Datasets:       $DATASETS"
echo "  Imputations:    $IMPUTATIONS"
echo "  Rankings:       $RANKINGS"
echo "  Orders:         $ORDERS"
echo "  Explanations:   $EXPL_METHODS"
echo "  Test Size:      ${TEST_SIZE:-all}"
echo "  Visualize:      $VISUALIZE"
echo "  Figure Format:  $FIG_FORMAT"
echo ""
echo "=============================================================="
echo ""

# =============================================================================
# Run Experiments
# =============================================================================

FAILED_DATASETS=""
SUCCESS_COUNT=0
TOTAL_COUNT=0

for dataset in $DATASETS; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    echo ""
    echo "############################################################"
    echo "# Dataset: $dataset ($TOTAL_COUNT)"
    echo "############################################################"
    echo ""
    
    if run_pipeline "$dataset"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_DATASETS="$FAILED_DATASETS $dataset"
    fi
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
echo ""
echo "Total datasets:    $TOTAL_COUNT"
echo "Successful:        $SUCCESS_COUNT"
echo "Failed:            $((TOTAL_COUNT - SUCCESS_COUNT))"

if [ -n "$FAILED_DATASETS" ]; then
    echo ""
    echo "Failed datasets:$FAILED_DATASETS"
    echo ""
    echo "Check logs in: $LOG_DIR"
    exit 1
else
    echo ""
    echo "All experiments completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - Tables:  $OUTPUT_DIR/{dataset}/tables/"
    echo "  - Figures: $OUTPUT_DIR/{dataset}/figures/"
    echo "  - Results: $OUTPUT_DIR/{dataset}/results/"
    exit 0
fi
