#!/bin/bash
# =============================================================================
# ROAD Benchmark Experiments Runner
# =============================================================================
# This script runs ROAD benchmark experiments for CIFAR-10, CIFAR-100, and ImageNet
# Usage: ./run_road_experiments.sh [dataset] [test_subset_size]
#   dataset: cifar10, cifar100, imagenet, or all (default: all)
#   test_subset_size: number of test samples (default: 10)
#  ./run_road_experiments.sh all 50
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/mnt/abka03/Projects/road_imputation"
PYTHON_BIN="/mnt/abka03/.conda/road/bin/python"
OUTPUT_DIR="./output"
IMPUTATIONS="linear telea ns"
RANKINGS="sort threshold"
STAGES="train explain benchmark analyze"

# Default values
DATASET="${1:-all}"
TEST_SUBSET_SIZE="${2:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================================="
    echo -e "${BLUE}$1${NC}"
    echo "============================================================================="
}

# Function to run CIFAR-10 experiment
run_cifar10() {
    print_header "Running CIFAR-10 Experiment"
    print_info "Test subset size: $TEST_SUBSET_SIZE"
    
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" $PYTHON_BIN -m experiments.unified.run_pipeline \
        --dataset cifar10 \
        --imputations $IMPUTATIONS \
        --rankings $RANKINGS \
        --stages $STAGES \
        --test_subset_size $TEST_SUBSET_SIZE \
        --output_dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        print_success "CIFAR-10 experiment completed!"
    else
        print_error "CIFAR-10 experiment failed!"
        return 1
    fi
}

# Function to run CIFAR-100 experiment
run_cifar100() {
    print_header "Running CIFAR-100 Experiment"
    print_info "Test subset size: $TEST_SUBSET_SIZE"
    
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" $PYTHON_BIN -m experiments.unified.run_pipeline \
        --dataset cifar100 \
        --data_path /mnt/abka03/xlvlm_data \
        --imputations $IMPUTATIONS \
        --rankings $RANKINGS \
        --stages $STAGES \
        --test_subset_size $TEST_SUBSET_SIZE \
        --output_dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        print_success "CIFAR-100 experiment completed!"
    else
        print_error "CIFAR-100 experiment failed!"
        return 1
    fi
}

# Function to run ImageNet experiment
run_imagenet() {
    print_header "Running ImageNet Experiment"
    print_info "Test subset size: $TEST_SUBSET_SIZE"
    
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" $PYTHON_BIN -m experiments.unified.run_pipeline \
        --dataset imagenet \
        --data_path /mnt/abka03/xlvlm_data/imagenet_1000 \
        --imputations $IMPUTATIONS \
        --rankings $RANKINGS \
        --stages $STAGES \
        --test_subset_size $TEST_SUBSET_SIZE \
        --output_dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        print_success "ImageNet experiment completed!"
    else
        print_error "ImageNet experiment failed!"
        return 1
    fi
}

# Function to run all experiments
run_all() {
    print_header "Running All ROAD Experiments"
    print_info "Datasets: CIFAR-10, CIFAR-100, ImageNet"
    print_info "Test subset size: $TEST_SUBSET_SIZE"
    print_info "Imputations: $IMPUTATIONS"
    print_info "Rankings: $RANKINGS"
    print_info "Stages: $STAGES"
    
    START_TIME=$(date +%s)
    
    run_cifar10
    run_cifar100
    run_imagenet
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    print_header "All Experiments Completed!"
    print_success "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [dataset] [test_subset_size]"
    echo ""
    echo "Arguments:"
    echo "  dataset         Dataset to run: cifar10, cifar100, imagenet, or all (default: all)"
    echo "  test_subset_size Number of test samples to use (default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all datasets with 10 samples"
    echo "  $0 cifar10            # Run only CIFAR-10 with 10 samples"
    echo "  $0 cifar10 100        # Run CIFAR-10 with 100 samples"
    echo "  $0 all 50             # Run all datasets with 50 samples"
    echo ""
    echo "Configuration:"
    echo "  Project root: $PROJECT_ROOT"
    echo "  Python: $PYTHON_BIN"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Imputations: $IMPUTATIONS"
    echo "  Rankings: $RANKINGS"
    echo "  Stages: $STAGES"
}

# Main execution
print_header "ROAD Benchmark Experiments Runner"
print_info "Started at: $(date)"

case "$DATASET" in
    imagenet)
        run_imagenet
        ;;
    all)
        run_all
        ;;
    -h|--help|help)
        show_usage
        exit 0
        ;;
    *)
        print_error "Unknown dataset: $DATASET"
        show_usage
        exit 1
        ;;
esac

print_info "Finished at: $(date)"
print_success "Results saved to: $PROJECT_ROOT/$OUTPUT_DIR"
