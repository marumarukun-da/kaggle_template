#!/bin/bash
set -e

# =============================================================================
# Kaggle Unified Submission Script
# =============================================================================
# Usage: sh scripts/submit.sh <experiment_name> [options]
#
# Options:
#   --version VER   Specify artifact version to upload (default: latest)
#   --dry-run       Show what would be done without actually pushing
#   --skip-validate Skip validation step
#   --skip-codes    Skip codes dataset upload
#   --skip-artifacts Skip artifacts model upload
#   --update        Update existing model instance instead of creating new
#
# This script handles the entire submission flow:
# 1. Validate submission setup
# 2. Upload codes as Dataset
# 3. Upload artifacts as Model
# 4. Update sub/kernel-metadata.json
# 5. Push submission kernel
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
EXP_NAME=""
VERSION="latest"
DRY_RUN=false
SKIP_VALIDATE=false
SKIP_CODES=false
SKIP_ARTIFACTS=false
UPDATE_MODEL=false

while [ $# -gt 0 ]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-validate)
            SKIP_VALIDATE=true
            shift
            ;;
        --skip-codes)
            SKIP_CODES=true
            shift
            ;;
        --skip-artifacts)
            SKIP_ARTIFACTS=true
            shift
            ;;
        --update)
            UPDATE_MODEL=true
            shift
            ;;
        *)
            if [ -z "$EXP_NAME" ]; then
                EXP_NAME="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo -e "${RED}Error: Experiment name is required.${NC}"
    echo ""
    echo "Usage: sh scripts/submit.sh <experiment_name> [options]"
    echo ""
    echo "Options:"
    echo "  --version VER   Specify artifact version to upload (default: latest)"
    echo "  --dry-run       Show what would be done without actually pushing"
    echo "  --skip-validate Skip validation step"
    echo "  --skip-codes    Skip codes dataset upload"
    echo "  --skip-artifacts Skip artifacts model upload"
    echo "  --update        Update existing model instance"
    exit 1
fi

# Header
echo ""
echo "========================================"
echo -e "${BLUE}Kaggle Submission${NC}"
echo "Experiment: $EXP_NAME"
echo "Version: $VERSION"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN MODE]${NC}"
fi
if [ "$UPDATE_MODEL" = true ]; then
    echo -e "${YELLOW}[UPDATE MODE]${NC}"
fi
echo "========================================"

# =============================================================================
# Step 1: Validation
# =============================================================================
if [ "$SKIP_VALIDATE" = false ]; then
    echo ""
    echo -e "${BLUE}Step 1/5: Validation${NC}"
    echo "----------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: sh scripts/validate.sh $EXP_NAME --skip-inference"
    else
        sh scripts/validate.sh "$EXP_NAME" --skip-inference
        if [ $? -ne 0 ]; then
            echo -e "${RED}Validation failed. Aborting submission.${NC}"
            exit 1
        fi
    fi
else
    echo ""
    echo -e "${YELLOW}Step 1/5: Validation [SKIPPED]${NC}"
fi

# =============================================================================
# Step 2: Upload codes as Dataset
# =============================================================================
if [ "$SKIP_CODES" = false ]; then
    echo ""
    echo -e "${BLUE}Step 2/5: Uploading codes dataset${NC}"
    echo "----------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: python src/upload.py codes"
    else
        echo "Uploading codes..."
        python src/upload.py codes
        if [ $? -ne 0 ]; then
            echo -e "${RED}Codes upload failed. Aborting submission.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Codes uploaded successfully.${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}Step 2/5: Codes upload [SKIPPED]${NC}"
fi

# =============================================================================
# Step 3: Upload artifacts as Model
# =============================================================================
if [ "$SKIP_ARTIFACTS" = false ]; then
    echo ""
    echo -e "${BLUE}Step 3/5: Uploading artifacts model${NC}"
    echo "----------------------------------------"

    UPDATE_FLAG=""
    if [ "$UPDATE_MODEL" = true ]; then
        UPDATE_FLAG="--update True"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: python src/upload.py artifacts --exp_name $EXP_NAME --version $VERSION $UPDATE_FLAG"
    else
        echo "Uploading artifacts for experiment: $EXP_NAME (version: $VERSION)"
        python src/upload.py artifacts --exp_name "$EXP_NAME" --version "$VERSION" $UPDATE_FLAG
        if [ $? -ne 0 ]; then
            echo -e "${RED}Artifacts upload failed. Aborting submission.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Artifacts uploaded successfully.${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}Step 3/5: Artifacts upload [SKIPPED]${NC}"
fi

# =============================================================================
# Step 4: Update sub/kernel-metadata.json and sub/code.ipynb
# =============================================================================
echo ""
echo -e "${BLUE}Step 4/5: Updating sub/kernel-metadata.json and sub/code.ipynb${NC}"
echo "----------------------------------------"

# --- 4a: Update model_sources experiment name in kernel-metadata.json ---
if grep -q "artifacts/other/$EXP_NAME/" "sub/kernel-metadata.json"; then
    echo -e "${GREEN}sub/kernel-metadata.json already references experiment $EXP_NAME${NC}"
else
    CURRENT_MODEL_SRC=$(grep 'artifacts/other/' sub/kernel-metadata.json | head -1 | sed 's/^[[:space:]]*//')
    echo "Updating sub/kernel-metadata.json model_sources..."
    echo "  Before: $CURRENT_MODEL_SRC"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would update model_sources to reference experiment $EXP_NAME"
    else
        sed -i '' -e "s|artifacts/other/[^/]*/|artifacts/other/$EXP_NAME/|g" sub/kernel-metadata.json
        UPDATED_MODEL_SRC=$(grep 'artifacts/other/' sub/kernel-metadata.json | head -1 | sed 's/^[[:space:]]*//')
        echo "  After:  $UPDATED_MODEL_SRC"
        echo -e "${GREEN}sub/kernel-metadata.json updated.${NC}"
    fi
fi

# --- 4b: Update inference.py path in sub/code.ipynb ---
if grep -q "experiments/$EXP_NAME/inference.py" "sub/code.ipynb"; then
    echo -e "${GREEN}sub/code.ipynb already references experiments/$EXP_NAME/inference.py${NC}"
else
    CURRENT_INF=$(grep -o 'experiments/[^/]*/inference.py' sub/code.ipynb | head -1)
    echo "Updating sub/code.ipynb inference path..."
    echo "  Before: $CURRENT_INF"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would update inference path to experiments/$EXP_NAME/inference.py"
    else
        sed -i '' -e "s|experiments/[^/]*/inference.py|experiments/$EXP_NAME/inference.py|g" sub/code.ipynb
        UPDATED_INF=$(grep -o 'experiments/[^/]*/inference.py' sub/code.ipynb | head -1)
        echo "  After:  $UPDATED_INF"
        echo -e "${GREEN}sub/code.ipynb updated.${NC}"
    fi
fi

# =============================================================================
# Step 5: Push submission kernel
# =============================================================================
echo ""
echo -e "${BLUE}Step 5/5: Pushing submission kernel${NC}"
echo "----------------------------------------"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run: cd ./sub && kaggle k push"
    echo ""
    echo -e "${GREEN}[DRY RUN] Submission would complete successfully.${NC}"
else
    echo "Pushing submission kernel..."
    cd ./sub || { echo -e "${RED}Failed to enter ./sub directory${NC}"; exit 1; }
    kaggle k push || { echo -e "${RED}Kaggle push failed${NC}"; cd - > /dev/null; exit 1; }
    cd - > /dev/null

    echo ""
    echo -e "${GREEN}========================================"
    echo "Submission pushed successfully!"
    echo "========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Check submission status: sh scripts/status.sh"
    echo "  2. View on Kaggle: https://www.kaggle.com/code"
fi

echo ""
echo "Done!"
