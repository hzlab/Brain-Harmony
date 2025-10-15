#!/bin/bash

# BrainHarmonix Code Formatter using Ruff
# This script uses ruff to format and lint Python code in the project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
CHECK_ONLY=false
FIX_UNSAFE=false
TARGET_PATH="."
EXCLUDE_PATHS="__pycache__,*.egg-info,.git,build,dist,experiments,checkpoints"

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS] [PATH]"
    echo ""
    echo "Options:"
    echo "  -c, --check-only    Only check code style, don't fix"
    echo "  -f, --fix-unsafe    Fix unsafe issues as well"
    echo "  -h, --help          Show this help message"
    echo "  -p, --path PATH     Specify target path (default: current directory)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Format all Python files in current directory"
    echo "  $0 -c               # Only check without fixing"
    echo "  $0 -p modules/      # Format only modules directory"
    echo "  $0 --fix-unsafe     # Fix both safe and unsafe issues"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--check-only)
            CHECK_ONLY=true
            shift
            ;;
        -f|--fix-unsafe)
            FIX_UNSAFE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--path)
            TARGET_PATH="$2"
            shift 2
            ;;
        *)
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

# Check if ruff is installed
check_ruff_installation() {
    echo -e "${BLUE}=== Checking Ruff Installation ===${NC}"
    
    if ! command -v ruff &> /dev/null; then
        echo -e "${YELLOW}Ruff is not installed. Installing ruff...${NC}"
        pip install ruff
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Ruff installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install ruff${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Ruff is already installed${NC}"
        ruff --version
    fi
}

# Create ruff configuration if it doesn't exist
create_ruff_config() {
    echo -e "${BLUE}=== Setting up Ruff Configuration ===${NC}"
    
    # Check if pyproject.toml exists and add ruff config
    if [ -f "pyproject.toml" ]; then
        # Check if ruff config already exists
        if ! grep -q "\[tool.ruff\]" pyproject.toml; then
            echo -e "${YELLOW}Adding ruff configuration to pyproject.toml...${NC}"
            cat >> pyproject.toml << 'EOF'

[tool.ruff]
# Python version
target-version = "py38"

# Line length
line-length = 88

# Enable specific rule sets
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (handled by formatter)
    "E203",  # Whitespace before ':'
    "W503",  # Line break before binary operator
    "B008",  # Do not perform function calls in argument defaults
]

# Exclude directories
exclude = [
    "__pycache__",
    "*.egg-info",
    ".git",
    "build",
    "dist",
    "experiments",
    "checkpoints",
]

[tool.ruff.format]
# Use single quotes for strings
quote-style = "double"

# Use spaces for indentation
indent-style = "space"

[tool.ruff.isort]
# Import sorting configuration
known-first-party = ["modules", "libs", "datasets", "configs"]
EOF
            echo -e "${GREEN}✓ Ruff configuration added to pyproject.toml${NC}"
        else
            echo -e "${GREEN}✓ Ruff configuration already exists in pyproject.toml${NC}"
        fi
    else
        echo -e "${YELLOW}Creating pyproject.toml with ruff configuration...${NC}"
        cat > pyproject.toml << 'EOF'
[tool.ruff]
# Python version
target-version = "py38"

# Line length
line-length = 88

# Enable specific rule sets
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (handled by formatter)
    "E203",  # Whitespace before ':'
    "W503",  # Line break before binary operator
    "B008",  # Do not perform function calls in argument defaults
]

# Exclude directories
exclude = [
    "__pycache__",
    "*.egg-info",
    ".git",
    "build",
    "dist",
    "experiments",
    "checkpoints",
]

[tool.ruff.format]
# Use double quotes for strings
quote-style = "double"

# Use spaces for indentation
indent-style = "space"

[tool.ruff.isort]
# Import sorting configuration
known-first-party = ["modules", "libs", "datasets", "configs"]
EOF
        echo -e "${GREEN}✓ Created pyproject.toml with ruff configuration${NC}"
    fi
}

# Run ruff check
run_ruff_check() {
    echo -e "${BLUE}=== Running Ruff Check ===${NC}"
    echo -e "${YELLOW}Checking: ${TARGET_PATH}${NC}"
    
    if ruff check "$TARGET_PATH" --exclude "$EXCLUDE_PATHS"; then
        echo -e "${GREEN}✓ No linting issues found${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Found linting issues${NC}"
        return 1
    fi
}

# Run ruff format
run_ruff_format() {
    echo -e "${BLUE}=== Running Ruff Format ===${NC}"
    echo -e "${YELLOW}Formatting: ${TARGET_PATH}${NC}"
    
    if ruff format "$TARGET_PATH" --exclude "$EXCLUDE_PATHS"; then
        echo -e "${GREEN}✓ Code formatted successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to format code${NC}"
        return 1
    fi
}

# Run ruff fix
run_ruff_fix() {
    echo -e "${BLUE}=== Running Ruff Fix ===${NC}"
    echo -e "${YELLOW}Fixing: ${TARGET_PATH}${NC}"
    
    local fix_args="--exclude $EXCLUDE_PATHS"
    if [ "$FIX_UNSAFE" = true ]; then
        fix_args="$fix_args --unsafe-fixes"
        echo -e "${YELLOW}⚠ Including unsafe fixes${NC}"
    fi
    
    if ruff check "$TARGET_PATH" --fix $fix_args; then
        echo -e "${GREEN}✓ Issues fixed successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Some issues may require manual attention${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${GREEN}=== BrainHarmonix Code Formatter ===${NC}"
    echo -e "Target path: ${BLUE}$TARGET_PATH${NC}"
    echo -e "Check only: ${BLUE}$CHECK_ONLY${NC}"
    echo -e "Fix unsafe: ${BLUE}$FIX_UNSAFE${NC}"
    echo ""
    
    # Ensure we're in the project root
    if [ ! -f "README.md" ] && [ ! -d "modules" ]; then
        echo -e "${YELLOW}⚠ Warning: Not in project root directory${NC}"
        echo -e "Current directory: $(pwd)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    # Step 1: Check ruff installation
    check_ruff_installation
    echo ""
    
    # Step 2: Create/update ruff configuration
    create_ruff_config
    echo ""
    
    # Step 3: Check if target path exists
    if [ ! -e "$TARGET_PATH" ]; then
        echo -e "${RED}✗ Error: Target path '$TARGET_PATH' does not exist${NC}"
        exit 1
    fi
    
    # Step 4: Run checks/formatting based on mode
    if [ "$CHECK_ONLY" = true ]; then
        # Only check, don't fix
        run_ruff_check
        check_result=$?
        
        if [ $check_result -eq 0 ]; then
            echo -e "\n${GREEN}🎉 All checks passed! Code is well-formatted.${NC}"
        else
            echo -e "\n${YELLOW}💡 Run without -c/--check-only flag to auto-fix issues.${NC}"
        fi
    else
        # Fix and format
        echo -e "${YELLOW}Starting code formatting process...${NC}"
        
        # Step 4a: Fix linting issues
        run_ruff_fix
        echo ""
        
        # Step 4b: Format code
        run_ruff_format
        echo ""
        
        # Step 4c: Final check
        echo -e "${BLUE}=== Final Verification ===${NC}"
        if run_ruff_check; then
            echo -e "\n${GREEN}🎉 Code formatting completed successfully!${NC}"
        else
            echo -e "\n${YELLOW}⚠ Some issues may still exist. Please review manually.${NC}"
        fi
    fi
}

# Run main function
main "$@"
