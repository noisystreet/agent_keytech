#!/bin/bash
# RST 文档预提交检查脚本
# ==========================
# 用法:
#   ./scripts/precommit-check.sh          # 检查所有 RST 文件
#   ./scripts/precommit-check.sh --staged  # 只检查暂存区中的 RST 文件
#   ./scripts/precommit-check.sh --hook    # 作为 git pre-commit hook 运行
#
# 返回码: 0=通过, 1=语法错误, 2=构建警告

set -e

if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel)"
else
    cd "$(dirname "$(readlink -f "$0" || echo "$0")")"
    PROJECT_ROOT="$(cd .. && pwd)"
fi
BUILD_DIR="_build/precommit-check"
EXIT_CODE=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_sphinx() {
    if ! python3 -c "import sphinx" 2>/dev/null; then
        echo -e "${RED}错误: 未安装 Sphinx。请先运行: pip install -r requirements.txt${NC}"
        exit 1
    fi
}

check_rst_inline_markup() {
    local files=("$@")
    local has_error=0

    for f in "${files[@]}"; do
        if grep -Pn ':\w+:`[^`]*`[（）]' "$f" &>/dev/null; then
            [ $has_error -eq 0 ] && echo -e "${YELLOW}⚠  角色标记后紧跟中文括号（缺少空格）:${NC}"
            echo -e "  ${YELLOW}$f${NC}"
            grep -Pn ':\w+:`[^`]*`[（）]' "$f" | while read -r line; do
                echo "    $line"
            done
            has_error=1
        fi
        if grep -Pn '\*\*[^*]*\*\*[（，]' "$f" &>/dev/null; then
            [ $has_error -eq 0 ] && echo -e "${YELLOW}⚠  **bold** 后紧跟中文标点（缺少空格）:${NC}"
            echo -e "  ${YELLOW}$f${NC}"
            grep -Pn '\*\*[^*]*\*\*[（，]' "$f" | while read -r line; do
                echo "    $line"
            done
            has_error=1
        fi
    done
    return $has_error
}

check_rst_files() {
    local files=("$@")
    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${YELLOW}没有 RST 文件需要检查。${NC}"
        return 0
    fi

    echo -e "${YELLOW}检查以下 RST 文件:${NC}"
    for f in "${files[@]}"; do
        echo "  - $f"
    done
    echo ""

    rm -rf "$PROJECT_ROOT/$BUILD_DIR"

    echo -e "${YELLOW}运行 Sphinx 语法检查...${NC}"
    if python3 -m sphinx -b dummy "$PROJECT_ROOT/source" "$PROJECT_ROOT/$BUILD_DIR" 2>/tmp/sphinx_precommit_err.txt 1>/dev/null; then
        if grep -qE '(WARNING|ERROR)' /tmp/sphinx_precommit_err.txt 2>/dev/null; then
            echo -e "${YELLOW}⚠  构建成功，但有警告:${NC}"
            grep -E '(WARNING|ERROR)' /tmp/sphinx_precommit_err.txt
            EXIT_CODE=2
        else
            echo -e "${GREEN}✓ 所有 RST 文件语法正确，无警告。${NC}"
            EXIT_CODE=0
        fi
    else
        echo -e "${RED}✗ RST 语法错误！${NC}"
        cat /tmp/sphinx_precommit_err.txt
        EXIT_CODE=1
    fi

    rm -rf "$PROJECT_ROOT/$BUILD_DIR"
    return $EXIT_CODE
}

check_sphinx

if [ "$1" = "--hook" ]; then
    STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rst$' || true)
    [ -z "$STAGED_FILES" ] && { echo -e "${GREEN}没有 RST 文件被暂存。${NC}"; exit 0; }
    check_rst_files $STAGED_FILES
    EXIT_CODE=$?
    [ -n "$STAGED_FILES" ] && check_rst_inline_markup $STAGED_FILES
elif [ "$1" = "--staged" ]; then
    STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rst$' || true)
    [ -z "$STAGED_FILES" ] && { echo -e "${GREEN}没有 RST 文件被暂存。${NC}"; exit 0; }
    check_rst_files $STAGED_FILES
else
    echo -e "${YELLOW}=== 检查所有 RST 文档 ===${NC}"
    RST_FILES=$(find "$PROJECT_ROOT/source" -name '*.rst' | sort)
    check_rst_files $RST_FILES
    check_rst_inline_markup $RST_FILES
fi

case $EXIT_CODE in
    0) echo -e "${GREEN}✓ 检查通过。${NC}" ;;
    2) echo -e "${YELLOW}⚠  检查通过但有警告。${NC}" ;;
    *) echo -e "${RED}✗ 检查未通过。${NC}" ;;
esac
exit $EXIT_CODE
