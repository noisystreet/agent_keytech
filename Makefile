# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = source
BUILDDIR      = _build

.DEFAULT_GOAL := html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean precommit check serve

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

precommit:
	@bash scripts/precommit-check.sh

html: Makefile precommit
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 本地预览服务器（默认 8000 端口）
serve: html
	@echo "Open http://localhost:8000/ in your browser"
	@cd $(BUILDDIR)/html && python3 -m http.server $(PORT)

clean:
	rm -rf $(BUILDDIR)
