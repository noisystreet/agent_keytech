# LLM Agent 核心技术 - Sphinx Configuration
# ===========================================

from datetime import datetime

project = '大模型 Agent 核心技术'
author = 'noisystreet'
copyright = f'{datetime.now().year}, {author}'

version = '1.0'
release = '1.0'

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]

mermaid_output_format = 'raw'

templates_path = ['_templates']
language = 'zh_CN'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

autosectionlabel_prefix_document = True
todo_include_todos = True

source_suffix = {
    '.rst': 'restructuredtext',
}
