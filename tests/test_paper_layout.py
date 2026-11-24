import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / 'checks' / 'render_paper_pdf.py'
SPEC = importlib.util.spec_from_file_location('render_paper_pdf', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inline_markup_is_removed_without_changing_math():
    value = MODULE.clean_inline(r'**Regret** is \(R_t^\pi\) and `V=1-r`.')
    assert value == 'Regret is $R_t^\\pi$ and V=1-r.'
    assert '**' not in value and '`' not in value


def test_markdown_table_becomes_typed_rows():
    rows = MODULE.parse_table([
        '| Context | cap 1 | cap 2 |',
        '|:---|---:|---:|',
        '| Sample | 1.905 | 1.528 |',
    ])
    assert rows == [['Context', 'cap 1', 'cap 2'], ['Sample', '1.905', '1.528']]


def test_parser_groups_tables_and_headings():
    blocks = MODULE.parse_blocks('# Title\n\n## 1. Introduction\n\n| A | B |\n|---|---|\n| 1 | 2 |\n')
    assert [block[0] for block in blocks] == ['title', 'section', 'table']
    assert blocks[-1][1] == [['A', 'B'], ['1', '2']]


def test_urls_are_not_broken_at_hyphens():
    lines = MODULE.wrap_text('https://github.com/Cunyu-Liu/proxy-function-gap', 80)
    assert lines == ['https://github.com/Cunyu-Liu/proxy-function-gap']
