import json
from pathlib import Path


def main():
    repo = Path('/home/proxy-function-gap')
    paper = repo / 'paper/PAPER1_DRAFT.md'
    text = paper.read_text()
    root = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap')
    checks = {}
    checks['references_15'] = text.count('\n[') >= 15 and '[15]' in text
    checks['retrospective_boundary'] = 'retrospective historical replay' in text and 'not prospective wet-lab validation' in text
    execution_record = (repo / 'docs/VGSC_THREE_PAPER_EXECUTION.md').read_text()
    checks['negative_vgsc_retained_internally'] = 'FP-REJ+Q 在 designed 上覆盖率为 0' in execution_record and '不得将 VGSC 写成 Paper 2 有效方法' in execution_record
    checks['vgsc_failure_not_submitted_as_paper1_result'] = 'FP-REJ without Q' not in text and 'Q rejects every selection' not in text
    checks['positive_main_findings'] = '0.931–0.965' in text and '1.905 to 0.681' in text
    checks['score_matching_boundary'] = 'All intervals include zero' in text
    checks['github_availability_current'] = 'https://github.com/Cunyu-Liu/proxy-function-gap' in text and 'GitHub upload remains pending' not in text
    forbidden = ['first-ever', 'proves that', 'wet-lab validated', 'universally superior', 'completed preprint']
    checks['no_overclaim'] = not any(value in text.lower() for value in forbidden)
    expected = {
        'castillo_n50_v2/bon_pressure.json': [6.422002342701147, 6.340792198324439, 6.616574110809051, 6.70510216133283, 6.518881162165146],
        'castillo_n50_sensitivity_v1/bon_pressure.json': [6.521185847054905, 5.9143699563918535, 5.624681693524055, 5.618007961847374, 5.726171101752204],
    }
    numeric = True
    for relative, values in expected.items():
        source = json.loads((root / 'evaluation' / relative).read_text())
        actual = [row['mean_regret'] for row in source['rows']]
        numeric = numeric and all(abs(a - b) < 1e-9 for a, b in zip(actual, values))
        for value in actual:
            numeric = numeric and f'{value:.3f}' in text
    checks['pressure_values_match_source'] = numeric
    cross = json.loads((root / 'mechanism/n50_v1/cross_arch/summary.json').read_text())
    checks['cross_arch_values_match_source'] = abs(cross['architecture_error_correlation'] - 0.9617204911661739) < 1e-12 and '0.962' in text
    checks['pdf_exists'] = (root / 'paper/PAPER1_DRAFT.pdf').is_file() and (root / 'paper/PAPER1_DRAFT.pdf').stat().st_size > 10000
    checks['external_reviews'] = all(json.loads((root / 'mechanism/n50_v1/cross_arch' / f'independent_validation_round{i}.json').read_text())['pass'] for i in (1, 2, 3))
    report = {'pass': all(checks.values()), 'checks': checks, 'paper': str(paper), 'pdf': str(root / 'paper/PAPER1_DRAFT.pdf')}
    (root / 'paper/preprint_claim_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise ValueError('paper claim audit failed')


if __name__ == '__main__':
    main()
