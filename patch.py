content = open('ablation_study.py', encoding='utf-8').read()
content = content.replace(
    "', '.join(shared[:5])}",
    "', '.join(t.get('target_name', str(t)) if isinstance(t, dict) else str(t) for t in shared[:5])}"
)
open('ablation_study.py', 'w', encoding='utf-8').write(content)
print('Fixed.')
