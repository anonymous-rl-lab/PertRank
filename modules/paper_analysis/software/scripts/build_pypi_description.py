"""Build PyPI prose without repository-relative links that PyPI cannot serve."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def render():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    def link(match):
        label, target = match.groups()
        if target.startswith(('https://', 'http://', 'mailto:', '#')):
            return match.group(0)
        return f'{label} (`{target}` in the source archive)'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)


if __name__ == '__main__':
    (ROOT / 'PYPI_DESCRIPTION.md').write_text(render(), encoding='utf-8')
