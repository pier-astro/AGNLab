__version__ = "0.1.0"

from .spectrum import *
from .tools import *

from . import models
from . import instrument
from . import host
from . import cubefit
# from . import spectrum
from . import tools





### Setup for dark themes in Jupyter/IPython environments
def _setup_notebook_theme():
    """Injects CSS for dark themes in IPython environments."""
    try:
        # Fail fast if not in an IPython environment
        get_ipython()

        from IPython.display import display, Javascript
        from pathlib import Path
        import textwrap

        css_path = Path(__file__).parent / 'aux' / 'css_settings.css'
        if not css_path.is_file():
            return

        css_content = css_path.read_text().replace('`', r'\`')

        js_script = textwrap.dedent(f"""
            (function() {{
                const isDark = document.body.classList.contains('vscode-dark') || // VS Code
                               document.body.classList.contains('jp-mod-dark') || // JupyterLab
                               (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches); // System preference

                if (isDark) {{
                    const styleId = 'custom-dark-theme-css';
                    if (!document.getElementById(styleId)) {{
                        const style = document.createElement('style');
                        style.id = styleId;
                        style.textContent = `{css_content}`;
                        document.head.appendChild(style);
                    }}
                }}
            }})();
        """)
        display(Javascript(js_script))
    except (NameError, ImportError):
        pass # Not in an IPython shell, do nothing.

_setup_notebook_theme()
del _setup_notebook_theme