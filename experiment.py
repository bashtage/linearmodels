from linearmodels.panel.model import BetweenOLS, PanelOLS, FirstDifferenceOLS, PooledOLS
from linearmodels.tests.panel._utility import generate_data

data = generate_data(0.20, 'pandas')
for model in [BetweenOLS, PanelOLS, FirstDifferenceOLS, PooledOLS]:
    formula = 'y ~ 1 + x0 + x1 + x2 + x3 + x4'
    if model is FirstDifferenceOLS:
        formula = 'y ~ x0 + x1 + x2 + x3 + x4'

    joined = data.x
    joined['y'] = data.y
    mod = model.from_formula(formula, joined)
    res = mod.fit()
    print(res)