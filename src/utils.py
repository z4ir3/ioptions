"""
"""

def bscolors(sens: str) -> dict:
    """
    """
    if sens in {"Price","price"}:
        # blue 
        return "#2980b9" 
    elif sens in {"Delta","delta"}:
        # red
        return "#cb4335"  
    elif sens in {"Gamma","gamma"}:
        # violet
        return "#a569bd" 
    elif sens in {"Theta","theta"}:
        # yellow
        return "#f1c40f" 
    elif sens in {"Vega","vega"}:
        # green
        return "#1abc9c"
    elif sens in {"Lambda","lambda"}:
        # gray
        return "#7D7D7D" 


@staticmethod 
def get_Smin(
    n: float, 
    bnd: float = 0.60
) -> float:
    """
    """
    nn = n * (1 - bnd)
    if n <= 1:
        return nn
    else:
        return round(nn,0)
    
@staticmethod 
def get_Smax(
    n: float, 
    bnd: float = 0.60
) -> float:
    """
    """
    nn = n * (1 + bnd)
    if n <= 1:
        return nn
    else:
        return round(nn,0)

def strategynames() -> list:
    strategies = [
        "Long Call",
        "Short Call",
        "Long Put",
        "Short Put",
        "Bull Call Spread", 
        "Bull Put Spread", 
        "Bear Call Spread", 
        "Bear Put Spread", 
        "Top Strip",
        "Bottom Strip",
        "Top Strap",
        "Bottom Strap",
        "Top Straddle", 
        "Bottom Straddle",
        "Top Strangle",
        "Bottom Strangle",
        "Top Butterfly",
        "Bottom Butterfly",
        "Top Iron Condor",
        "Bottom Iron Condor",
        "Custom strategy"
    ]
    return strategies
