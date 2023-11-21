"""
"""

def bscolors(sens: str) -> dict:
    """
    """
    if sens in ["Price","price"]:
        return "#2980b9" # blue 
    elif sens in ["Delta","delta"]:
        return "#cb4335" # red 
    elif sens in ["Gamma","gamma"]:
        return "#a569bd" # violet
    elif sens in ["Theta","theta"]:
        return "#f1c40f" # yellow
    elif sens in ["Vega","vega"]:
        return "#1abc9c" # green
    elif sens in ["Lambda","lambda"]:
        return "#7D7D7D" # gray


@staticmethod 
def get_Smin(
    n: float, 
    bnd: float = 0.60
) -> float:
    """
    """
    return round(n * (1 - bnd), 0)

@staticmethod 
def get_Smax(
    n: float, 
    bnd: float = 0.60
) -> float:
    """
    """
    return round(n * (1 + bnd), 0)