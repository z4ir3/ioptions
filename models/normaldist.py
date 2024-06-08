"""
models.normaldist.py
"""

import numpy as np

class Normal:
    """
    Normal distribution
    """
    def pdf(self, x: float) -> float:
        """
        Standard Normal PDF evaluated at the input point x
        """  
        return 1 / (np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2) 
    
    def cdf(
        self, 
        x: float,
        a: float = +0.4361836,
        b: float = -0.1201676,
        c: float = +0.9372980,
    ) -> float:
        """
        Standard Normal CDF evaluated at the input point x.
        Computed with 3rd degree polynomial approximation 
        precise up to the 5th decimal point (Abramowitz and Stegun)
        """
        vk = 1 / (1 + 0.33267 * np.abs(x))
        pdf = self.pdf(x) * (a * vk + b * vk**2 + c * vk**3) 
        return pdf if x <= 0 else 1-pdf
    