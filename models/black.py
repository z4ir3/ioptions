"""
models.black.py
Black '76 option pricing model - (European) Index Options
"""
import pandas as pd
import numpy as np

from models.normaldist import Normal
from models.utils import DotDict

class Black:
    """
    Black '76 Option Pricing Model for (European) Index Options
    """
    def __init__(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        v: float, 
        q: float = 0
    ):
        """
        S: underlying Price 
        K: strike Price
        r: risk-free interest rate
        T: years-to-maturity (YTM) 
        v: implied volatility
        """
        self.S = Black._valid_underlying(S)    
        self.K = Black._valid_strike(K)
        self.T = Black._valid_maturity(T)
        self.r = Black._valid_intrate(r)
        self.v = Black._valid_vola(v)
    
    @staticmethod 
    def _valid_underlying(S: float) -> float:
        """
        Validate input underlying price
        """
        if S > 0:
            return S
        else:
            raise ValueError(f"The Underlying Price must be greater than 0 (got{S})")
            
    @staticmethod 
    def _valid_strike(K: float) -> float:
        """
        Validate input strike price
        """
        if K > 0:
            return K
        else:
            raise ValueError(f"The Srike Price must be greater than 0 (got{K})")
    
    @staticmethod 
    def _valid_maturity(T: float) -> float:
        """
        Validate input Years-to-Maturity (YTE)
        """
        if T >= 0:
            return T
        else:
            raise ValueError(f"The Years-to-Maturity cannot be negative (got{T})")

    @staticmethod 
    def _valid_intrate(r: float) -> float:
        """
        Validate input interest rate
        """
        if r >= 0:
            return r
        else:
            raise ValueError(f"The Interest Rate cannot be negative (got{r})")

    @staticmethod 
    def _valid_vola(v: float) -> float:
        """
        Validate input volatility
        """
        if v > 0:
            return v
        else:
            raise ValueError(f"The Implied Volatility must be greater than 0 (got{v})")

    # Original: Npdf and Ncdf inside
    # @staticmethod
    # def _Npdf(x: float) -> float:
    #     """
    #     Standard Normal PDF evaluated at the input point x
    #     """  
    #     return 1 / (np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2) 

    # def _Ncdf(
    #     self, 
    #     x: float,
    #     a: float = +0.4361836,
    #     b: float = -0.1201676,
    #     c: float = +0.9372980,
    # ) -> float:
    #     """
    #     Standard Normal CDF evaluated at the input point x.
    #     Computed with 3rd degree polynomial approximation 
    #     precise up to the 5th decimal point (Abramowitz and Stegun)
    #     """
    #     vk = 1 / (1 + 0.33267 * np.abs(x))
    #     pdf = self._Npdf(x) * (a * vk + b * vk**2 + c * vk**3) 
    #     return pdf if x <= 0 else 1-pdf
    
    # New: Npdf and Ncdf imported
    @staticmethod
    def _Npdf(x: float):
        N = Normal()
        return N.pdf(x)
    
    @staticmethod
    def _Ncdf(x: float):
        N = Normal()
        return N.cdf(x)

    def _d1(self, S: float) -> float:
        """
        Compute the quantity "d1" of Black '76 option pricing model
        """
        return (np.log(S/self.K) + 0.5 * self.v**2 * self.T) / (self.v * np.sqrt(self.T))
        
    def _d2(self, S: float) -> float:
        """
        Compute the quantity "d2" of Black '76 option pricing model
        """
        return self._d1(S) - self.v * np.sqrt(self.T)
    

class BlackCall(Black):
    """
    Black '76 Model for Call Options
    """
    def __init__(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        v: float, 
    ):
        super().__init__(S=S, K=K, T=T, r=r, v=v)

    @property
    def params(self) -> dict:
        """
        Returns all input option parameters
        """
        dic = {
            "type": "C",
            "style": "European",
            "model": "Black",
            "S": self.S, 
            "K": self.K, 
            "T": self.T,
            "r": self.r,
            "v": self.v,
        }
        return DotDict(dic)
    
    def greeks(
        self,
        grk: str | None = None
        # rnd: int = 2
    ) -> dict:
        """
        Call greeks
        """
        if grk == "Price":
            return self.price()
        elif grk == "Delta":
            return self.delta()
        elif grk == "Gamma":
            return self.gamma()
        elif grk == "Vega":
            return self.vega()
        elif grk == "Theta":
            return self.theta()
        elif grk == "Rho":
            return self.rho()
        elif grk is None:
            return {
                "Price": self.price(),
                "Delta": self.delta(),
                "Gamma": self.gamma(),
                "Theta": self.theta(),
                "Vega": self.vega(),
                "Rho": self.rho()
            }
        else:
            raise ValueError("Wrong input greek name")
    
    def price(self, *argv) -> float:
        """
        Call Price
        """
        try:
            S = argv[0]
        except:
            S = self.S
        if self.T > 0:
            # The Option has not expired yet
            return (
                + S 
                * np.exp(-self.r * self.T) 
                * self._Ncdf(self._d1(S))
                - self.K 
                * np.exp(-self.r * self.T) 
                * self._Ncdf(self._d2(S))
            )
        else:
            # The Option has expired
            return max(S - self.K, 0)
        
    def delta(self, *argv) -> float:
        """
        Call Delta
        First derivative of the Price with respect to the Underlying Price 
        """  
        try:
            S = argv[0]
        except:
            S = self.S 
        if self.T > 0:
            # The Option has not expired yet
            return np.exp(-self.r * self.T) * self._Ncdf(self._d1(S)) 
        else:
            # The Option has expired
            return +1 if self.price(S) > 0 else 0

    def gamma(self, *argv) -> float:
        """
        Call Gamma (equivalent to the Put gamma)
        First derivative of the Delta with respect to the Underlying Price, i.e., 
        second derivative of the Price with respect to the Underlying Price 
        """  
        try:
            S = argv[0]
        except:
            S = self.S
        if self.T > 0:
            # The Option has not expired yet
            return (
                + np.exp(-self.r * self.T) 
                * self._Npdf(self._d1(S))
                / (S * self.v * np.sqrt(self.T))
            )
        else:
            # The Option has expired
            return 0
    
    def theta(self, *argv) -> float:
        """
        Call Theta
        First Derivative of the Price with respect to the Years-to-Maturity
        """
        try:
            S = argv[0]
        except:
            S = self.S
        # Call Option
        if self.T > 0:
            # The Option has not expired yet
            return (
                - np.exp(-self.r * self.T) 
                * self.S 
                * self.v 
                * self._Npdf(self._d1(S)) 
                / (2 * np.sqrt(self.T)) 
                + self.r * self.S * np.exp(-self.r * self.T) * self._Ncdf(self._d1(S))
                - self.r * self.K * np.exp(-self.r * self.T) * self._Ncdf(self._d2(S))
            ) * 1/365
        else:
            # The Option has expired
            return 0

    def vega(self, *argv) -> float:
        """
        Call Vega (equivalent to Put vega):
        First Derivative of the Price with respect to the (Implied) Volatility
        """    
        try:
            S = argv[0]
        except:
            S = self.S  
        if self.T > 0:
            # The Option has not expired yet
            return (
                + np.exp(-self.r*self.T) 
                * self.S 
                * np.sqrt(self.T)
                * self._Npdf(self._d1(S)) 
            ) * 1/100
        else:
            # The Option has expired
            return 0

    def rho(self, *argv) -> float:
        """
        Call Rho:
        First Derivative of the Price with respect to the Interest Rate
        """    
        try:
            S = argv[0]
        except:
            S = self.S  
        if self.T > 0:
            # The Option has not expired yet
            return (
                - self.T 
                * np.exp(-self.r * self.T)
                * self.S * self._Ncdf(self._d1(S))
                + self.T 
                * np.exp(-self.r * self.T)
                * self.K * self._Ncdf(self._d2(S))
            ) * 1/100
        else:
            # The Option has expired
            return 0


class BlackPut(Black):
    """
    Black '76 Model for Put Options
    """
    def __init__(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        v: float, 
    ):
        super().__init__(S=S, K=K, T=T, r=r, v=v)

    @property
    def params(self) -> dict:
        """
        Returns all input option parameters
        """
        dic = {
            "type": "P",
            "style": "European",
            "model": "Black",
            "S": self.S, 
            "K": self.K, 
            "T": self.T,
            "r": self.r,
            "v": self.v,
        }
        return DotDict(dic)
    
    def greeks(
        self,
        grk: str | None = None
        # rnd: int = 2
    ) -> dict:
        """
        Call greeks
        """
        if grk == "Price":
            return self.price()
        elif grk == "Delta":
            return self.delta()
        elif grk == "Gamma":
            return self.gamma()
        elif grk == "Vega":
            return self.vega()
        elif grk == "Theta":
            return self.theta()
        elif grk == "Rho":
            return self.rho()
        elif grk is None:
            return {
                "Price": self.price(),
                "Delta": self.delta(),
                "Gamma": self.gamma(),
                "Theta": self.theta(),
                "Vega": self.vega(),
                "Rho": self.rho()
            }
        else:
            raise ValueError("Wrong input greek name")
    
    def price(self, *argv) -> float:
        """
        Put Price
        """
        try:
            S = argv[0]
        except:
            S = self.S
        if self.T > 0:
            # The Option has not expired yet
            return (
                - self.S 
                * np.exp(-self.r * self.T) 
                * self._Ncdf(-self._d1(S)) 
                + self.K 
                * np.exp(-self.r * self.T) 
                * self._Ncdf(-self._d2(S))
             )
        else:
            # The Option has expired
            return max(self.K - S, 0)
    
    def delta(self, *argv) -> float:
        """
        Put Delta
        First derivative of the Price with respect to the Underlying Price 
        """  
        try:
            S = argv[0]
        except:
            S = self.S 
        if self.T > 0:
            # The Option has not expired yet
            return -np.exp(-self.r * self.T) * self._Ncdf(-self._d1(S))
        else:
            # The Option has expired
            return -1 if self.price(S) > 0 else 0
        
    def gamma(self, *argv) -> float:
        """
        Put Gamma (equivalent to the Call Gamma)
        First derivative of the Delta with respect to the Underlying Price, i.e., 
        second derivative of the Price with respect to the Underlying Price 
        """  
        try:
            S = argv[0]
        except:
            S = self.S
        if self.T > 0:
            # The Option has not expired yet
            return (
                + np.exp(-self.r * self.T) 
                * self._Npdf(self._d1(S))
                / (S * self.v * np.sqrt(self.T))
            )
        else:
            # The Option has expired
            return 0
    
    def theta(self, *argv) -> float:
        """
        Put Theta
        First Derivative of the Price with respect to the Years-to-Maturity
        """
        try:
            S = argv[0]
        except:
            S = self.S
        # Call Option
        if self.T > 0:
            # The Option has not expired yet
            return (
                - np.exp(-self.r * self.T) 
                * self.S 
                * self.v 
                * self._Npdf(self._d1(S)) 
                / (2 * np.sqrt(self.T))
                - self.r * self.S * np.exp(-self.r * self.T) * self._Ncdf(-self._d1(S))
                + self.r * self.K * np.exp(-self.r * self.T) * self._Ncdf(-self._d2(S))
            ) * 1/365   
        else:
            # The Option has expired
            return 0
    
    def vega(self, *argv) -> float:
        """
        Put Vega (equivalent to Call vega):
        First Derivative of the Price with respect to the (Implied) Volatility
        """    
        try:
            S = argv[0]
        except:
            S = self.S  
        if self.T > 0:
            # The Option has not expired yet
            return (
                + np.exp(-self.r * self.T) 
                * self.S 
                * np.sqrt(self.T)
                * self._Npdf(self._d1(S)) 
            ) * 1/100
        else:
            # The Option has expired
            return 0

    def rho(self, *argv) -> float:
        """
        Put Rho:
        First Derivative of the Price with respect to the Interest Rate
        """    
        try:
            S = argv[0]
        except:
            S = self.S  
        if self.T > 0:
            # The Option has not expired yet
            return (
                - self.T 
                * np.exp(-self.r * self.T) 
                * self.K
                * self._Ncdf(-self._d2(S)) 
                + self.T 
                * np.exp(-self.r * self.T) 
                * self.S
                * self._Ncdf(-self._d1(S)) 
            ) * 1/100
        else:
            # The Option has expired
            return 0
