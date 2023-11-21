"""
Black-Scholes option pricing class
"""

import pandas as pd
import numpy as np

from scipy.stats import norm

class BSOption:
    def __init__(
        self, 
        CP: str, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        v: float, 
        q: float = 0
    ):
        """
        CP    : either "C" (Call) or "P" (Put)
        S     : underlying Price 
        K     : strike Price
        r     : risk-free interest rate
        T     : time-to-maturity in years 
        v     : implied volatility
        q     : dividend yield
        """
        self.CP = BSOption._valid_option(CP)
        self.S = BSOption._valid_underlying(S)    
        self.K = BSOption._valid_strike(K)
        self.T = BSOption._valid_maturity(T)
        self.r = BSOption._valid_intrate(r)
        self.v = BSOption._valid_vola(v)
        self.q = BSOption._valid_yield(q)
    
    @staticmethod 
    def _valid_option(CP):
        """
        Validate input option type
        """
        if CP in ["C","P"]:
            return CP
        else:
            raise ValueError("Argument 'CP' must be either 'C' or 'P'")
            
    @staticmethod 
    def _valid_underlying(S: float) -> float:
        """
        Validate input underlying price
        """
        if S > 0:
            return S
        else:
            raise ValueError("Argument 'S' (underlying price) must be greater than 0")
            
    @staticmethod 
    def _valid_strike(K: float) -> float:
        """
        Validate input strike price
        """
        if K > 0:
            return K
        else:
            raise ValueError("Argument 'K' (strike price) must be greater than 0")
    
    @staticmethod 
    def _valid_maturity(T: float) -> float:
        """
        Validate input maturity
        """
        if T >= 0:
            return T
        else:
            raise ValueError("Argument 'T' (maturity) cannot be negative")

    @staticmethod 
    def _valid_intrate(r: float) -> float:
        """
        Validate input interest rate
        """
        if r >= 0:
            return r
        else:
            raise ValueError("Argument 'r' (interest rate) cannot be negative")

    @staticmethod 
    def _valid_vola(v: float) -> float:
        """
        Validate input volatility
        """
        if v > 0:
            return v
        else:
            raise ValueError("Argument 'v' (volatility) must be greater than 0")
    
    @staticmethod 
    def _valid_yield(q: float) -> float:
        """
        Validate input dividend yield
        """
        if q >= 0:
            return q
        else:
            raise ValueError("Argument 'q' (dividend yield) cannot be negative")
       
    @property
    def params(self) -> dict:
        """
        Returns all input option parameters
        """
        opar = {
            "type": self.CP,
            "S": self.S, 
            "K": self.K, 
            "T": self.T,
            "r": self.r,
            "v": self.v,
            "q": self.q
        }
        return opar
     
    @staticmethod
    def N(
        x: float, 
        cum: bool = True
    ) -> float:
        """
        Standard Normal CDF or PDF evaluated at the input point x.
        """  
        if cum:
            # Returns the standard normal CDF
            return norm.cdf(x, loc=0, scale=1)
        else:
            # Returns the standard normal PDF
            return norm.pdf(x, loc=0, scale=1)

    def _d1(self, S: float) -> float:
        """
        Compute the quantity "d1" of Black-Scholes option pricing
        """
        return ( np.log(S/self.K) + (self.r - self.q + 0.5*self.v**2)*self.T ) \
               / (self.v * np.sqrt(self.T))
        
    def _d2(self, S: float) -> float:
        """
        Compute the quantity "d2" of Black-Scholes option pricing
        """
        return self._d1(S) - self.v * np.sqrt(self.T)
    
    def price(self, *argv) -> float:
        """
        Black-Scholes pricing model - Premium (Price)
        """
        try:
            S = argv[0]
        except:
            S = self.S
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return + S * np.exp(-self.q*self.T) * self.N(self._d1(S))  \
                       - self.K * np.exp(-self.r*self.T) * self.N(self._d2(S))
            else:
                # The Call has expired
                return max(S - self.K, 0)
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return - S * np.exp(-self.q*self.T) * self.N(-self._d1(S))  \
                       + self.K * np.exp(-self.r*self.T) * self.N(-self._d2(S))
            else:
                # The Put has expired
                return max(self.K - S, 0)
      
    def delta(self, *argv) -> float:
        """
        Black-Scholes pricing model - Delta
        """  
        try:
            S = argv[0]
        except:
            S = self.S 
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return np.exp(-self.q*self.T) * self.N(self._d1(S)) 
            else:
                # The Call has expired
                if self.price(S) > 0:
                    return +1
                else: 
                    return 0     
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return np.exp(-self.q*self.T) * self.N(self._d1(S)) - 1
            else:
                # The Put has expired
                if self.price(S) > 0:
                    return -1
                else:             
                    return 0

    def llambda(self, *argv) -> float:
        """
        Black-Scholes pricing model - Lambda
        """
        try:
            S = argv[0]
        except:
            S = self.S
        if self.CP == "C":
            # Call option
            if self.delta(S) < 1e-10 or self.price(S) < 1e-10:
                return +np.inf
            else:
                return self.delta(S) * self.S / self.price(S)
        else:
            # Put option 
            if self.delta(S) > -1e-10 or self.price(S) < 1e-10:
                return -np.inf
            else:
                return self.delta(S) * self.S / self.price(S)
                     
    def gamma(self, *argv) -> float:
        """
        Black-Scholes pricing model - Gamma 
        """  
        try:
            S = argv[0]
        except:
            S = self.S  
        # Gamma is the same for both Call and Put            
        if self.T > 0:
            # The Option has not expired yet
            return + np.exp(-self.q*self.T) * self.N(self._d1(S), cum=0) \
                   / (self.S * self.v * np.sqrt(self.T))
        else:
            # The Option has expired
            return 0
    
    def theta(self, *argv) -> float:
        """
        Black-Scholes pricing model - Theta
        """
        try:
            S = argv[0]
        except:
            S = self.S  
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return - np.exp(-self.q*self.T) * self.S * self.v * self.N(self._d1(S), cum=0) \
                       / (2*np.sqrt(self.T)) \
                       + self.q*np.exp(-self.q*self.T) * self.S * self.N(self._d1(S))   \
                       - self.r*np.exp(-self.r*self.T) * self.K * self.N(self._d2(S))
            else:
                # The Call has expired
                return 0    
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return - np.exp(-self.q*self.T) * self.S * self.v * self.N(self._d1(S), cum=0) \
                       / (2*np.sqrt(self.T)) \
                       - self.q*np.exp(-self.q*self.T) * self.S * (1 - self.N(self._d1(S)))   \
                       + self.r*np.exp(-self.r*self.T) * self.K * (1 - self.N(self._d2(S)))
            else:
                # The Put has expired
                return 0

    def vega(self, *argv) -> float:
        """
        Black-Scholes pricing model - Vega 
        """    
        try:
            S = argv[0]
        except:
            S = self.S  
        # Vega is the same for both Call and Put            
        if self.T > 0:
            # The Option has not expired yet
            return + np.exp(-self.q*self.T) * self.S * np.sqrt(self.T) \
                   * self.N(self._d1(S), cum=False) 
        else:
            # The Option has expired
            return 0          
            
    def greeks(
        self,
        grk: str or None = None, 
        rnd: int = 2
    ) -> dict:
        """
        Black-Scholes pricing model - All greeks (and price)
        """
        if grk is None:
            return {
                "Price": round(BSOption.llambda(self), rnd),
                "Lambda": round(BSOption.llambda(self), rnd),
                "Delta": round(BSOption.delta(self), rnd),
                "Gamma": round(BSOption.gamma(self), rnd),
                "Theta": round(BSOption.theta(self), rnd),
                "Vega": round(BSOption.vega(self), rnd)
            }
        if grk == "Price":
            return round(BSOption.price(self), rnd)
        elif grk == "Delta":
            return round(BSOption.delta(self), rnd)
        elif grk == "Gamma":
            return round(BSOption.gamma(self), rnd)
        elif grk == "Vega":
            return round(BSOption.vega(self), rnd)
        elif grk == "Theta":
            return round(BSOption.theta(self), rnd)
        elif grk == "Lambda":
            return round(BSOption.llambda(self), rnd)
        else:
            raise ValueError("Wrong input greek name")

    
    def underlying_set(
        self, 
        bnd: float = 0.4, 
        npr: int = 10,
        *argv
    ) -> list:
        """
        Generate a set of underlying prices lower and higher 
        than the current input underlying price.
        The limit is (-bnd,+bnd) of the input underlying price (in %).
        """
        try:
            S = argv[0]
        except:
            S = self.S
        Smin = S * (1 - bnd)
        Smax = S * (1 + bnd)
        SS = np.linspace(Smin,S,npr).tolist() + np.linspace(S,Smax,npr)[1:].tolist()
        return SS
    
    def oprices(self, ps: int = "P") -> pd.Series:
        """
        Generate a pd.Series of option prices or greenks using 
        for a generated underlying prices 
        """
        if ps == "P":
            name = "Price"
            ops = [self.price(s) for s in self.underlying_set()]
        elif ps == "D":
            name = "Delta"
            ops = [self.delta(s) for s in self.underlying_set()]
        ops = pd.Series(ops, index=self.underlying_set())
        ops.name = name
        return ops 