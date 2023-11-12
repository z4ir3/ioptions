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
    def _valid_underlying(S):
        '''
        Validate input underlying price
        '''
        if S > 0:
            return S
        else:
            raise ValueError("Argument 'S' (underlying price) must be greater than 0")
            
    @staticmethod 
    def _valid_strike(K):
        '''
        Validate input strike price
        '''
        if K > 0:
            return K
        else:
            raise ValueError("Argument 'K' (strike price) must be greater than 0")
    
    @staticmethod 
    def _valid_maturity(T):
        '''
        Validate input maturity
        '''
        if T >= 0:
            return T
        else:
            raise ValueError("Argument 'T' (maturity) cannot be negative")

    @staticmethod 
    def _valid_intrate(r):
        '''
        Validate input interest rate
        '''
        if r >= 0:
            return r
        else:
            raise ValueError("Argument 'r' (interest rate) cannot be negative")

    @staticmethod 
    def _valid_vola(v):
        '''
        Validate input volatility
        '''
        if v > 0:
            return v
        else:
            raise ValueError("Argument 'v' (volatility) must be greater than 0")
    
    @staticmethod 
    def _valid_yield(q):
        '''
        Validate input dividend yield
        '''
        if q >= 0:
            return q
        else:
            raise ValueError("Argument 'q' (dividend yield) cannot be negative")
       
    @property
    def params(self):
        """
        Returns all input option parameters
        """
        return {"type": self.CP,
                "S"   : self.S, 
                "K"   : self.K, 
                "T"   : self.T,
                "r"   : self.r,
                "v"   : self.v,
                "q"   : self.q}
     
    @staticmethod
    def N(x, cum=1):
        '''
        Standard Normal CDF or PDF evaluated at the input point x.
        '''  
        if cum:
            # Returns the standard normal CDF
            return norm.cdf(x, loc=0, scale=1)
        else:
            # Returns the standard normal PDF
            return norm.pdf(x, loc=0, scale=1)

    def d1(self, S):
        '''
        Compute the quantity d1 of Black-Scholes option pricing
        '''
        return ( np.log(S / self.K) + (self.r - self.q + 0.5*self.v**2)*self.T ) / (self.v * np.sqrt(self.T))
        
    def d2(self, S):
        '''
        Compute the quantity d2 of Black-Scholes option pricing
        '''
        return self.d1(S) - self.v * np.sqrt(self.T)
    
    def price(self, *argv):
        '''
        Black-Scholes pricing model - Premium (Price)
        '''
        try:
            S = argv[0]
        except:
            S = self.S
        
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return + S * np.exp(-self.q*self.T) * self.N(self.d1(S))  \
                       - self.K * np.exp(-self.r*self.T) * self.N(self.d2(S))
                       
            else:
                # The Call has expired
                return max(S - self.K, 0)
            
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return - S * np.exp(-self.q*self.T) * self.N(-self.d1(S))  \
                       + self.K * np.exp(-self.r*self.T) * self.N(-self.d2(S))
                       
            else:
                # The Put has expired
                return max(self.K - S, 0)
      
    def delta(self):
        '''
        Black-Scholes pricing model - Delta
        '''    
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return np.exp(-self.q*self.T) * self.N(self.d1()) 
            
            else:
                # The Call has expired
                if self.price() > 0:
                    return +1
                else: 
                    return 0
                
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return np.exp(-self.q*self.T) * self.N(self.d1()) - 1
            
            else:
                # The Put has expired
                if self.price() > 0:
                    return -1
                else:             
                    return 0

    def Lambda(self):
        '''
        Black-Scholes pricing model - Lambda
        '''
        if self.CP == "C":
            # Call option
            if self.delta() < 1e-10 or self.price() < 1e-10:
                return +np.inf
            else:
                return self.delta() * self.S / self.price()
        
        else:
            # Put option 
            if self.delta() > -1e-10 or self.price() < 1e-10:
                return -np.inf
            else:
                return self.delta() * self.S / self.price()
                     
    def gamma(self):
        '''
        Black-Scholes pricing model - Gamma 
        '''    
        # Gamma is the same for both Call and Put            
        if self.T > 0:
            # The Option has not expired yet
            return + np.exp(-self.q*self.T) * self.N(self.d1(), cum=0) / (self.S * self.v * np.sqrt(self.T))
            
        else:
            # The Option has expired
            return 0
    
    def theta(self):
        '''
        Black-Scholes pricing model - Theta
        '''
        if self.CP == "C":
            # Call Option
            if self.T > 0:
                # The Call has not expired yet
                return - np.exp(-self.q*self.T) * self.S * self.v * self.N(self.d1(), cum=0) / (2*np.sqrt(self.T)) \
                       + self.q*np.exp(-self.q*self.T) * self.S * self.N(self.d1())   \
                       - self.r*np.exp(-self.r*self.T) * self.K * self.N(self.d2())
            
            else:
                # The Call has expired
                return 0
            
        else: 
            # Put Option
            if self.T > 0:
                # The Put has not expired yet
                return - np.exp(-self.q*self.T) * self.S * self.v * self.N(self.d1(), cum=0) / (2*np.sqrt(self.T)) \
                       - self.q*np.exp(-self.q*self.T) * self.S * (1 - self.N(self.d1()))   \
                       + self.r*np.exp(-self.r*self.T) * self.K * (1 - self.N(self.d2()))
                       
            else:
                # The Put has expired
                return 0

    def vega(self):
        '''
        Black-Scholes pricing model - Vega 
        '''    
        # Vega is the same for both Call and Put            
        if self.T > 0:
            # The Option has not expired yet
            return +np.exp(-self.q*self.T) * self.S * np.sqrt(self.T) * self.N(self.d1(), cum=0) 
            
        else:
            # The Option has expired
            return 0          
            
    def greeks(self):
        '''
        Black-Scholes pricing model - All greeks
        '''   
        return {"Lambda": np.round( BSOption.Lambda(self), 2),
                "Delta" : np.round( BSOption.delta(self),  2),
                "Gamma" : np.round( BSOption.gamma(self),  2),
                "Theta" : np.round( BSOption.theta(self),  2),
                "Vega"  : np.round( BSOption.vega(self) ,  2)}
    
    def underlying_set(self, *argv):
        '''
        Generate a set of underlying prices lower and higher than the current input underlying price
        The limit is currently (-40%,+40%) of the input underlying price
        '''
        try:
            S = argv[0]
        except:
            S = self.S
        Smin = S * (1 - 0.40)
        Smax = S * (1 + 0.40)
        return list(np.linspace(Smin,S,100)[:-1]) + list(np.linspace(S,Smax,100))
        
    def setprices(self):
        '''
        Generate a pd.Series of option prices using the set of the generated underlying prices 
        '''
        oprices = [self.price(p) for p in self.underlying_set()]
        oprices = pd.Series(oprices, index=self.underlying_set())
        return oprices 