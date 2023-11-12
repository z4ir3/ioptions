'''
Black-Scholes option pricing strategy class
'''

import pandas as pd
# import numpy as np
from scipy.stats import norm

from models.blackscholes import BSOption


class BSOptStrat:
    """
    Class for generating option strategies with the Black-Scholes model
    """
    def __init__(
        self, 
        S: float = 100, 
        r: float = 0.02, 
        q: float = 0
    ):
        """
        S : Underlying Price
        r : Risk-free interest rate
        q : Dividend yield
        """
        self.S = S
        self.r = r
        self.q = q
        self.instruments    = [] 
        self.payoffs        = BSOptStrat.init_payoffs(S)
        self.payoffs_exp    = BSOptStrat.init_payoffs(S)
        self.payoffs_exp_df = pd.DataFrame()
    

    def init_payoffs(S):
        '''
        Generating a set of underlying prices from the given input current underlying price S
        '''
        ss = pd.Series([0] * len(BSOption.underlying_set(0,S)), index = BSOption.underlying_set(0,S))
        return ss
 
        
    def call(self, NP=+1, K=100, T=0.25, v=0.30, M=100, optprice=None):
        '''
        Creating a Call Option 
        - NP : Net Position, >0 for long positions, <0 for short positions 
        - K  : Strike price
        - T  : Time-to-Maturity in years 
        - v  : Volatility
        - M  : Multiplier of the Option (number of stocks allowed to buy/sell)
        '''
        # Create Call Option with current data
        option = BSOption("C", self.S, K, T, self.r, v, q=self.q)
        
        # Call payoff before expiration (T>0)
        if optprice is not None:
            call_price = optprice
        else:
            call_price = option.price()
        
        # Generating the set of payoff for at different underlying prices 
        # Here, option.setprices() are the prices of the call (i.e., as if it were long)
        # - If NP > 0, the price must be paid, then:
        #   payoffs = option.setprices() * NP * M  - Call price * NP * M
        # - If NP < 0, the price is to be received, then:
        #   payoffs = Call price * abs(NP) * M - option.setprices() * abs(NP) * M 
        #           = option.setprices() * NP * M  - Call price * NP * M
        # Summary: ( option.setprices() - call_price ) * NP * M
        payoffs = ( option.setprices() - call_price ) * NP * M

        # Update strategy instruments with current instrument data
        self.update_strategy("C", call_price, NP, K, T, v, M, payoffs)
        
        # Update strategy with current instrument data at maturity 
        self.option_at_exp("C", call_price, NP, K, v, M)
            

    def put(self, NP=+1, K=100, T=0.25, v=0.30, M=100, optprice=None):
        '''
        Creating a Put Option 
        - NP : Net Position, >0 for long positions, <0 for short positions 
        - K  : Strike price
        - T  : Time-to-Maturity in years 
        - v  : Volatility
        - M  : Multiplier of the Option (number of stocks allowed to buy/sell)
        '''
        # Create Put Option with current data
        option = BSOption("P", self.S, K, T, self.r, v, q=self.q)
        
        # Call payoff before expiration (T>0)
        if optprice is not None:
            put_price = optprice
        else:
            put_price = option.price()
        
        # Generating the set of payoff for at different underlying prices 
        payoffs = ( option.setprices() - put_price) * NP * M
            
        # Update strategy instruments with current instrument data
        self.update_strategy("P", put_price, NP, K, T, v, M, payoffs)
        
        # Update strategy with current instrument data at maturity 
        self.option_at_exp("P", put_price, NP, K, v, M)


    def update_strategy(self, CP, price, NP, K, T, v, M, payoffs):
        '''
        Updates the current payoffs of the option strategy as soon as that a new option is inserted. 
        The current list of instruments composing the strategy is also updated        
        
        New input(s): 
        - price   : price of the input option 
        - payoffs : Payoff of the input option (for a set of given underlying prices)
        '''
        # Update current payoff strategy with new instrument payoff 
        self.update_payoffs(payoffs, T=T)

        # Create a dictionary with the data of the given option  
        inst = {"CP": CP,
                "NP": NP,
                "K" : K,
                "T" : T,
                "v" : v,
                "M" : M,
                "Pr": round(price,2)}
        
        # Concat new instrument to total strategy instrument list
        self.instruments.append(inst)


    def update_payoffs(self, payoffs, T=0):
        '''
        Update current payoff strategy with new instrument payoff.
        It updates either the payoff for T>0 or at maturity for T=0
        '''
        if T > 0:
            self.payoffs     = payoffs + self.payoffs
        else:
            self.payoffs_exp = payoffs + self.payoffs_exp
        
 
    def option_at_exp(self, CP, price, NP, K, v, M):
        '''
        Calculates the payoff of the option at maturity (T=0)
        '''
        # Create Option at maturity (calling class with T = 0) 
        option = BSOption(CP, self.S, K, 0, self.r, v, q=self.q)

        # Option payoff at maturity: 
        # - Call: (max(S - K;0) - C) * NP * M
        # - Put:  (P - max(S - K;0)) * NP * M
        payoffs_exp = ( option.setprices() - price ) * NP * M        

        # Update the dataframe of payoff at maturity of single options with the new current inserted option 
        self.update_payoffs_exp_df(payoffs_exp)

        # Update the strategy payoff at maturity with the one of the new current inserted option 
        self.update_payoffs(payoffs_exp, T=0)


    def update_payoffs_exp_df(self, payoffs_exp):
        '''
        Update the dataframe of payoff at maturity of single options with the new current inserted option 
        '''
        # Concat new option payoff with current dataframe
        self.payoffs_exp_df = pd.concat([self.payoffs_exp_df, pd.DataFrame(payoffs_exp)], axis=1)

        # Update columns 
        self.payoffs_exp_df.columns = [n for n in range(1,self.payoffs_exp_df.shape[1]+1)]
                
        
    def describe_strategy(self): #, stratname=None):
        '''
        This method can be called once the option has been set.
        Here, all option data saved so far in the list of instrument are now saved 
        in a dictionary and the cost of entering the strategy is also computed 
        '''       
        # Create dictionary of options inserted in the strategy 
        StratData = dict()
        
        stratcost = 0
        for n, o in enumerate(self.instruments):
        
            # Key of the dictionay (Option number) 
            StratData["Option_{}".format(n+1)] = o
            
            # Compute total strategy cost: sum of NLVs (Net Liquidation Value) of the option
            # NLV = price * net position * multiplier
            stratcost = stratcost + o["Pr"] * o["NP"] * o["M"]
            
        # Save the strategy cost
        StratData["Cost"] = stratcost

        return StratData
        
        
    def get_payoffs_exp_df(self):
        '''
        Returns a dataframe with the payoff at maturity of the strategy's option
        '''
        return self.payoffs_exp_df


    def get_payoffs(self):
        '''
        Returns the current strategy payoff
        '''
        return self.payoffs

    
    def get_payoffs_exp(self):
        '''
        Returns the strategy payoff at maturity 
        '''
        return self.payoffs_exp