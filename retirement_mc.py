#!/usr/bin/env python
"""

Use Monte Carlo to estimate the probability of outliving
one's savings in retirement.

Uses historical economic data to estimate investment return,
and CDC life tables to estimate chance of death.



"""

import numpy as np
from scipy.optimize import brentq
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp

import matplotlib.pyplot as plt
from matplotlib import rcParams

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

from cdc_life_tables import life_table
import shiller

# Historical financial data
inflation = shiller.inflation.iloc[1:-1]
stock_returns = shiller.stock_returns.iloc[1:-1]
interest_rates = shiller.interest_rates.iloc[1:-1]

rand = np.random.random_sample


def run_histories(starting_assets, 
                  yearly_expense,
                  stock_fraction,
                  starting_age,
                  state_abbrev,
                  demographic_group,
                  n_mc=1000, plotting=False, verbose=False):
    """
    Run a Monte Carlo simulation for a person starting with the given
    amount of assets in savings. The yearly_expense are withdrawn 
    once a year, while the assets grow according to historical
    US stock and bond returns, dividing between the two according to
    stock_fraction. The chance of dying each year is taken from 
    life tables.

    Sources of data:
      
       * Shiller's historical economic data
       * CDC's life tables

    Inputs:
      
       * starting_assets : amount of initial savings to invest for income
       * yearly_expense  : amount of money needed per year. This value
                            will be adjusted for inflation.
       * stock_fraction : fraction (between 0.0 and 1.0) of the money invested
                            in stocks. The remainder is invested in bonds.
       * starting_age : the subject's age at which yearly withdraws will be made from 
                            the investment
       * state_abbrev : mailing abbreviation for the state in which the subject lives
       * demographic_group : the subject's demographic group accepted by
                               cdc_life_tables.life_table
       * n_mc : the number of Monte Carlo histories
       * plotting : produce a plot showing the Monte Carlo histories
       * verbose : produce verbose diagnostic messages

    Output:
       * probability of running out of money

    """

    # Life table
    table = life_table(state_abbrev, demographic_group)

    mc_histories = []

    for i in range(n_mc):

        age = starting_age
        current_assets = starting_assets
        expenses_per_year = yearly_expense

        assets = [current_assets]

        # Loop over years
        while current_assets > 0:

            # Death this year.
            if age >= 110 or rand() <= table[int(age)]:
                # Die at random point in year
                current_assets -= expenses_per_year*rand()
                break

            # Subtracting expenses for year
            current_assets -= expenses_per_year

            # Pick past year by random to base inflation, stock return data
            i = np.random.randint(inflation.size, size=1)
            i = int(i)

            # Adjust expenses for inflation.
            expenses_per_year *= 1.0+inflation.iloc[i]

            # Adding stock investment increase
            stock_gains = stock_returns.iloc[i] * (current_assets*stock_fraction)

            # Adding bond investment increase
            bond_gains = interest_rates.iloc[i] * (current_assets*(1-stock_fraction))

            current_assets += stock_gains
            current_assets += bond_gains

            # Saving current assets
            assets.append(current_assets)

            # Getting old
            age += 1.0


        assets = np.array(assets)

        mc_histories.append( (assets) )


    if plotting:

        rcParams['figure.figsize'] = [7.0, 3.5]

        # Plot of asset-over-lifetime histories
        plt.figure()

        final_ages = []
        final_assets = []
        for i in range(n_mc):
            y = mc_histories[i] / 1e6
            x = np.arange(starting_age, starting_age+y.size)
            plt.plot(x, y, color='gray', linewidth=0.5)

            final_assets.append(y[-1])
            final_ages.append(x[-1])

        plt.plot(final_ages, final_assets, color='red', ls='.',
                 marker='.', markersize=1.5)

        plt.xlabel('Age')
        plt.ylabel('Remaining assets (million USD)')

        # plt.savefig('figs/histories.pdf')


        # Plot of age of death
        rcParams['figure.subplot.left'] = 0.15
        plt.figure()

        final_ages = np.array(final_ages)

        max_age = 110
        bins = np.linspace(int(starting_age)-0.5, max_age+0.5, max_age-starting_age+2)

        plt.hist(final_ages, bins=bins, normed=True)

        plt.xlabel('Age of Death')
        plt.ylabel('Probability')

        # plt.savefig('figs/final-age.pdf')

    else:
        final_assets = []
        for i in range(n_mc):
            final_assets.append(mc_histories[i][-1])

    final_assets = np.array(final_assets)

    run_out_of_money_hist = np.array(final_assets < 0.0, dtype=np.float64)
    run_out_of_money = unc.ufloat(run_out_of_money_hist.mean(),
                                  run_out_of_money_hist.std()/np.sqrt(n_mc))

    if verbose:
        print ' Chance of running out of money is {:%}'.format(run_out_of_money)

    return run_out_of_money


def how_much_to_save(
                     acceptable_risk=0.01,
                     yearly_expense=40e3,
                     stock_fraction=0.5,
                     starting_age=65,
                     state_abbrev='CA',
                     demographic_group='total',
                     n_mc=500, plotting=False, verbose=False):
    """
    Computes f(x) = f_0, where f is the MC simulation of the retirement
    process returning the probability of running out of money and
    x is the size of the starting assets.

    Inputs:
      
       * yearly_expense : amount of money needed per year. This value
                            will be adjusted for inflation.
       * stock_fraction : fraction (between 0.0 and 1.0) of the money invested
                            in stocks. The remainder is invested in bonds.
       * starting_age : the subject's age at which yearly withdraws will be made from 
                            the investment
       * state_abbrev : mailing abbreviation for the state in which the subject lives
       * demographic_group : the subject's demographic group accepted by
                               cdc_life_tables.life_table
       * acceptable_risk : probability of running out of money
       * n_mc : the number of Monte Carlo histories
       * plotting : produce a plot showing the Monte Carlo histories
       * verbose : produce verbose diagnostic messages

    Output:

       * starting_assets : amount of initial savings to invest for income

    """

    def f(x):
        prob_outlive_savings = run_histories(x, yearly_expense, stock_fraction,
                                             starting_age, state_abbrev,
                                             demographic_group,
                                             n_mc=n_mc, plotting=False, verbose=False)
        return acceptable_risk - prob_outlive_savings.nominal_value

    lo_bound = 5.0*yearly_expense
    hi_bound = 40.0*yearly_expense

    while True:
        try:
            res = brentq(f, lo_bound, hi_bound, rtol=1e-2, full_output=True)
            break
        except ValueError:
            n_mc *= 2
            lo_bound /= 2
            hi_bound *= 2

    return res[0]




def cascade_plot(yearly_expense,
                 stock_fraction,
                 starting_age,
                 state_abbrev,
                 demographic_group,
                 stock_fractions = [0.25, 0.5, 0.75],
                 n_mc=5000):
    """
    Inputs:
      
       * yearly_expense : amount of money needed per year. This value
                            will be adjusted for inflation.
       * stock_fraction : fraction (between 0.0 and 1.0) of the money invested
                            in stocks. The remainder is invested in bonds.
       * starting_age : the subject's age at which yearly withdraws will be made from 
                            the investment
       * state_abbrev : mailing abbreviation for the state in which the subject lives
       * demographic_group : the subject's demographic group accepted by
                               cdc_life_tables.life_table
       * n_mc : the number of Monte Carlo histories

    Output:
       * Matplotlib figure object

    """
    rcParams['figure.figsize'] = [9, 5]
    fig = plt.figure()

    starting_assets = np.linspace(1e5, 10e6, 100)
    starting_assets = np.array(starting_assets)

    for stock_fraction in stock_fractions:

        run_out_of_money = []
        for x in starting_assets:
            p = 100*run_histories(x, 
                                  yearly_expense,
                                  stock_fraction,
                                  starting_age,
                                  state_abbrev,
                                  demographic_group,
                                  n_mc=n_mc)
            run_out_of_money.append(p)

            # Don't keep going for probability <1%
            if p < 1: break

        run_out_of_money = np.array(run_out_of_money)
        n = run_out_of_money.size

        plt.errorbar(starting_assets[:n]/1e6, unp.nominal_values(run_out_of_money),
                     yerr=unp.std_devs(run_out_of_money), 
                     capsize=0.0, marker='.', markersize=3.5, ls=next(linecycler),
                     label='{:.0%} stocks'.format(stock_fraction))

    plt.xlabel('Starting Assets (million USD)')
    plt.ylabel('Prob. of running out of money (%)')

    str_id = '{}-{}-{}-{}'.format(demographic_group, state_abbrev, starting_age,
                                                yearly_expense)
    plt.title('{}-{}, starting at age {} with \${}/year expenses'.format(demographic_group, state_abbrev, starting_age,
                                                yearly_expense))

    plt.legend(fontsize='x-small')
    plt.ylim(ymin=0, ymax=100)

    # plt.savefig('figs/{}.pdf'.format(str_id))

    return fig


def sensitivity_plots(
                      state_abbrev='CA',
                      demographic_group='total',
                      yearly_expense=40e3,
                      yearly_expenses=np.logspace(3.69897, 5, 10),
                      starting_age=65,
                      starting_ages=np.linspace(40, 85, 10),
                      acceptable_risk=0.02,
                      acceptable_risks=np.logspace(-3, -0.2, 7),
                      stock_fraction=0.5,
                      stock_fractions=np.linspace(0.0, 1.0, 11),
                      n_mc=5000,
                      verbose=False):
    """
    Inputs:
      
       * yearly_expense : amount of money needed per year. This value
                            will be adjusted for inflation.
       * stock_fraction : fraction (between 0.0 and 1.0) of the money invested
                            in stocks. The remainder is invested in bonds.
       * starting_age : the subject's age at which yearly withdraws will be made from 
                            the investment
       * state_abbrev : mailing abbreviation for the state in which the subject lives
       * demographic_group : the subject's demographic group accepted by
                               cdc_life_tables.life_table
       * n_mc : the number of Monte Carlo histories
       * plotting : produce a plot showing the Monte Carlo histories
       * verbose : produce verbose diagnostic messages

    Output:
       * Matplotlib figure object

    """

    factors = { 
                'stock_fraction'  : {'value' : stock_fraction,  'values' : stock_fractions },
                'acceptable_risk' : {'value' : acceptable_risk, 'values' : acceptable_risks },
                 'yearly_expense' : {'value' : yearly_expense,  'values' : yearly_expenses },
                   'starting_age' : {'value' : starting_age,    'values' : starting_ages },
              }

    rcParams['figure.figsize'] = [9, 11]
    fig, axs = plt.subplots(nrows=len(factors.keys()), sharey=True)


    base_opts = {
                  'stock_fraction'    : stock_fraction,
                  'acceptable_risk'   : acceptable_risk,
                  'yearly_expense'    : yearly_expense,
                  'starting_age'      : starting_age,
                  'state_abbrev'      : state_abbrev,
                  'demographic_group' : demographic_group,
                }

    base_save = how_much_to_save(**base_opts)/1e6

    for i, factor in enumerate(factors.keys()):

        opts = base_opts.copy()

        factor_res = []

        for factor_value in factors[factor]['values']:

            opts[factor] = factor_value

            factor_res.append( how_much_to_save(**opts)/1e6 )


        axs[i].plot(factors[factor]['values'], factor_res,
                    marker='.', markersize=3.5, ls='-', color='gray')

        axs[i].plot(base_opts[factor], base_save,
                    marker='o', markersize=5.5, color='black')
                 
        axs[i].set_xlabel(factor)

    axs[1].set_ylabel('Amount to save (million USD)')
    fig.tight_layout()

    if verbose:
        print ' You should save ${:.2f} million.'.format(base_save)

    #fig.savefig('figs/{}.pdf'.format('sensitivity-plots'))

    return fig



if __name__ == '__main__':
    starting_age = 65.0
    state_abbrev = 'IA'
    demographic_group = 'wf'

    # Expenses per year
    yearly_expense = 50e3

    # Assets
    starting_assets = 3e6

    # Investment allocation
    stock_fraction = 0.5

    # Run one simulation for a given starting assets and stock fraction
    run_histories(starting_assets, yearly_expense,
                  stock_fraction,
                  starting_age,
                  state_abbrev,
                  demographic_group,
                  n_mc=5000, plotting=True, verbose=True)

    # The highest probability of running out of money 
    #  that you are comfortable with.
    acceptable_risk = 0.01  # = 1%

    savings_goal = how_much_to_save(acceptable_risk, 
                                    yearly_expense,
                                    stock_fraction,
                                    starting_age,
                                    state_abbrev,
                                    demographic_group)

    # Run many simulations over a range of input variables.
    sens_fig = sensitivity_plots(acceptable_risk=acceptable_risk, 
                                 yearly_expense=yearly_expense,
                                 stock_fraction=stock_fraction,
                                 starting_age=starting_age,
                                 state_abbrev=state_abbrev,
                                 demographic_group=demographic_group
                                )
                     
    '''
    # Run many simulations over a range of starting assets and stock fractions
    cascade_plot(yearly_expense,
                 stock_fraction,
                 starting_age,
                 state_abbrev,
                 demographic_group)
    '''
