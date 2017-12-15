# Monte Carlo simulations for retirement

This software provides a convenient means to perform
a Monte Carlo simulation on the question of whether 
savings for retirement will actually last through 
the entire retirement. This is also relevant for how 
much life insurance one should carry for the purposes
of income replacement.

The idea is to use historical economic data to estimate the
returns on investment, and use life tables to estimate life
expectancy. Inflation is also accounted for by increasing the
yearly expenses by the rate of inflation.

An IPython notebook with an introductory example can be viewed on 
[nbviewer](http://nbviewer.ipython.org/github/jhykes/retirement-mc/blob/master/retirement_mc.ipynb).

## Dependencies

   * matplotlib
   * numpy
   * scipy
   * pandas
   * uncertainties
   * xlrd

## License

This software is released under the 3-clause BSD license.
See LICENSE file.

## Companion website

For those not proficient in Python or programming in general, I 
have also coded much of the same in Javascript, 
which can be accessed through an HTML form at the following website:

www.willioutlivemysavings.net
